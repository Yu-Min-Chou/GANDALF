from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import asyncio
import signal
import pickle
import argparse, sys
import numpy as np
import multiprocessing
from absl import app
from absl import flags
from tensorflow import keras
from tensorflow.python.ops import control_flow_util
from nats.aio.client import Client as NATS
from diffprivlib.mechanisms.gaussian import Gaussian

from model import resnet_models
import config

keras.backend.clear_session()
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True

def train(C_weight, ID_tmp, ID):
    try:
        import tensorflow as tf
        from tensorflow.python.keras import layers
        os.environ["CUDA_VISIBLE_DEVICES"] = str(ID_tmp % config.n_gpus)
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        data_augmentation = tf.keras.Sequential([
                layers.ZeroPadding2D(padding=2),
                layers.RandomCrop(32, 32),
                layers.RandomRotation(0.1)
            ])
        data_augmentation.build(input_shape=(None, config.image_size, config.image_size, config.channel_size))

        # load augmented_dataset from disk
        path = config.original_dataset_path + str(ID)
            
        dataset = tf.data.experimental.load(path)
        dataset = dataset.shuffle(50000, reshuffle_each_iteration=True)
        dataset = dataset.batch(config.batch_size, drop_remainder=True).prefetch(AUTOTUNE)

        resnet = resnet_models.Resnet()
        resnet.set_model_weight(C_weight)

        c_loss_result, c_accu_result = resnet.train(dataset)

        return resnet.get_model_weight(), c_loss_result, c_accu_result
        
    except Exception as e:
        print('train: ', e)

class Client_cb():
    def __init__(self, loop, nc, socket):
        self.loop = loop
        self.nc = nc
        self.socket = socket
        self.dataset = None
        self.ID = -1
        self.group_ID = -1
        self.dataset_dist = None
        self.dataset_size = None
        self.gpu_used = False
        self.C_weight = None
        self.c_loss_result = None
        self.c_accu_result = None
        self.ID_tmp = -1
        

    def get_id(self):
        return self.ID

    def get_dist(self):
        DP = Gaussian(epsilon=config.dist_epsilon, delta=config.dist_delta, sensitivity=config.dist_sensitivity)
        randomise_dataset_dist = np.zeros(self.dataset_dist.shape)
        for i in range(len(randomise_dataset_dist)):
            randomise_dataset_dist[i] = DP.randomise(self.dataset_dist[i])
            if(randomise_dataset_dist[i] > 1.0):
                randomise_dataset_dist[i] = 1.0
            elif(randomise_dataset_dist[i] < 0.0):
                randomise_dataset_dist[i] = 0.0
        return randomise_dataset_dist

    def load_datainfo(self):
        inputfile = config.datainfo_path
        fw = open(inputfile, 'rb')
        loaded = pickle.load(fw)
        dataset_dist = loaded['imbalance_dist']
        num_data = loaded['num_data']

        self.dataset_dist = dataset_dist[self.ID] / np.sum(dataset_dist[self.ID])
        self.dataset_size = np.sum(dataset_dist[self.ID] * num_data).astype(int)
        print(self.dataset_dist)
        print(self.dataset_size)

    async def create_training_process(self):
        try:
            pool = multiprocessing.get_context('spawn').Pool(1)
            result = pool.starmap(train, [(self.C_weight, self.ID_tmp, self.ID)])
            self.C_weight = result[0][0]
            self.c_loss_result = result[0][1]
            self.c_accu_result = result[0][2]
            pool.close()
            pool.join()

        except Exception as e:
            print('create_training_process: ', e)

    async def gpu_used_cb(self, msg):
        try:
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            finished_ID = loaded['ID']
            if((self.ID_tmp - config.n_gpus) == finished_ID):
                self.gpu_used = True

        except Exception as e:
            print(e)
        
    async def request_id_cb(self, msg):
        # receive a message from [request_id-reply]
        # message is a int, ID of this client
        try:
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            if(loaded['message'] == 'success'):
                print("get ID: ", loaded['ID'])
                self.ID = loaded['ID']
                self.load_datainfo()

                message_bytes = pickle.dumps({'ID': self.ID, 'dataset_size': self.dataset_size})
                await self.socket.request('request_datainfo', message_bytes, self.request_datainfo_cb)
            else:
                print("Amount of clients has been reached the upper limit")
                await self.socket.terminate()
                await asyncio.sleep(5)
        except Exception as e:
            print(e)

    async def request_datainfo_cb(self, msg):
        try:
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            if(loaded['message'] == 'success'):
                print("client {}: request_info success".format(self.ID))
            else:
                print("client {}: request_info fail".format(self.ID))

        except Exception as e:
            print(e)

    async def train_cb(self, msg):
        try:
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            C_weight = loaded['C_weight']
            pool = loaded['pool']

            self.C_weight = C_weight
            self.gpu_used = False

            if self.ID in pool:
                self.ID_tmp = pool.index(self.ID)
                print("client id: {}, ID_tmp: {}".format(self.ID, self.ID_tmp))
            else:
                return

            await asyncio.sleep(1)

            if(self.ID_tmp < config.n_gpus):
                self.gpu_used = True
            
            while(not(self.gpu_used)):
                await asyncio.sleep(2)

            print('client{} receive weight and get a gpu, starting training...'.format(self.ID))
            
            task = self.loop.create_task(self.create_training_process())
            await task

            await asyncio.sleep(5)

            self.gpu_used = False
            target_subject = 'gpu_used'
            message_bytes = pickle.dumps({'ID': self.ID_tmp})
            await self.socket.publish(target_subject, message_bytes)

            target_subject = 'train_client_to_server'
            message_send = 'train_client_to_server'
            message_bytes = pickle.dumps({'message': message_send, 'C_weight': self.C_weight, 'ID': self.ID})
            await self.socket.publish(target_subject, message_bytes)

        except Exception as e:
            print('train_cb', e)

    async def terminate_process_cb(self, msg):
        try:
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            message = loaded['message']
            print(message)

            await self.socket.terminate()

        except Exception as e:
            print('terminate_process_cb: ', e)
