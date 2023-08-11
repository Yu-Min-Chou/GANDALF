from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import asyncio
import utils
import signal
import pickle
import argparse, sys
import numpy as np
import multiprocessing
from absl import app
from nats.aio.client import Client as NATS
from diffprivlib.mechanisms.gaussian import Gaussian

import config

def train(G_Weight, D_weight, ID, group_ID, num_data, round_times):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(group_ID)

        import tensorflow as tf
        from model import models
        AUTOTUNE = tf.data.experimental.AUTOTUNE

        # load dataset from disk
        path = config.ds_path + str(ID) 
        dataset = tf.data.experimental.load(path)

        dataset = dataset.shuffle(50000, reshuffle_each_iteration=True)
        dataset = dataset.batch(config.batch_size, drop_remainder=True).prefetch(AUTOTUNE)

        # initialize model and weight
        wgangp = models.WGANGP(num_data)
        wgangp.set_models_weight(G_Weight, D_weight)

        # training
        g_loss_result, d_loss_result = wgangp.train(dataset=dataset, group_ID=group_ID, round_times=round_times)

        return wgangp.get_models_weight(), g_loss_result, d_loss_result
    except Exception as e:
        print('Train: ', e)

class Client_cb():
    def __init__(self, loop, nc, socket):
        self.loop = loop
        self.nc = nc
        self.socket = socket
        self.dataset = None
        self.ID = -1
        self.group_ID = -1
        self.num_data = None
        self.training_result = None # 0 for weight, 1 for g_loss, 2 for d_loss

    def get_id(self):
        return self.ID

    def get_group_id(self):
        return self.group_ID

    def set_num_data(self, num_data):
        self.num_data = num_data

    async def create_training_process(self, G_weight, D_weight, round_times):
        try:
            with multiprocessing.get_context('spawn').Pool(1) as pool:
                self.training_result = pool.starmap(train, [(G_weight, D_weight, self.ID, self.group_ID, self.num_data, round_times)])
        except Exception as e:
            print("create_training_process: ", e)
        

    async def request_id_cb(self, msg):
        # receive a message from [request_id-reply]
        # message is a int, ID of this client
        try:
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            if(loaded['message'] == 'success'):
                print("get ID: ", loaded['ID'])
                self.ID = loaded['ID']
            else:
                print("Amount of clients has been reached the upper limit")
                await self.socket.terminate()

        except Exception as e:
            print(e)
    
    async def request_datainfo_cb(self, msg):
        # receive a message from [request_info-reply]
        # message is a int, 0: success, 1: fail
        try:
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            if(loaded['message'] == 'success'):
                print("client {}: request_info success".format(self.ID))
            else:
                print("client {}: request_info fail".format(self.ID))

        except Exception as e:
            print(e)

    async def publish_groupinfo_cb(self, msg):
        # receive a message from [request_groupinfo]
        # message is a list, recoding which group clients belong
        try:
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            group_info = loaded["group_info"]
            dataset_ids = loaded["dataset_ids"]
            for i in range(len(dataset_ids)):
                if(dataset_ids[i] == self.ID):
                    self.group_ID = group_info[i]
                    print("Client{}, group ID: {}".format(self.ID, self.group_ID))
                    break

        except Exception as e:
            print(e)

    async def train_cb(self, msg):
        try:
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            message = loaded['message']
            G_weight = loaded['G_weight']
            D_weight = loaded['D_weight']
            round_times = loaded['round']

            print("\nClient{}: start training\n".format(self.ID))

            train_task = self.loop.create_task(self.create_training_process(G_weight, D_weight, round_times))
            await train_task

            message_send = 'from client{}, who is next one'.format(self.ID)
            g_loss = self.training_result[0][1]
            d_loss = self.training_result[0][2]
            message_bytes = pickle.dumps({'message': message_send, 'g_loss': g_loss, 'd_loss': d_loss})

            target_subject = 'request_pass_weight_{}'.format(self.group_ID)
            await self.socket.request(target_subject, message_bytes, self.request_pass_weight_cb)

        except Exception as e:
            print("train_cb: ", e)

    async def request_pass_weight_cb(self, msg):
        try:
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            message = loaded['message']
            next_client_id = loaded['next_client_id']
            round_times = loaded['round']

            message_send = 'from client{}'.format(self.ID)
            G_weight, D_weight = self.training_result[0][0]

            # DP for model weight
            DP = Gaussian(epsilon=config.weight_epsilon, delta=config.weight_delta, sensitivity=config.weight_sensitivity)
            np_G_weight = np.asarray(G_weight, dtype=object)
            for weight_list in np_G_weight:
                for x in np.nditer(weight_list, flags=['refs_ok'], op_flags=['readwrite']):
                    x[...] = np.array(DP.randomise(float(x)))
            G_weight = np_G_weight.tolist()
            
            np_D_weight = np.asarray(D_weight, dtype=object)
            for weight_list in np_D_weight:
                for x in np.nditer(weight_list, flags=['refs_ok'], op_flags=['readwrite']):
                    x[...] = np.array(DP.randomise(float(x)))
            D_weight = np_D_weight.tolist()

            weights_bytes = pickle.dumps({'message': message_send, 'G_weight': G_weight, 'D_weight': D_weight, 'round': round_times})

            if(message == 'client'):
                target_subject = 'train_client{}'.format(next_client_id)
            elif(message == 'mediator'):
                target_subject = 'train_mediator{}'.format(self.group_ID)

            await self.socket.publish(target_subject, weights_bytes)

        except Exception as e:
            print("request_pass_weight: ", e)

    async def request_check_network_cb(self, msg):
        try:
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            message = loaded['message']

            message_send = 'success'
            message_bytes = pickle.dumps({'message': message_send})
            await self.socket.publish(msg.reply, message_bytes)

        except Exception as e:
            print(e)

    async def after_pass_weight_cb(self, msg):
        try:
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            message = loaded['message']

        except Exception as e:
            print(e)

    async def terminate_cb(self, msg):
        try:
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            message = loaded['message']

            print(message)

            await self.socket.terminate()

        except Exception as e:
            print(e)