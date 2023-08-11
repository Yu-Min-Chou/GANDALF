import os
import math
import asyncio
import pickle
import random
import numpy as np
import multiprocessing
from nats.aio.client import Client as NATS

import config
from mq import comm
from model import resnet_models
from model import datasetpipeline

import heapq

def get_init_weight(ID):
    os.environ["CUDA_VISIBLE_DEVICES"]= str(ID)

    resnet = resnet_models.Resnet()
    C_weight = resnet.get_model_weight()

    return C_weight

def testing(ID, C_weight):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"]= str(ID)

        pipeline = datasetpipeline.DatasetPipeline()
        _, dataset_test = pipeline.load_dataset()

        resnet = resnet_models.Resnet()
        resnet.set_model_weight(C_weight)

        c_accu_result = resnet.test(dataset_test)

        return c_accu_result

    except Exception as e:
        print('testing: ', e)

class Server_cb():
    def __init__(self, loop, nc, socket):
        self.loop = loop
        self.nc = nc
        self.socket = socket
        self.n_clients = config.n_clients
        self.n_clients_per_round = config.n_clients_per_round
        self.client_pool = list(range(self.n_clients))
        self.client_id = 0
        self.n_reg_client = 0
        self.round = 0
        self.C_weights = []
        self.C_weight = None
        self.dataset_sizes = [None] * self.n_clients
        self.n_return_client = 0
        self.return_id = []

        self.fw = open('accu.txt', 'a')

    async def init_weight(self):
        try:
            pool = multiprocessing.get_context('spawn').Pool(1)
            result = pool.map(get_init_weight, [0])
            self.C_weight = result[0]
            pool.close()
            pool.join()

        except Exception as e:
            print('init_weight: ', e)

    async def create_testing_process(self):
        try:
            pool = multiprocessing.get_context('spawn').Pool(1)
            result = pool.starmap(testing, [(0, self.C_weight)])
            c_accu_result = result[0]
            pool.close()
            pool.join()

            self.fw.write('%f\n' % c_accu_result)
            print('\nEpoch: {} - accuracy: {:.2f}%'.format(self.round, c_accu_result))

        except Exception as e:
            print('create_testing_process: ', e)

    async def request_id_cb(self, msg):
        # receive message from [request_id]
        # message will be a string
        try:
            subject = msg.subject
            data = msg.data.decode()
            print("receive message: ", data)
            if(data[:6] == 'client'):
                if(self.client_id < self.n_clients):
                    send_message = pickle.dumps({'message': 'success', 'ID': self.client_id})
                    await self.nc.publish(msg.reply, send_message)
                    print("Send Client ID: ", self.client_id)
                    self.client_id += 1
                else:
                    send_message = pickle.dumps({'message': 'fail'})
                    await self.nc.publish(msg.reply, send_message)
            else:
                print("Unknown message in request_id_cb: ", data)
        except Exception as e:
            print('request_id: ', e)

    async def request_datainfo_cb(self, msg):
        try:
            send_message = pickle.dumps({'message': 'success'})
            await self.nc.publish(msg.reply, send_message)
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            self.dataset_sizes[loaded['ID']] = loaded["dataset_size"]

            if((self.n_reg_client + 1) == self.n_clients):
                await asyncio.sleep(5)

                task = self.loop.create_task(self.init_weight())
                await task

                random.shuffle(self.client_pool)

                message_bytes = pickle.dumps({'C_weight': self.C_weight, 'pool': self.client_pool[:self.n_clients_per_round]})
                await self.socket.publish('train_server_to_client', message_bytes)

            self.n_reg_client += 1

        except Exception as e:
            print("request_datainfo: ", e)

    async def train_client_to_server_cb(self, msg):
        try:
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            message = loaded['message']
            self.C_weights.append(loaded['C_weight'])
            self.return_id.append(loaded['ID'])
            self.n_return_client += 1

            if(self.n_return_client == self.n_clients_per_round):
                self.round += 1

                # weighted FedAvg
                np_weights = []
                for i in range(self.n_clients_per_round):
                    np_weight = np.asarray(self.C_weights[i], dtype=object)
                    np_weight *= 1 / self.n_clients_per_round
                    np_weights.append(np_weight)

                sum_np_weight = sum(np_weights)
                avg_weight = sum_np_weight.tolist()
                self.C_weight = avg_weight

                task = self.loop.create_task(self.create_testing_process())
                await task

                if(self.round < config.global_epochs):
                    self.C_weights = []
                    self.return_id = []
                    self.n_return_client = 0
                    
                    random.shuffle(self.client_pool)

                    message_bytes = pickle.dumps({'C_weight': self.C_weight, 'pool': self.client_pool[:self.n_clients_per_round]})
                    await self.socket.publish('train_server_to_client', message_bytes)
                else:
                    print('Training finished, saving model...')

                    outputfile = 'saved_weight/cnn_weight'
                    fw = open(outputfile, 'wb')
                    modelinfo = {'C_weight': self.C_weight}
                    pickle.dump(modelinfo, fw)
                    fw.close

                    await self.terminate_process()
            
        except Exception as e:
            print('train_mediator_to_client_cb: ',e)

    async def terminate_process(self):
        try:
            message_bytes = pickle.dumps({'message': 'Training has been finished, terminate all processes...'})
            target_subject = 'terminate_process'
            await self.socket.publish(target_subject, message_bytes)

            print('Training has been finished, terminate all processes...')
            self.fw.close()

            await asyncio.sleep(5)
            await self.socket.terminate()

        except Exception as e:
            print('terminate_process: ', e)

            
