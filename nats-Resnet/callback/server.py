import os
import math
import asyncio
import pickle
import numpy as np
import multiprocessing
from absl import flags
from nats.aio.client import Client as NATS

import utils
from mq import comm
from model import resnet_models
from model import datasetpipeline

import heapq
import config

FLAGS = flags.FLAGS

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
        self.n_mediators = config.n_mediators
        self.n_gans = config.n_gans
        self.client_id = 0
        self.mediator_id = 0
        self.n_reg_client = 0
        self.n_reg_mediator = 0
        self.n_return_mediator = 0
        self.round = 0
        self.training = False
        self.G_weights = []
        self.D_weights = []
        self.C_weights = []
        self.C_weight = None
        self.gan_dists = []
        self.dataset_ids = []
        self.dataset_dists = []
        self.dataset_sizes = []
        self.client_info = []
        self.group_info = []
        self.return_id = []
        self.size_info = [0] * self.n_mediators

        self.load_gan_weights()
        self.fw = open('accu_3.txt', 'a')

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

    def load_gan_weights(self):
        for i in range(self.n_gans):
            inputfile = os.path.join(config.gan_weight_path, 'gan_weight{}'.format(i))
            fw = open(inputfile, 'rb')
            loaded = pickle.load(fw)
            self.G_weights.append(loaded['G_weight'])
            self.D_weights.append(loaded['D_weight'])
            self.gan_dists.append(loaded['dist'])
            print(self.gan_dists[i])
            print("Load gan weight{} success".format(i))

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
            elif(data[:8] == 'mediator'):
                if(self.mediator_id < self.n_mediators):
                    send_message = pickle.dumps({'message': 'success', 'ID': self.mediator_id})
                    await self.nc.publish(msg.reply, send_message)
                    print("Send Mediator ID: ", self.mediator_id)
                    self.mediator_id += 1
                else:
                    send_message = pickle.dumps({'message': 'fail'})
                    await self.nc.publish(msg.reply, send_message)
            else:
                print("Unknown message in request_id_cb: ", data)
        except Exception as e:
            print('request_id: ', e)

    async def request_gan_weight_cb(self, msg):
        def KL(P, Q):
            """ Epsilon is used here to avoid conditional code for
            checking that neither P nor Q is equal to 0. """
            epsilon = 0.00001

            P = P+epsilon
            Q = Q+epsilon

            divergence = np.sum(P*np.log(P/Q))
            return divergence
        try:
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            message = loaded['message']
            client_id = loaded['ID']
            client_dist = loaded['dist']
            n_gans = loaded['n_gans']

            score_list = []
            for i in range(self.n_gans):
                KL_score = KL(client_dist, self.gan_dists[i])
                score_list.append(KL_score)

            max_num_index_list = list(map(score_list.index, heapq.nlargest(n_gans, score_list)))

            target_subject = msg.reply
            send_message = pickle.dumps({'message': 'success', 'gan_model_ids': max_num_index_list, 'gan_dist': self.gan_dists})
            await self.socket.publish(target_subject, send_message)

        except Exception as e:
            print('request_gan_weight: ', e)

    async def request_datainfo_cb(self, msg):
        def KL(P, Q):
            """ Epsilon is used here to avoid conditional code for
            checking that neither P nor Q is equal to 0. """
            epsilon = 0.00001

            P = P+epsilon
            Q = Q+epsilon

            divergence = np.sum(P*np.log(P/Q))
            return divergence

        try:
            send_message = pickle.dumps({'message': 'success'})
            await self.nc.publish(msg.reply, send_message)
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            self.dataset_ids.append(loaded["ID"])
            self.dataset_dists.append(loaded["dataset_dist"])
            self.dataset_sizes.append(loaded["dataset_size"])

            if((self.n_reg_client + 1) == self.n_clients):
                await asyncio.sleep(5)
                # grouping greedy algorithm
                mediator_ID = -1
                dataset_ids = self.dataset_ids
                dataset_dists = self.dataset_dists
                dataset_sizes = self.dataset_sizes
                uniform_dist = np.full((dataset_dists[0].shape[0]), 1/dataset_dists[0].shape[0], dtype = np.float32)
                n_clients_each_group = math.ceil(self.n_clients / self.n_mediators)

                while(dataset_ids):
                    mediator_ID += 1
                    n_clients_group = 0
                    dataset_size_group = 0
                    dataset_dist_group = np.zeros((self.dataset_dists[0].shape[0]), dtype = np.float32)
                    while(dataset_ids and n_clients_group < n_clients_each_group):
                        KL_score = []
                        for i in range(len(dataset_ids)):
                            tmp_dist = dataset_dist_group + dataset_dists[i] * dataset_sizes[i]
                            tmp_dist_normal = tmp_dist / (dataset_size_group + dataset_sizes[i])
                            KL_score.append(KL(tmp_dist_normal, uniform_dist))
                        min_score = min(KL_score)
                        index = KL_score.index(min_score)
                        self.client_info.append(dataset_ids[index])
                        self.group_info.append(mediator_ID)
                        dataset_size_group += dataset_sizes[index]
                        dataset_dist_group += dataset_dists[index] * dataset_sizes[index]
                        dataset_ids.pop(index)
                        dataset_dists.pop(index)
                        dataset_sizes.pop(index)
                        n_clients_group += 1
                    self.size_info[mediator_ID] = dataset_size_group
                
                print("client_info: {}".format(self.client_info))
                print("group_info:  {}".format(self.group_info))
                print("All clients have been registered, broadasting groupinfo...")

                message_bytes = pickle.dumps({'client_info': self.client_info, 'group_info': self.group_info})
                target_subject = 'publish_groupinfo'
                await self.socket.publish(target_subject, message_bytes)

            self.n_reg_client += 1

        except Exception as e:
            print("request_datainfo: ", e)

    async def request_groupinfo_cb(self, msg):
        try:
            subject = msg.subject
            message = msg.data.decode()
            if(self.n_reg_client < self.n_clients):
                message_bytes = pickle.dumps({'message': 'fail'})
                await self.socket.publish(msg.reply, message_bytes)
            else:
                message_bytes = pickle.dumps({'message': 'success', 'client_info': self.client_info, 'group_info': self.group_info})
                await self.socket.publish(msg.reply, message_bytes)

                self.n_reg_mediator += 1
                if(self.n_reg_mediator == self.n_mediators):
                    await asyncio.sleep(5)
                    print("All mediators have been registered, broadcasting init weight...")
                    task = self.loop.create_task(self.init_weight())
                    await task

                    message_bytes = pickle.dumps({'C_weight': self.C_weight})
                    await self.socket.publish('train_server_to_mediator', message_bytes)

        except Exception as e:
            print('request_groupinfo: ', e)

    async def train_mediator_to_server_cb(self, msg):
        try:
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            message = loaded['message']
            self.return_id.append(loaded['ID'])
            self.C_weights.append(loaded['C_weight'])
            self.n_return_mediator += 1

            if(self.n_return_mediator == self.n_mediators):
                self.round += 1

                # weighted FedAvg
                np_weights = []
                for i in range(self.n_mediators):
                    np_weight = np.asarray(self.C_weights[i], dtype=object)
                    np_weight *= self.size_info[self.return_id[i]]/sum(self.size_info)
                    np_weights.append(np_weight)

                sum_np_weight = sum(np_weights)
                avg_weight = sum_np_weight.tolist()
                self.C_weight = avg_weight

                task = self.loop.create_task(self.create_testing_process())
                await task

                if(self.round < config.global_epochs):
                    self.C_weights = []
                    self.n_return_mediator = 0

                    message_bytes = pickle.dumps({'C_weight': self.C_weight})
                    await self.socket.publish('train_server_to_mediator', message_bytes)
                else:
                    print('Training finished')

                    await self.terminate_process()
            
        except Exception as e:
            print('train_mediator_to_server_cb: ',e)

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

            
