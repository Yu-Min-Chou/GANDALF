from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import asyncio
import utils
import signal
import pickle
import argparse, sys
import numpy as np
import multiprocessing
from absl import app
from nats.aio.client import Client as NATS
import config
from model import models

def get_init_weight(ID):
    from tensorflow import keras
    from tensorflow.python.ops import control_flow_util

    os.environ["CUDA_VISIBLE_DEVICES"]=str(ID)
    keras.backend.clear_session()
    control_flow_util.ENABLE_CONTROL_FLOW_V2 = True

    wgangp = models.WGANGP(0)
    G_weight, D_weight = wgangp.get_models_weight()

    return G_weight,D_weight

class Mediator_cb():
    def __init__(self, loop, nc, socket):
        self.loop = loop
        self.nc = nc
        self.socket = socket
        self.ID = -1
        self.round = 0
        self.dataset_dists = []
        self.dataset_sizes = []
        self.times_participated = []
        self.group_members = []
        self.g_loss = []
        self.d_loss = []

        self.fw = None

    def get_id(self):
        return self.ID
    
    async def request_id_cb(self, msg):
        # receive a message from [request_id-reply]
        # message is a int, ID of this mediator
        try:
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            if(loaded['message'] == 'success'):
                print("get ID: ", loaded['ID'])
                self.ID = loaded['ID']
            else:
                print("Amount of mediators has been reached the upper limit")
                await self.socket.terminate()
        except Exception as e:
            print(e)

    async def request_groupinfo_cb(self, msg):
        try:
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            message = loaded['message']

            if(message == 'success'):
                dataset_ids = loaded['dataset_ids']
                group_info = loaded['group_info']
                dists = loaded['dataset_dists']
                sizes = loaded['dataset_sizes']
                for i in range(len(group_info)):
                    if(group_info[i] == self.ID):
                        self.group_members.append(dataset_ids[i])
                        self.dataset_dists.append(dists[i])
                        self.dataset_sizes.append(sizes[i])
                        self.times_participated.append(0)
                print('Mediator{}, group members ID : {}'.format(self.ID, self.group_members))

                # test
                for i in range(len(self.group_members)):
                    print(self.dataset_dists[i])
                    print(self.dataset_sizes[i])

                await asyncio.sleep(10)

                await self.choose_first_client()
            else:
                await asyncio.sleep(10)
                await self.socket.request('request_groupinfo', 'mediator_groupinfo_request'.encode(), self.request_groupinfo_cb)
            
        except Exception as e:
            print("request_groupinfo_cb: ", e)

    async def choose_first_client(self):
        first_client_id = -1
        perform_eval = False

        with multiprocessing.Pool(1) as pool:
            weights = pool.map(get_init_weight, [self.ID])
            G_weight, D_weight = weights[0]

        print("Mediator{}: Get init weights, choosing first client....".format(self.ID))

        for i in range(len(self.group_members)):
            if(self.times_participated[i] <= self.round):
                message_send = 'check_network'
                message_bytes = pickle.dumps({'message': message_send})
                target_subject = 'request_check_network_client{}'.format(self.group_members[i])
                try:
                    await self.socket.request(target_subject, message_bytes, self.request_check_network_cb)
                except Exception as e:
                    print(e)
                    continue

                self.times_participated[i] += 1
                message_send = 'from mediator{}'.format(self.ID)
                message_bytes = pickle.dumps({'message': message_send, 'G_weight': G_weight, 'D_weight': D_weight, 'round': self.round})
                first_client_id = self.group_members[i]
                target_subject = 'train_client{}'.format(first_client_id)
                try:
                    await self.socket.publish(target_subject, message_bytes)
                except Exception as e:
                    print(e)

                break

    async def choose_next_client(self, msg, perform_eval):
        next_client_id = -1
        for i in range(len(self.group_members)):
            if(self.times_participated[i] <= self.round):
                message_send = 'check_network'
                message_bytes = pickle.dumps({'message': message_send})
                target_subject = 'request_check_network_client{}'.format(self.group_members[i])
                try:
                    await self.socket.request(target_subject, message_bytes, self.request_check_network_cb)
                except Exception as e:
                    print(e)
                    continue

                self.times_participated[i] += 1
                next_client_id = self.group_members[i]
                message_send = 'client'
                message_bytes = pickle.dumps({'message': message_send, 'next_client_id': next_client_id, 'round': self.round})
                try:
                    await self.socket.publish(msg.reply, message_bytes)
                except Exception as e:
                    print(e)

                break
        return next_client_id

    async def request_pass_weight_cb(self, msg):
        try:
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            message = loaded['message']
            g_loss = loaded['g_loss']
            d_loss = loaded['d_loss']
            print(message)
            print("g_loss: {}, d_loss: {}\n".format(g_loss, d_loss))

            self.g_loss.append(g_loss)
            self.d_loss.append(d_loss)
            
            next_client_id = await self.choose_next_client(msg, False)

            if(next_client_id == -1):
                perform_eval = True
                if(self.round < config.min_rounds):
                    self.round += 1
                    print("Mediator{}: {}th round finish".format(self.ID, self.round))
                    next_client_id = await self.choose_next_client(msg, perform_eval)
                elif(self.round < config.max_rounds):
                    n_converged = config.n_converged
                    converged_threshold = config.converged_threshold
                    g_loss_last = self.g_loss[-1]
                    d_loss_last = self.d_loss[-1]
                    g_loss_countdown = self.g_loss[-(n_converged)]
                    d_loss_countdown = self.d_loss[-(n_converged)]
                    g_slope = abs(g_loss_last - g_loss_countdown) / n_converged
                    d_slope = abs(d_loss_last - d_loss_countdown) / n_converged

                    if(g_slope <= converged_threshold and d_slope <= converged_threshold):
                        print("Model weight converged, let clients send model to mediator")
                        message_send = 'mediator'
                        message_bytes = pickle.dumps({'message': message_send, 'next_client_id': next_client_id, 'round': self.round})
                        await self.socket.publish(msg.reply, message_bytes)
                    else:
                        self.round += 1
                        print("Mediator{}: {}th round finish".format(self.ID, self.round))
                        next_client_id = await self.choose_next_client(msg, perform_eval)
                    
                    
                else:
                    print("finish training, weight does not converged")
                    message_send = 'mediator'
                    message_bytes = pickle.dumps({'message': message_send, 'next_client_id': next_client_id, 'round': self.round})
                    await self.socket.publish(msg.reply, message_bytes)
                
        except Exception as e:
            print("pass_weight:", e)

    async def request_check_network_cb(self, msg):
        try:
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            message = loaded['message']

        except Exception as e:
            print(e)

    async def save_weight_cb(self, msg):
        try:
            print("Mediator{}: Get model weight, saving......".format(self.ID))
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            message = loaded['message']
            G_weight = loaded['G_weight']
            D_weight = loaded['D_weight']

            total_dist = np.zeros([self.dataset_dists[0].shape[0]], dtype=np.float32)

            for i in range(len(self.group_members)):
                total_dist += self.dataset_dists[i] * self.dataset_sizes[i]

            total_dist /= np.sum(total_dist)
            print("The data distribution of this mediaotr: ")
            print(total_dist)

            outputfile = config.saved_gan_weight_path + 'gan_weight{}'.format(self.ID)
            fw = open(outputfile, 'wb')
            modelinfo = {'G_weight': G_weight, 'D_weight': D_weight, 'dist': total_dist}
            pickle.dump(modelinfo, fw)
            fw.close()

            await self.terminate_process()

        except Exception as e:
            print('save_weight_cb: ', e)

    async def terminate_process(self):
        try:
            print('Mediator{}: Saving success, terminating mediator.....'.format(self.ID))
            message_send = 'Mediator{} finish training, it would be terminated...'.format(self.ID)
            message_bytes = pickle.dumps({'message': message_send})
            target_subject = 'work_done'
            await self.socket.publish(target_subject, message_bytes)

            message_send = 'Terminating group{}'.format(self.ID)
            message_bytes = pickle.dumps({'message': message_send})
            target_subject = 'terminating_group{}'.format(self.ID)
            await self.socket.publish(target_subject, message_bytes)

            await asyncio.sleep(5)

            await self.socket.terminate()

        except Exception as e:
            print('terminate_process: ', e)







            