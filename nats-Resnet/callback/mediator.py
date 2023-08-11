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
import random
import argparse, sys
import numpy as np
import multiprocessing
from absl import app
from absl import flags
from tensorflow import keras
from tensorflow.python.ops import control_flow_util
from nats.aio.client import Client as NATS

from model import resnet_models
from model import gan_models
import config

keras.backend.clear_session()
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True

FLAGS = flags.FLAGS

class Mediator_cb():
    def __init__(self, loop, nc, socket):
        self.loop = loop
        self.nc = nc
        self.socket = socket
        self.ID = -1
        self.round = 0
        self.C_weight = None
        self.times_participated = []
        self.group_members = []
        self.g_loss = []
        self.d_loss = []

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
                client_info = loaded['client_info']
                group_info = loaded['group_info']
                for i in range(len(group_info)):
                    if(group_info[i] == self.ID):
                        self.group_members.append(client_info[i])
                        self.times_participated.append(0)
                print('Mediator{}, group members ID : {}'.format(self.ID, self.group_members))

            else:
                await asyncio.sleep(5)
                await self.socket.request('request_groupinfo', 'mediator_groupinfo_request'.encode(), self.request_groupinfo_cb)
            
        except Exception as e:
            print("request_groupinfo_cb: ", e)

    async def choose_first_client(self):
        first_client_id = -1
        for i in range(len(self.group_members)):
            if(self.times_participated[i] <= self.round):
                message_bytes = pickle.dumps({'message': 'check_network'})
                target_subject = 'request_check_network_client{}'.format(self.group_members[i])
                try:
                    await self.socket.request(target_subject, message_bytes, self.request_check_network_cb)
                except Exception as e:
                    print(e)
                    continue
                
                self.times_participated[i] += 1
                message_bytes = pickle.dumps({'message': 'from mediaotr{}'.format(self.ID), 'C_weight': self.C_weight})
                first_client_id = self.group_members[i]
                target_subject = 'train_client{}'.format(first_client_id)
                try:
                    await self.socket.publish(target_subject, message_bytes)
                except Exception as e:
                    print(e)
                
                break

    async def choose_next_client(self, msg):
        next_client_id = -1
        for i in range(len(self.group_members)):
            if(self.times_participated[i] <= self.round):
                message_bytes = pickle.dumps({'message': 'check_network'})
                target_subject = 'request_check_network_client{}'.format(self.group_members[i])
                try:
                    await self.socket.request(target_subject, message_bytes, self.request_check_network_cb)
                except Exception as e:
                    print(e)
                    continue

                self.times_participated[i] += 1
                next_client_id = self.group_members[i]
                message_bytes = pickle.dumps({'message': 'client', 'next_client_id': next_client_id})
                try:
                    await self.socket.publish(msg.reply, message_bytes)
                except Exception as e:
                    print(e)
                
                break
        return next_client_id

    async def train_server_to_mediator_cb(self, msg):
        try:
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            self.C_weight = loaded['C_weight']

            print("Starting a round of training")

            self.round = 0
            for i in range(len(self.times_participated)):
                self.times_participated[i] = 0

            await self.choose_first_client()

        except Exception as e:
            print('train_server_to_mediator_cb: ',e)

    async def request_pass_weight_cb(self, msg):
        try:
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            message = loaded['message']
            loss = loaded['loss']
            accu = loaded['accu']

            print(message + 'loss: {:.4f}, accu: {:.4f}'.format(loss, accu))

            next_client_id = await self.choose_next_client(msg)

            if(next_client_id == -1):
                print("Mediator{}: {}th round finish".format(self.ID, self.round))
                print('#' * 45)
                self.round += 1
                # random.shuffle(self.group_members)
                if(self.round < config.group_epochs):
                    next_client_id = await self.choose_next_client(msg)
                else:
                    message_bytes = pickle.dumps({'message': 'mediator', 'next_client_id': next_client_id})
                    await self.socket.publish(msg.reply, message_bytes)

        except Exception as e:
            print('request_pass_weight_cb: ',e)


    async def train_client_to_mediator_cb(self, msg):
        try:
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            message = loaded['message']
            self.C_weight = loaded['C_weight']

            message_bytes = pickle.dumps({'message': 'success', 'C_weight': self.C_weight, 'ID': self.ID})
            target_subject = 'train_mediator_to_server'
            await self.socket.publish(target_subject, message_bytes)

        except Exception as e:
            print('train_client_to_mediator_cb: ',e)

    async def request_check_network_cb(self, msg):
        try:
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            message = loaded['message']

        except Exception as e:
            print('request_check_network:', e)

    async def terminate_process_cb(self, msg):
        try:
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            message = loaded['message']
            print(message)

            await self.socket.terminate()

        except Exception as e:
            print('terminate_process_cb: ', e)

    
            