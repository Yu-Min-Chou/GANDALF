import asyncio
import pickle
from nats.aio.client import Client as NATS

import utils
from mq import comm
from model import models
import config

class Server_cb():
    def __init__(self, loop, nc, socket):
        self.loop = loop
        self.nc = nc
        self.socket = socket
        self.n_clients = config.n_clients
        self.n_mediators = config.n_mediators
        self.client_id = 0
        self.mediator_id = 0
        self.n_reg_client = 0
        self.n_terminated_mediator = 0
        self.dataset_ids = []
        self.dataset_dists = []
        self.dataset_sizes = []
    
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
            print(e)

    async def request_datainfo_cb(self, msg):
        # receive message from [request_datainfo]
        # message contain dataset distribution and size of dataset
        try:
            send_message = pickle.dumps({'message': 'success'})
            await self.nc.publish(msg.reply, send_message)
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            self.dataset_ids.append(loaded["ID"])
            self.dataset_dists.append(loaded["dataset_dist"])
            self.dataset_sizes.append(loaded["dataset_size"])
            self.n_reg_client += 1

            if(self.n_reg_client == self.n_clients):
                await asyncio.sleep(5)
                print("All client has been registered, broadcasting groupinfo....")
                group_info = utils.k_cluster(self.dataset_dists)
                send_message = pickle.dumps({'dataset_ids': self.dataset_ids, 'group_info': group_info})
                await self.socket.publish('publish_groupinfo', send_message)

        except Exception as e:
            semd_message = pickle.dumps({'message': 'fail'})
            await self.nc.publish(msg.reply, send_message)
            print(e)

    async def request_groupinfo_cb(self, msg):
        try:
            subject = msg.subject
            message = msg.data.decode()
            if(self.n_reg_client < self.n_clients):
                send_message = pickle.dumps({'message': 'fail'})
                await self.socket.publish(msg.reply, send_message)
            else:
                group_info = utils.k_cluster(self.dataset_dists)
                send_message = pickle.dumps({'message': 'success', 'dataset_ids': self.dataset_ids, 'group_info': group_info, 'dataset_dists': self.dataset_dists, 'dataset_sizes': self.dataset_sizes})
                await self.socket.publish(msg.reply, send_message)
        except Exception as e:
            print(e)

    async def work_done_cb(self, msg):
        try:
            subject = msg.subject
            loaded = pickle.loads(msg.data)
            message = loaded['message']
            
            print(message)

            self.n_terminated_mediator += 1 

            if(self.n_terminated_mediator == self.n_mediators):
                print('All mediator finish training, terminate server....')
                await self.socket.terminate()

        except Exception as e:
            print(e)
            
            
