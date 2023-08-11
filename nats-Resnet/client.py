import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse, sys
import asyncio
import signal
import pickle
import multiprocessing
from absl import app
from absl import flags
from nats.aio.client import Client as NATS

import utils
from mq import comm
from callback import client

import numpy as np
import config

def show_usage():
    usage = """
nats-sub SUBJECT [-s SERVER] [-q QUEUE]
Example:
nats-sub help -q workers -s nats://127.0.0.1:4222 -s nats://127.0.0.1:4223
"""
    print(usage)

def show_usage_and_die():
    show_usage()
    sys.exit(1)

async def run(loop):

    nc = NATS()
    socket = comm.Comm(loop, nc)
    client_cb = client.Client_cb(loop, nc, socket)
    
    await socket.connect()

    await socket.request('request_id', "client_register_request".encode(), client_cb.request_id_cb)

    await socket.subscribe('gpu_used', '', client_cb.gpu_used_cb)

    message = 'client_gan_weight_request'
    message_bytes = pickle.dumps({'message': message, 'ID': client_cb.get_id(), 'dist': client_cb.get_dist(), 'n_gans': config.n_gans})
    await socket.request('request_gan_weight', message_bytes, client_cb.request_gan_weight_cb)

    await socket.subscribe('publish_groupinfo', '', client_cb.publish_groupinfo_cb)

    await socket.subscribe('request_check_network_client{}'.format(client_cb.get_id()), '', client_cb.request_check_network_cb)

    await socket.subscribe('train_client{}'.format(client_cb.get_id()), '', client_cb.train_cb)

    await socket.subscribe('terminate_process', '', client_cb.terminate_process_cb)

def main(argv):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run(loop))
    try:
        loop.run_forever()
    finally:
        loop.close()

if __name__ == '__main__':
    app.run(main)