import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse, sys
import asyncio
import signal
import multiprocessing
from absl import app
from absl import flags
from nats.aio.client import Client as NATS

import utils
import config
from mq import comm
from callback import client

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

    ############################# Get ID #############################
    await socket.request('request_id', "client_register_request".encode(), client_cb.request_id_cb)

    datainfo_bytes, num_data = utils.get_datainfo(client_cb.get_id())
    client_cb.set_num_data(num_data)
    
    ############# Send data distribution & dataset size ##############
    await socket.request('request_datainfo', datainfo_bytes, client_cb.request_datainfo_cb)


    #################### Get grouping information #####################
    await socket.subscribe('publish_groupinfo', '', client_cb.publish_groupinfo_cb)

    ################### Check network connection ####################
    await socket.subscribe('request_check_network_client{}'.format(client_cb.get_id()), '', client_cb.request_check_network_cb)

    ########### Ready to receive model weight and training ###########
    await socket.subscribe('train_client{}'.format(client_cb.get_id()), '', client_cb.train_cb)

    ############### Finish training, terminate clients ###############

    await asyncio.sleep(15)
    await socket.subscribe('terminating_group{}'.format(client_cb.get_group_id()), '', client_cb.terminate_cb)


def main(argv):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run(loop))
    try:
        loop.run_forever()
    finally:
        loop.close()

if __name__ == '__main__':
    app.run(main)