import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse, sys
import asyncio
import os
import signal
from absl import app
from absl import flags
from nats.aio.client import Client as NATS

from mq import comm
from callback import server

import numpy as np

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
    server_cb = server.Server_cb(loop, nc, socket)

    await socket.connect()

    await socket.subscribe('request_id', '', server_cb.request_id_cb)

    await socket.subscribe('request_gan_weight', '', server_cb.request_gan_weight_cb)

    await socket.subscribe('request_datainfo', '', server_cb.request_datainfo_cb)

    await socket.subscribe('request_groupinfo', '', server_cb.request_groupinfo_cb)

    await socket.subscribe('train_mediator_to_server', '', server_cb.train_mediator_to_server_cb)

def main(argv):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run(loop))
    try:
        loop.run_forever()
    finally:
        loop.close()

if __name__ == '__main__':
    app.run(main)