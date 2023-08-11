import argparse, sys
import asyncio
import os
import signal
from absl import app
from absl import flags
from nats.aio.client import Client as NATS

from mq import comm
from callback import mediator

def show_usage_and_die():
    show_usage()
    sys.exit(1)

async def run(loop):
    
    nc = NATS()
    socket = comm.Comm(loop, nc)
    mediator_cb = mediator.Mediator_cb(loop, nc, socket)
    
    await socket.connect()

    await socket.request('request_id', 'mediator_register_request'.encode(), mediator_cb.request_id_cb)

    await socket.request('request_groupinfo', 'mediator_groupinfo_request'.encode(), mediator_cb.request_groupinfo_cb)

    await socket.subscribe('train_server_to_mediator', '', mediator_cb.train_server_to_mediator_cb)

    await socket.subscribe('request_pass_weight_{}'.format(mediator_cb.get_id()), '', mediator_cb.request_pass_weight_cb)

    await socket.subscribe('train_client_to_mediator{}'.format(mediator_cb.get_id()), '', mediator_cb.train_client_to_mediator_cb)

    await socket.subscribe('terminate_process', '', mediator_cb.terminate_process_cb)


def main(argv):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run(loop))
    try:
        loop.run_forever()
    finally:
        loop.close()

if __name__ == '__main__':
    app.run(main)
