import sys
import signal
import asyncio
from nats.aio.client import Client as NATS
import config

class Comm:
    def __init__(self, loop, nc):
        self.loop = loop
        self.nc = nc

        # for sig in ('SIGINT', 'SIGTERM'):
        #     self.loop.add_signal_handler(getattr(signal, sig), self.signal_handler)
        self.loop.add_signal_handler(getattr(signal, 'SIGINT'), self.signal_handler)

    
    def signal_handler(self):
        if self.nc.is_closed:
            return
        print("Disconnecting...")
        self.loop.create_task(self.nc.close())
    
    async def error_cb(self, e):
        print("Error:", e)

    async def closed_cb(self):
        print("Connection to NATS is closed")
        self.loop.create_task(self.nc.close())
        await asyncio.sleep(0.1, loop=self.loop)
        self.loop.stop()
        # if(self.terminate):
        #     await asyncio.sleep(0.1, loop=self.loop)
        #     self.loop.stop()
        # else:
        #     print('reconnect again')
        #     self.nc = NATS()
        #     await self.connect()
    
    async def reconnected_cb(self):
        print(f"Connected to NATS at {self.nc.connected_url.netloc}...")

    async def connect(self):
        options = {
            # "loop": self.loop,
            "error_cb": self.error_cb,
            "closed_cb": self.closed_cb,
            "reconnected_cb": self.reconnected_cb
        }

        try:
            if len(config.servers) > 0:
                options['servers'] = config.servers

                await self.nc.connect(**options)
        except Exception as e:
            print(e)
            sys.exit(1)

        print(f"Connected to NATS at {self.nc.connected_url.netloc}...")

    async def subscribe(self, subject, queue, cb):
        await self.nc.subscribe(subject, queue, cb)

    async def publish(self, subject, payload):
        await self.nc.publish(subject, payload)
        await self.nc.flush(3)

    async def request(self, subject, data, cb):
        msg = await self.nc.request(subject, data, timeout=3)
        await cb(msg)

    async def terminate(self):
        self.loop.create_task(self.nc.close())