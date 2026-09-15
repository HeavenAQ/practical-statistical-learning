import socket
import time

import ray

ray.init(address="auto")


@ray.remote(resources={"mps": 1})
def identify():
    hostname = socket.gethostname()
    time.sleep(10)
    return hostname


refs = [
    identify.remote(),
    identify.remote(),
]

print(ray.get(refs))
