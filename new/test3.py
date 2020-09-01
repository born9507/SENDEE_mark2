import multiprocessing as mp

import numpy as np
from shared_ndarray import SharedNDArray
import time

def func(shm,):
    while True:
        print(shm.array)
        time.sleep(0.5)

def func2(shm,):
    i = 0
    while True:
        i += 1
        shm.array[:] = np.ones((4,4))[:]
        shm.array[0, 0] = i
        print(i)
        time.sleep(1)


try:
    shm = SharedNDArray((4,4))
    p = mp.Process(target=func, args=(shm,))
    p2 = mp.Process(target=func2, args=(shm,))
    p.start()
    p2.start()
    p.join()
    p2.join()
finally:
    shm.unlink()