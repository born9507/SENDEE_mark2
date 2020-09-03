import multiprocessing as mp
from multiprocessing import Process

import numpy as np
# from shared_ndarray import SharedNDArray
import time

def func1():
    while True:
        p = Process(target=func3)
        p.start()
        p.join()
        time.sleep(1)

    # while True:
    #     print(shm.array)
    #     time.sleep(0.5)

def func2(shm,):
    i = 0
    while True:
        i += 1
        shm.array[:] = np.ones((4,4))[:]
        shm.array[0, 0] = i
        print(i)
        time.sleep(1)

def func3():
    print("son of func1")


try:
    # shm = SharedNDArray((4,4))
    p1 = mp.Process(target=func1)
    p1.start()
    p1.join()
    # p2 = mp.Process(target=func2, args=(shm,))
    # p2.start()
    # p2.join()
finally:
    # shm.unlink()
    pass