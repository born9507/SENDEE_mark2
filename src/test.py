from multiprocessing import Process, shared_memory
from multiprocessing.managers import SharedMemoryManager
import time

def test1(data):
    time.sleep(3)
    data = True
    print(data)
    
def test2(data):
    for i in range(10):
        time.sleep(1)
        print(f"{i+1}second, {data}")

if __name__ == '__main__':
    with SharedMemoryManager() as smm:
        data = smm.SharedMemory(size=128)
        data = False
        print(data)
        Process(target=test1, args=(data, )).start()
        Process(target=test2, args=(data, )).start()
