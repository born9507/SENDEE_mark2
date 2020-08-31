from multiprocessing import Process
import cv2
import numpy as np
import time 


def view():
    camera = picamera.PiCamera()
    output = np.empty((240, 320, 3), dtype=np.uint8)
    camera.resolution = (320, 240)
    while True:
        start = time.time()
        camera.capture(output, format="rgb")
        print(f"cycle time: {time.time() - start}")
        # return output

if __name__ == "__main__":
    while True:
        p = Process(target=view)
        p.start()
        p.join()