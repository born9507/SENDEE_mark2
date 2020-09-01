from multiprocessing import Process
from shared_ndarray import SharedNDArray
import cv2
import numpy as np
import time 
import picamera


def view():
    HEIGHT = 360
    WIDTH =  480

    capture = cv2.VideoCapture(-1)
    capture.set(3, WIDTH)
    capture.set(4, HEIGHT)

    # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    info = ''
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        start = time.time()
        
        ret, frame = capture.read()
        if not ret: break

        print(frame.shape)  

        print(f"cycle time: {time.time()-start}" )

if __name__ == "__main__":
    try:
        frame = SharedNDArray((360 , 480, 3))
        while True:
            Process(target=view, args=(frame,)).start
            print(frame)
    finally:
        frame.unlink()
