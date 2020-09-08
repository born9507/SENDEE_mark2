
from multiprocessing import Process, Value
import multiprocessing
from shared_ndarray import SharedNDArray
import numpy as np
import time
import cv2
# import face_recognition

# view 함수는 카메라로 촬영한 것을 frame 변수에 저장(np.array, dtype=float64)
def view(frame, is_running, ):
    capture = cv2.VideoCapture(-1)
    while True:
        if is_running.value==1:
            ret, frame_ = capture.read()
            if not ret: break
            frame.array[:] = frame_[:]
            # 중요! 값을 하나하나 넣어준다. 통채로 대입하면 메모리 주소가 달라지는 듯
        else:
            time.sleep(1)
            pass

# face_tracking 함수는 얼굴 위치를 받아서 모터 제어까지(기존의 )
def face_tracking(frame, is_running, ):
    face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt2.xml')
    while True:
        uint8 = frame.array.astype(np.uint8)
        gray = cv2.cvtColor(uint8, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # cv2.putText(frame, info, (5, 15), font, 0.5, (255, 0, 255), 1)

        print(faces)




def recognition(frame):
    while True:
        pass
        

if __name__ == "__main__":
    try:
        frame = SharedNDArray((480, 640, 3))

        view_running = Value('i', 1) 
        face_tracking_running = Value('i', 1)
        
        view = Process(target=view, args=(frame, view_running, ))
        face_tracking = Process(target=face_tracking, args=(frame, face_tracking_running, ))
        
        view.start()
        face_tracking.start()

        view.join()
        face_tracking.join()
        
    finally:
        frame.unlink()
