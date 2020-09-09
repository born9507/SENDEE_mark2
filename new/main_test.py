from multiprocessing import Process, Value
import multiprocessing
from shared_ndarray import SharedNDArray
import numpy as np
import time
import json
import pigpio

import view
import recognition
import img2encoding
import face_tracking
import save_emotion

def expression():
    display = Process(target=display)
    display.start()
    display.join()
    left_arm = Process(target=left_arm)
    left_arm.start()
    left_arm.join()
    right_arm = Process(target=right_arm)
    right_arm.start()
    right_arm.join()
    doridori = Process(target=doridori)
    doridori.start()
    doridori.join()
    pass
def display(): pass
def left_arm(): 
    la = 13 
    ra = 19  

    right_mindc = 1700
    right_maxdc = 950
    right_interval = (right_maxdc - right_mindc)/40

    left_mindc = 950
    left_maxdc = 1900
    left_interval = (left_maxdc - left_mindc)/40

    for j in range(0, 3):
        for i in range(1, 41):
            rduty = i * right_interval + right_mindc
            lduty = i * left_interval + left_mindc
            pi.set_servo_pulsewidth(la, lduty)
            pi.set_servo_pulsewidth(ra, rduty)
            time.sleep(0.01)

        for i in range(41, 1, -1):
            rduty = i * right_interval + right_mindc
            lduty = i * left_interval + left_mindc
            pi.set_servo_pulsewidth(la, lduty)
            pi.set_servo_pulsewidth(ra, rduty)
            time.sleep(0.01)

    pi.set_servo_pulsewidth(la, 0)
    pi.set_servo_pulsewidth(ra, 0)

def right_arm(): 

    la = 13 
    ra = 19  

    right_mindc = 1700
    right_maxdc = 950
    right_interval = (right_maxdc - right_mindc)/40

    left_mindc = 950
    left_maxdc = 1900
    left_interval = (left_maxdc - left_mindc)/40

    for j in range(0, 3):
        for i in range(1, 41):
            rduty = i * right_interval + right_mindc
            lduty = i * -left_interval + left_maxdc
            pi.set_servo_pulsewidth(la, lduty)
            pi.set_servo_pulsewidth(ra, rduty)
            time.sleep(0.01)

        for i in range(1, 41):
            rduty = i * -right_interval + right_maxdc
            lduty = i * left_interval + left_mindc
            pi.set_servo_pulsewidth(la, lduty)
            pi.set_servo_pulsewidth(ra, rduty)
            time.sleep(0.01)

    pi.set_servo_pulsewidth(la, 0)
    pi.set_servo_pulsewidth(ra, 0)


def doridori(): 
    bm = 5

    body_mindc = 600
    body_maxdc = 2400
    body_interval = (body_maxdc - body_mindc)/80

    for j in range(0, 2):
        # for i in range(36, 56):
        #     bduty = round(i * body_interval + body_mindc)
        #     pi.set_servo_pulsewidth(bm, bduty)
        #     time.sleep(0.1)
        # time.sleep(0.4)

        for i in range(56, 36, -1):
            bduty = round(i * body_interval + body_mindc)
            pi.set_servo_pulsewidth(bm, bduty)
            print(pi.get_servo_pulsewidth(bm))
            time.sleep(0.1)
        time.sleep(0.4)

    pi.set_servo_pulsewidth(bm, 0)

def nodnodnod():
    hm = 6

    head_mindc = 1200
    head_maxdc = 1700
    head_interval = (head_maxdc - head_mindc)/40

    for j in range(0, 3):
        for i in range(31, 41):
            hduty = i * head_interval + head_mindc
            pi.set_servo_pulsewidth(hm, hduty)
            time.sleep(0.01)

        for i in range(41, 31, -1):
            hduty = i * head_interval + head_mindc
            pi.set_servo_pulsewidth(hm, hduty)
            time.sleep(0.01)

    pi.set_servo_pulsewidth(hm, 0)

# 언노운 일정 횟수 이상 들어오면 이 함수 실행하도록

# cv2 로 사진 캡쳐해서 찍기
def save_img():
    pass

# 해당 사진 지우기(웹 or 앱으로 처리)
def delete_img():
    pass

#############################################################

if __name__ == "__main__":
    try:
        pi = pigpio.pi()
        img2encoding.img2encoding()

        with open("face/face_list.json", "r") as f:
            face_list = json.load(f)
        known_face_names = list(face_list.keys()) # list
        known_face_names.append("unknown") # 리스트 맨 마지막에 unknown 추가, known_face_names[-1] 로 접근

        # model = model()
        # model.load_weights('model/model.h5')

        HEIGHT = Value('i', 320)
        WIDTH = Value('i', 480)

        frame = SharedNDArray((HEIGHT.value, WIDTH.value, 3))
        face_location = SharedNDArray((1,4))
        emotion = SharedNDArray((1,7))
        emotion_total = SharedNDArray((1,7))
        # print(face_location.array)
        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

        view_running = Value('i', 1)
        face_tracking_running = Value('i', 1)
        recognition_running = Value('i', 1)
        is_detected = Value('i', 0)
        name_index = Value('i', -1) # -1 이 unknown

        view = Process(target=view.view, args=(frame, HEIGHT, WIDTH, face_location ,view_running, is_detected, ))
        face_tracking = Process(target=face_tracking.face_tracking, args=(face_location, face_tracking_running, pi, ))
        recognition = Process(target=recognition.recognition, args=(frame, face_location, name_index, emotion, is_detected, ))
        save_emotion = Process(target=save_emotion.save_emotion, args=(is_detected, emotion, emotion_total, name_index, known_face_names, ))

        view.start()
        face_tracking.start()
        recognition.start()
        save_emotion.start()

        while True:
            # is_running 제어하기
            # print(is_detected.value)

            # print(np.argmax(emotion.array))
            time.sleep(0.5)
            pass
        
    finally:
        frame.unlink()
        face_location.unlink()
        emotion.unlink()
        emotion_total.unlink()
