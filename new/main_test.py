from multiprocessing import Process, Value
import multiprocessing
from shared_ndarray import SharedNDArray
import numpy as np
import time
import json
import pigpio
import random
import cv2
import view
import recognition
import img2encoding
import face_tracking
import save_emotion
import expression

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
        face_tracking = Process(target=face_tracking.face_tracking, args=(face_location, face_tracking_running, pi, is_detected, ))
        recognition = Process(target=recognition.recognition, args=(frame, face_location, name_index, emotion, is_detected, ))
        save_emotion = Process(target=save_emotion.save_emotion, args=(is_detected, emotion, emotion_total, name_index, known_face_names, ))

        view.start()
        face_tracking.start()
        recognition.start()
        save_emotion.start()
        
        while True:
            # is_running 제어하기
            name = known_face_names[name_index.value]
            emotion_ = emotion_dict[np.argmax(emotion.array)]
            expression.emo2reaction(name, emotion_)
            # if is_detected.value == 1:
            expression.randMove(pi, random.randint(0, 8))
            time.sleep(0.5)
            pass
        
    finally:
        frame.unlink()
        face_location.unlink()
        emotion.unlink()
        emotion_total.unlink()
