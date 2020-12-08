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


if __name__ == "__main__":
    try:
        pi = pigpio.pi()
        img2encoding.img2encoding()

        with open("face/face_list.json", "r") as f:
            face_list = json.load(f)
        known_face_names = list(face_list.keys()) # list
        known_face_names.append("unknown")

        HEIGHT = Value('i', 320)
        WIDTH = Value('i', 480)

        frame = SharedNDArray((HEIGHT.value, WIDTH.value, 3))
        face_location = SharedNDArray((1,4))
        emotion = SharedNDArray((1,7))
        emotion_total = SharedNDArray((1,7))
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
            name = known_face_names[name_index.value]
            emotion_ = emotion_dict[np.argmax(emotion.array)]
            expression.emo2reaction(name, emotion_)

            time.sleep(0.5)
            pass
        
    finally:
        frame.unlink()
        face_location.unlink()
        emotion.unlink()
        emotion_total.unlink()
