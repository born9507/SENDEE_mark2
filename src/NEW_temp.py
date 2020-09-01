# -*- coding: utf-8 -*-
from multiprocessing import Process
import motordrive
import _pickle as pickle
import time
import cv2
import RPi.GPIO as GPIO
import numpy as np
import face_recognition
import os, gc
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation, BatchNormalization

import model as md
import display
import recognition

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def webcam():

    HEIGHT = 360
    WIDTH =  480

    capture = cv2.VideoCapture(-1)
    capture.set(3, WIDTH)
    capture.set(4, HEIGHT)
    # capture.set(10, 60) #brightness
    # capture.set(11, 60) #contrast
    # capture.set(21, 0.25) #auto exposure
    #capture.set(5, 60)

    hor_error_Sum = 0
    hor_error_Prev = 0
    ver_error_Sum = 0
    ver_error_Prev = 0
    past_dc = 4
    ##########

    face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt2.xml')
    # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    info = ''
    font = cv2.FONT_HERSHEY_SIMPLEX

    count = 0
    speed = 10
    isDetected = False

    cycle_time = 0.05 # 1 frame time

    while True:
        start = time.time() 
        
        ## stop when 
        # if onprocess == True:
        #     # GPIO.cleanup()
        #     time.sleep(3.5)
        # else: 
        ret, frame = capture.read()
        if not ret: break

        rgb_for_face = frame[::]
        gray_for_emotion = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray_for_emotion, 1.3, 5)

        cv2.putText(frame, info, (5, 15), font, 0.5, (255, 0, 255), 1)

        ############# once in 10 frames ###############
        if count==speed:
            with open("pkl/rgb_for_face.pkl", "wb") as file:
                pickle.dump(rgb_for_face, file) 
            with open("pkl/gray_for_emotion.pkl", "wb") as file:
                pickle.dump(gray_for_emotion, file)
            count = 0
        else:
            count += 1
        ###################################################
        
        if len(faces)>1:
            face_list = []

            for face in faces:
                face_list.append(face[2])
            faces = np.array([faces[np.argmax(np.array(face_list))]])

        if len(faces)==1:
            if isDetected == False:
                isDetected = True   #
                with open("pkl/isDetected.pkl", "wb") as file:
                    pickle.dump(isDetected, file)
            # for (x, y, w, h) in faces:
            [x, y, w, h] = faces[0]
            
            face_locations = np.array([[y, x+w, y+h, x]])
            (top, right, bottom, left) = (y, x+w, y+h, x)
            cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)

            with open("pkl/face_locations.pkl", "wb") as file:
                pickle.dump(face_locations, file)
            
            x_pos = x + w/2
            y_pos = y + h/2

            x_pos = 2 * (x_pos - WIDTH/2) / WIDTH + 0.1
            y_pos = -2 * (y_pos - (HEIGHT/2)) / HEIGHT

            ###########
            hor_error_Sum = hor_error_Sum + x_pos
            ver_error_Sum = ver_error_Sum + y_pos
            motordrive.MPIDCtrl(x_pos, 0.05, hor_error_Sum, hor_error_Prev)
            past_dc = motordrive.Servo(y_pos, 0.05, past_dc, ver_error_Sum, ver_error_Prev)
            hor_error_Prev = x_pos
            ver_error_Prev = y_pos
            ###########

        else:     # No face detected
            if isDetected==True:
                motordrive.headsleep()
                isDetected = False #no man True
                with open("pkl/isDetected.pkl", "wb") as file:
                    pickle.dump(isDetected, file)
            else:
                motordrive.headsleep()

        frame = cv2.flip(frame, 1)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'): break

        # print("time :", time.time() - start)
        if (time.time() - start) < cycle_time:
            time.sleep(cycle_time - (time.time() - start))
        print(isDetected)

    GPIO.cleanup()

def armMove():
    cycle_time = 0
    while True:
        try:
            start = time.time()
            gc.disable()
            with open("pkl/emotion.pkl", "rb") as file:
                emotion = pickle.load(file)
            gc.enable()
            
            if emotion == 'neutral1':
                cycle_time= 11*90/1000
            else:
                cycle_time = 33*90/1000
            
            # motordrive.emoreact(emotion)
            
            if (time.time() - start) < cycle_time:
                time.sleep(cycle_time - (time.time() - start))

        except EOFError:
            pass

def main():
    recognition.img2encoding()
    model = md.model_basic()
    model.load_weights('models/model.h5')

    cycle_time = 1


    while True:
        try:
            start = time.time() 
            # gc.disable()
            with open("pkl/isDetected.pkl", "rb") as file:
                isDetected = pickle.load(file)
                file.close()
            # gc.enable()
            if isDetected == True: #recognized

                emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
                # 
                # count=0
                # while count==3:
                #     prediction = face_emo(model)
                #     prediction_sum = prediction_sum + prediction
                #     count+=1
                # prediction = prediction_sum
                # prediction_sum = [0,0,0,0,0,0,0]

                prediction = recognition.face_emo(model)
                emotion = emotion_dict[np.argmax(prediction)]
                print(emotion)

                name = recognition.face_reco()
                print(name)
                ###################### action #####################################
                # onprocess = True
                # with open("pkl/onprocess.pkl", "wb") as file:
                #     pickle.dump(onprocess, file)

                display.emo2reaction(emotion, name)  ##unknown sangwon 

                # onprocess = False
                # with open("pkl/onprocess.pkl", "wb") as file:
                #     pickle.dump(onprocess, file)
                ################################################################
            else: # not recognized
                display.noface()
 
            print("time :", (time.time() - start), "\n") 
            
            #cycle_time, 1 action
            if (time.time() - start) < cycle_time:
                time.sleep(cycle_time - (time.time() - start))

        except EOFError:
            pass
        except pickle.UnpicklingError as e:
            pass
        pass

if __name__ == "__main__":
    Process(target=webcam).start()
    Process(target=armMove).start()
    Process(target=main).start()
