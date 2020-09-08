import pigpio
import time
import cv2
import numpy as np

capture = cv2.VideoCapture(-1)
capture.set(3, 480)
capture.set(4, 360)
capture.set(5, 60)

bm = 5
hm = 6

head_mindc = 1200
head_maxdc = 1700
head_interval = (head_maxdc - head_mindc)/40

body_mindc = 600
body_maxdc = 2400
body_interval = (body_maxdc - body_mindc)/40

right_mindc = 500
right_maxdc = 1300
right_interval = (right_maxdc - right_mindc)/40

left_mindc = 1250
left_maxdc = 2000
left_interval = (left_maxdc - left_mindc)/40

hor_error_Sum = 0
hor_error_Prev = 0
ver_error_Sum = 0
ver_error_Prev = 0
past_hor_dc = 1500
past_ver_dc = 2000

pi = pigpio.pi()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

def headServo(error_Now, waittime, past_dc, error_Sum, error_Prev, head_mindc, head_maxdc, head_interval, pi, hm):
    Kp = 1
    Ki = 0
    Kd = 0
    
    error = error_Now
    error_sum = error_Sum + error
    error_diff = (error-error_Prev)/waittime
    
    ctrlval = -(Kp*error + Ki*error_sum*waittime + Kd*error_diff)
           
    head_duty = past_dc - head_interval * ctrlval
    
    if head_duty < head_mindc:
        head_duty = head_mindc
        
    elif head_duty > head_maxdc:
        head_duty = head_maxdc
    
    print('ctrlval',ctrlval)
    
    if head_duty == past_dc:
        print(head_duty, past_dc,'steady')
        head_duty = past_dc
        pi.set_servo_pulsewidth(hm, 0)
    else:
        print(head_duty, past_dc,'move')
        pi.set_servo_pulsewidth(hm, head_duty)
        time.sleep(waittime)
        pi.set_servo_pulsewidth(hm, 0)

    return head_duty

def bodyServo(error_Now, waittime, past_dc, error_Sum, error_Prev, body_mindc, body_maxdc, body_interval, pi, bm):    
    Kp = 0.5
    Ki = 0
    Kd = 0
    
    error = error_Now
    error_sum = error_Sum + error
    error_diff = (error-error_Prev)/waittime
    
    ctrlval = -(Kp*error + Ki*error_sum*waittime + Kd*error_diff)
           
    body_duty = past_dc - body_interval * ctrlval
    
    if body_duty < body_mindc:
        body_duty = body_mindc
        
    elif body_duty > body_maxdc:
        body_duty = body_maxdc
    
    print('ctrlval',ctrlval)
    
    if body_duty == past_dc:
        print(body_duty, past_dc,'steady')
        body_duty = past_dc
        pi.set_servo_pulsewidth(bm, 0)
    else:
        print(body_duty, past_dc,'move')
        pi.set_servo_pulsewidth(bm, body_duty)
        time.sleep(waittime)
        pi.set_servo_pulsewidth(bm, 0)

    return body_duty

while True:
    ret, frame = capture.read()
    frame = cv2.flip(frame, 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print(faces)
    if len(faces) > 1:
        face_list = []
        for face in faces:
            face_list.append(face[2])
        faces = np.array([faces[np.argmax(np.array(face_list))]])
        
    if len(faces)==1:
        [x,y,w,h] = faces[0]
        x_pos = (x+w/2-240)/240
        y_pos = (y+h/2-180)/180
        
        hor_error_Sum = hor_error_Sum + x_pos
        ver_error_Sum = ver_error_Sum + y_pos
        past_ver_dc = headServo(y_pos, 0.01, past_ver_dc, ver_error_Sum, ver_error_Prev, head_mindc, head_maxdc, head_interval, pi, hm, )
        past_hor_dc = bodyServo(x_pos, 0.01, past_hor_dc, hor_error_Sum, hor_error_Prev, body_mindc, body_maxdc, body_interval, pi, bm)
        hor_error_Prev = x_pos
        ver_error_Prev = y_pos
        
    else:
        print('no face')
    
    #cv2.imshow('frame',frame)
        
capture.release()
cv2.destroyAllWindows()
pi.stop()


