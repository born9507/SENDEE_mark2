from multiprocessing import Process, Value
import multiprocessing
from shared_ndarray import SharedNDArray
import numpy as np
import time
import cv2
import json
import asyncio
import face_recognition
import model.model as md
import os
import pigpio

# view 함수는 카메라로 촬영한 것을 frame 변수에 저장(np.array, dtype=float64)
# 얼굴 위치도 계산해서 내보냄(2명 이상이면 한명만 골라서)

# 프로세스 1
def view(frame, HEIGHT, WIDTH, face_location, is_running, is_detected, ):

    capture = cv2.VideoCapture(-1)
    capture.set(3, WIDTH.value)
    capture.set(4, HEIGHT.value)

    face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt2.xml')
    # info = ''
    # font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        if is_running.value==1:
            ret, frame_ = capture.read()
            if not ret: break
            #frame_ 의 해상도를 낮춰서 haar 에 넣어볼까?
            gray = cv2.cvtColor(frame_, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            # cv2.putText(frame_, info, (5, 15), font, 0.5, (255, 0, 255), 1)
            
            frame.array[:] = frame_[:] # export

            if len(faces)>1:
                face_list = []
                for face in faces:
                    face_list.append(face[2])
                faces = np.array([faces[np.argmax(np.array(face_list))]])

            if len(faces)==1:
                if is_detected.value == 0:
                    is_detected.value = 1 
                [x, y, w, h] = faces[0]
                face_locations = np.array([[y, x+w, y+h, x]])
                face_location.array[:] = face_locations[:] # export
                # (top, right, bottom, left) = (y, x+w, y+h, x)
                # cv2.rectangle(frame_, (left, top), (right, bottom), (0,0,255), 2)
            # No face detected
            else:
                if is_detected.value == 1:
                    is_detected.value = 0
                else:
                    pass
            # frame = cv2.flip(frame, 1)
            # cv2.imshow("frame", frame_)
            # if cv2.waitKey(1) == ord('q'): break
        else:
            time.sleep(1)
            pass
    capture.release() 
    cv2.destroyAllWindows()


# 프로세스 2
def face_tracking(face_location, is_running, pi, ):
    # 모터 제어 전역변수
    pi = pigpio.pi()
    lm = 5
    rm = 13
    bm = 6
    hm = 19

    head_mindc = 1650
    head_maxdc = 2150
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
    # 모터 제어 파트 추가
    while True:
        for (top, right, bottom, left) in face_location.array:
            x = left
            w = right - left
            y = top
            h = bottom - top
            x_pos = (x + w/2 - 240)/240
            y_pos = (y + h/2 - 160)/160
            # print("x: ", x_pos,"y:", y_pos)
            
            # time.sleep(0.1)
            
        hor_error_Sum = hor_error_Sum + x_pos
        ver_error_Sum = ver_error_Sum + y_pos
        past_ver_dc = headServo(y_pos, 0.05, past_ver_dc, ver_error_Sum, ver_error_Prev, head_mindc, head_maxdc, head_interval, pi,)
        past_hor_dc = bodyServo(x_pos, 0.05, past_hor_dc, hor_error_Sum, hor_error_Prev, body_mindc, body_maxdc, body_interval, pi,)
        hor_error_Prev = x_pos
        ver_error_Prev = y_pos
    

def headServo(error_Now, time, past_dc, error_Sum, error_Prev, head_mindc, head_maxdc, head_interval, pi,):
    Kp = 50
    Ki = 0
    Kd = 0
    
    error = error_Now
    error_sum = error_Sum + error
    error_diff = (error-error_Prev)/time
    
    ctrlval = -(Kp*error + Ki*error_sum*time + Kd*error_diff)
    
    if abs(ctrlval) < 2:
        ctrlval = 0
    ctrlval = round(ctrlval, 1)
           
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
        pi.set_servo_pulsewidth(hm, ctrlval)

    return head_duty

def bodyServo(error_Now, time, past_dc, error_Sum, error_Prev, body_mindc, body_maxdc, body_interval, pi, ):    
    Kp = 50
    Ki = 0
    Kd = 0
    
    error = error_Now
    error_sum = error_Sum + error
    error_diff = (error-error_Prev)/time
    
    ctrlval = -(Kp*error + Ki*error_sum*time + Kd*error_diff)
    
    if abs(ctrlval) < 0.02:
        ctrlval = 0
    ctrlval = round(ctrlval, 1)
           
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
        pi.set_servo_pulsewidth(bm, ctrlval)

    return body_duty

######################################################################################

# 얼굴 인식은 is_detected 일때만 돌아가도록 하자
# 얼굴 인식과 표정 인식을 멀티프로세싱을 돌려 빠르게 처리하도록
# 인식하고, 인식 횟수가 몇회 이상이면 

# 프로세스 3
def recognition(frame, face_location_, name_index, emotion, is_detected, ):
    model = md.model()
    model.load_weights('model/model.h5')
    # 아는 
    while True:
        if is_detected.value == 1:
            start = time.time()

            rgb = frame.array.astype(np.uint8)
            bgr = rgb[:,:,::-1]
            # cv2.imwrite("rgb.jpg", rgb)
            # cv2.imwrite("bgr.jpg", bgr)
            face_location = face_location_.array.astype(np.int16)

            face_reco(rgb, face_location, name_index, )
            face_emo(rgb, face_location, model, emotion, )
            # 여기 왜 비동기 처리가 안되는가
            
            # print("recognition time: ",time.time()-start)
            # print("")
        else:
            time.sleep(0.03)
            pass

def face_reco(rgb, face_location, name_index, ):
    start = time.time()
    with open("face/face_list.json", "r") as f:
        face_list = json.load(f)
    
    known_face_names = list(face_list.keys()) # list
    known_face_encodings = np.array(list(face_list.values())) # numpy.ndarray
    
    ##불러온 파일 이용해서 인코딩 구한다
    face_encoding = face_recognition.face_encodings(rgb, face_location)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding[0])
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding[0])

    # for i in range(len(known_face_names)):
    #     print(f"{known_face_names[i]} : {round((1 - face_distances[i]) / (4 - sum(face_distances)) * 100)}%", end=" ")
    # print("")

    best_match_index = np.argmin(face_distances)

    if matches[best_match_index]:
        name = known_face_names[best_match_index]
        name_index.value = best_match_index
    else:
        name = "unknown"
        name_index.value = -1
    # print(name)
    # print("face-rec time: ", time.time()-start)
    
def face_emo(rgb, face_location, model, emotion, ):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    
    for (top, right, bottom, left) in face_location:
        roi_gray = gray[top:bottom, left:right]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        
        if len(prediction) != 0:
            prediction = prediction[0]
            emotion.array[:] = prediction[:]

##################################################################################

# 클래스를 활용해야 하나? 

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
def left_arm(): pass
def right_arm(): pass
def doridori(): pass


# 언노운 일정 횟수 이상 들어오면 이 함수 실행하도록

def img2encoding():
    with open("face/face_list.json", "r") as f:
        face_list = json.load(f)
    
    names = []
    images = os.listdir("face/img/")
    for image in images:
        name = image.split('.')[0]
        names.append(name)
        if name in face_list.keys():
            pass
        else:
            # 이 부분이 느린 부분이라 if 문에 넣기
            name_image = face_recognition.load_image_file(f"face/img/{image}")
            name_encoding = face_recognition.face_encodings(name_image)[0]
            face_list[name] = name_encoding.tolist()
    
    for key, val in list(face_list.items()):
        if key not in names:
            del face_list[key]

    with open("face/face_list.json", "w") as f:
        json.dump(face_list, f, indent=2)

# cv2 로 사진 캡쳐해서 찍기
def save_img():
    pass

# 해당 사진 지우기(웹 or 앱으로 처리)
def delete_img():
    pass

####################################################################################

def save_emotion(is_detected, emotion):
    now = time.gmtime(time.time())
    # 1분 동안의 정보를 다 합산하여 1분에 한번씩 저장 - 나중에 구현 
    # 우선은 몇초동안 각각의 감정이 나타났는지만 저장, emotion_total 변수 구현
    # emotion_total 은 서버로 전송 후 초기화
    # 서버에서는 받아서 합산하도록
    
    while True:
        if time.gmtime(time.time()).tm_sec != now.tm_sec:
            if is_detected.value = 1:
                year = now.tm_year
                month = now.tm_mon 
                day = now.tm_mday
                hour = now.tm_hour
                minute = now.tm_min
                sec = now.tm_sec
                wday = now.tm_wday
                emotion.array 

            now = time.gmtime(time.time())

    print(time.time() - start)

if __name__ == "__main__":
    try:
        pi = pigpio.pi()
        img2encoding()

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
        # print(face_location.array)
        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

        view_running = Value('i', 1)
        face_tracking_running = Value('i', 1)
        recognition_running = Value('i', 1)
        is_detected = Value('i', 0)
        name_index = Value('i', -1) # -1 이 unknown

        view = Process(target=view, args=(frame, HEIGHT, WIDTH, face_location ,view_running, is_detected, ))
        face_tracking = Process(target=face_tracking, args=(face_location, face_tracking_running, pi, ))
        recognition = Process(target=recognition, args=(frame, face_location, name_index, emotion, is_detected, ))

        view.start()
        face_tracking.start()
        recognition.start()

        while True:
            # is_running 제어하기
            # print(is_detected.value)
            # print(emotion.array)
            # print(np.argmax(emotion.array))
            # print(known_face_names[name_index.value])
            # print(emotion_dict[np.argmax(emotion.array)])
            time.sleep(0.5)
            pass
        
    finally:
        frame.unlink()
        face_location.unlink()
