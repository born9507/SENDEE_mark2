from multiprocessing import Process, Value
import multiprocessing
from shared_ndarray import SharedNDArray
import numpy as np
import time
import cv2
import json
import face_recognition

# model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

#recognition
from keras.models import load_model

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

######################################################################

# face_tracking 함수는 얼굴 위치를 계산해서 내보냄(2명 이상이면 한명만 골라서)
# 입력: frame, is_running  출력: face_location, is_detected
def face_location(frame, face_location, is_running, is_detected, ):
    face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt2.xml')
    while True:
        uint8 = frame.array.astype(np.uint8)
        gray = cv2.cvtColor(uint8, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # --------------------추가된 부분--------------------
        # 얼굴 회전할 각도 설정하는 array, 테스트 결과 +-15도까지는 haar cascade filter가 동작함
        # 각각의 각도만큼 회전된 이미지를 바탕으로 얼굴 위치 검출, faces array에 추가
        
        # rotationAngleArray = [-30, 0, 30]
        # faces = []
        # (height, width) = gray.shape[:2]
        # imageCenter = (width / 2, height / 2)

        # for angle in rotationAngleArray:
        #     rotMatrix = cv2.getRotationMatrix2D(imageCenter, angle, 1)
        #     Cos = rotMatrix[0, 0]
        #     Sin = rotMatrix[0, 1]
        #     rotatedWidth = int(height * abs(Sin) + width * abs(Cos))
        #     rotatedHeight = int(height * abs(Cos) + width * abs(Sin))
        #     rotMatrix[0, 2] += rotatedWidth / 2 - imageCenter[0]
        #     rotMatrix[1, 2] += rotatedHeight / 2 - imageCenter[1]
        #     targetFrame = cv2.warpAffine(gray, rotMatrix, (rotatedWidth, rotatedHeight))

        #     faceLocations = face_cascade.detectMultiScale(gray, 1.3, 5)
            
        #     if faceLocations != ():
        #         for location in faceLocations:
        #             (x, y, w, h) = location
        #             midx = x + w/2
        #             midy = y + h/2
        #             x = int(Cos * (midx - rotatedWidth/2) - Sin * (midy - rotatedHeight/2) + width/2 - w/2)
        #             y = int(Sin * (midx - rotatedWidth/2) + Cos * (midy - rotatedHeight/2) + height/2 - h/2)
        #             print("face found in angle", angle)
        #             location[0] = x
        #             location[1] = y

        #             faces.append(location)
                    
        # -----------------------------------------------
        # 두명 이상이면 얼굴 큰 사람
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
            (top, right, bottom, left) = (y, x+w, y+h, x)
            # cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)

        # No face detected
        else:
            if is_detected.value == 1:
                is_detected.value = 0
            else:
                pass
        print(face_locations)
        # print(is_detected.value)

def face_tracking(face_location, ):
    np.array([[y, x+w, y+h, x]]) = face_location.array
    x_pos = x + w/2
    y_pos = y + h/2 
    # 모터 제어 파트 추가 


######################################################################################

# 얼굴 인식은 is_detected 일때만 돌아가도록 하자
# 얼굴 인식과 표정 인식을 멀티프로세싱을 돌려 빠르게 처리하도록
# 인식하고, 인식 횟수가 몇회 이상이면 
def recognition(frame, face_location_, emotion, is_detected, ):
    model = model()
    model.load_weights('model/model.h5')
    # 아는 
    while True:
        if is_detected.value == 1:
            rgb = frame.array.astype(np.uint8)
            face_location = face_location_.array.astype(np.uint8)
            face_reco = Process(target=face_reco, args=(rgb, face_location, )) # 이 부분이 문제.
            # 함수 안에서 다른 함수를 호출할 수는 있지만, 멀티프로세싱으로 돌리지는 못하나? 되는데... 일단 다른것부터
            face_reco.start()
            face_reco.join()
            face_emo = Process(target=face_emo, args=(rgb, face_location, model, ))
            face_emo.start()
            face_emo.join()
        else:
            time.sleep(0.03)
            pass

def face_reco(rgb, face_location, ):
    with open("face/face_list.json", "r") as f:
        face_list = json.load(f)
    
    known_face_names = face_list.keys() # list
    known_face_encodings = np.array(face_list.values()) # numpy.ndarray
    print(known_face_encodings)
    
    ##불러온 파일 이용해서 인코딩 구한다
    face_encoding = face_recognition.face_encodings(rgb, face_location)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding[0])
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding[0])
    best_match_index = np.argmin(face_distances)

    if matches[best_match_index]:
        name = known_face_names[best_match_index]
    else:
        name = "unknown"
    
    print(name)
    return name

def face_emo(rgb, face_location, model,):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    
    for (top, right, bottom, left) in face_location:
        roi_gray = gray_for_emotion[top:bottom, left:right]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        # cv2.imwrite('cropped.png', roi_gray)
        
        if len(prediction) != 0:
            prediction = prediction[0]
            prediction = np.rint(prediction/sum(prediction)*100) # %
            return prediction

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

############################################################################################

def model():    
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    return model
  
############################################################################

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

if __name__ == "__main__":
    try:
        save_img()
        frame = SharedNDArray((480, 640, 3))
        face_location = SharedNDArray((1,4))
        print(face_location.array)
        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

        view_running = Value('i', 1)
        face_tracking_running = Value('i', 1)
        recognition_running = Value('i', 1)
        is_detected = Value('i', 0)
        emotion = Value('i', 4)

        
        view = Process(target=view, args=(frame, view_running, ))
        face_tracking = Process(target=face_tracking, args=(frame, face_location, face_tracking_running, is_detected, ))
        recognition = Process(target=recognition, args=(frame, face_location, emotion, is_detected, ))

        view.start()
        face_tracking.start()
        recognition.start()

        while True:
            # is_running 제어하기
            pass
        
    finally:
        frame.unlink()
        face_location.unlink()
