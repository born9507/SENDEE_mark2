from multiprocessing import Process, Value
import multiprocessing
from shared_ndarray import SharedNDArray
import numpy as np
import time
import cv2
# import face_recognition

# model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

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

# face_tracking 함수는 얼굴 위치를 받아서 모터 제어까지
def face_tracking(frame, face_location, is_running, is_detected, ):
    face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt2.xml')
    while True:
        uint8 = frame.array.astype(np.uint8)
        gray = cv2.cvtColor(uint8, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # cv2.putText(frame, info, (5, 15), font, 0.5, (255, 0, 255), 1)

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

            x_pos = x + w/2
            y_pos = y + h/2 
            #### 여기 아래에다 모터 제어 파트 쓰면 될 듯 ####
        
        # No face detected
        else:
            if is_detected.value == 1:
                is_detected.value = 0
            else:
                pass
        # print(is_detected.value)

######################################################################################

# 얼굴 인식은 is_detected 일때만 돌아가도록 하자
# 얼굴 인식과 표정 인식을 멀티프로세싱을 돌려 빠르게 처리하도록
# 인식하고, 인식 횟수가 몇회 이상이면 
def recognition(frame, face_location_, is_detected, ):
    while True:
        if is_detected.value == 1:
            rgb = frame.array.astype(np
            .uint8)
            face_location = face_location_.array.astype(np.uint8)
            print(face_location)
            Process(target=face_reco, args=(rgb, face_location, )).start()
            Process(target=face_emo, args=(rgb, face_location, )).start()
        else:
            time.sleep(0.03)
            pass

def face_reco():
    ##rgb_for_face 불러오기
    with open("pkl/rgb_for_face.pkl", "rb") as file:
        rgb_for_face = pickle.load(file)
        file.close()
    ##face_locations 불러오기
    with open("pkl/face_locations.pkl", "rb") as file:
        face_location = pickle.load(file)
        file.close()
    with open("pkl/known_face_names.pkl", "rb") as file:
        known_face_names = pickle.load(file)
        file.close()
    with open("pkl/known_face_encodings.pkl", "rb") as file:
        known_face_encodings = pickle.load(file)
        file.close()
    
    ##불러온 파일 이용해서 인코딩 구한다
    face_encoding = face_recognition.face_encodings(rgb_for_face, face_location)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding[0])
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding[0])
    best_match_index = np.argmin(face_distances)

    if matches[best_match_index]:
        name = known_face_names[best_match_index]
    else:
        name = "unknown"
    
    return name

def face_emo(model):
    with open("pkl/gray_for_emotion.pkl", "rb") as file:
        gray_for_emotion = pickle.load(file)
        file.close()
    with open("pkl/face_locations.pkl", "rb") as file:
        face_location = pickle.load(file)
        file.close()

    # model = load_model("models/20200622_2242_model.h5")
    model = model
    # emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    for (top, right, bottom, left) in face_location:
        roi_gray = gray_for_emotion[top:bottom, left:right]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        # cv2.imwrite('cropped.png', roi_gray)
        
        if len(prediction) != 0:
            prediction = prediction[0]
            prediction = np.rint(prediction/sum(prediction)*100)# %
            return prediction

##################################################################################

def expression():
    # Process(target=).start()
    # Process(target=).start()
    # Process(target=).start()
    # Process(target=).start()
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

if __name__ == "__main__":
    try:
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
        recognition = Process(target=recognition, args=(frame, face_location, is_detected, ))

        view.start()
        face_tracking.start()
        recognition.start()

        while True:
            # is_running 제어하기
            pass
        
    finally:
        frame.unlink()
        face_location.unlink()
