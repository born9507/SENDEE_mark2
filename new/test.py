from multiprocessing import Process, Value
import multiprocessing
from shared_ndarray import SharedNDArray
import numpy as np
import time
import cv2
import json
import asyncio
import face_recognition

# from model.model import model
#recognition
# from keras.models import load_model

# view 함수는 카메라로 촬영한 것을 frame 변수에 저장(np.array, dtype=float64)
# 얼굴 위치도 계산해서 내보냄(2명 이상이면 한명만 골라서)

def view(frame, HEIGHT, WIDTH, face_location, is_running, ):

    capture = cv2.VideoCapture(-1)
    capture.set(3, WIDTH.value)
    capture.set(4, HEIGHT.value)

    face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt2.xml')
    # info = ''
    # font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        start = time.time()
        if is_running.value==1:
            ret, frame_ = capture.read()
            if not ret: break

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
                print("detected!")
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
        print("view time: ", time.time()-start)

    capture.release() 
    cv2.destroyAllWindows()


def face_tracking(face_location, is_running, ):
    while True:
        for (top, right, bottom, left) in face_location.array:
            x = left
            w = right - left
            y = top
            h = bottom - top
        x_pos = x + w/2
        y_pos = y + h/2 
    # 모터 제어 파트 추가 


######################################################################################


# 얼굴 인식은 is_detected 일때만 돌아가도록 하자
# 얼굴 인식과 표정 인식을 멀티프로세싱을 돌려 빠르게 처리하도록
# 인식하고, 인식 횟수가 몇회 이상이면 
def recognition(frame, face_location_, emotion, is_detected, ):
    # 아는 
    while True:
        if is_detected.value == 1:
            rgb = frame.array.astype(np.uint8)
            face_location = face_location_.array.astype(np.uint8)
            
            loop = asyncio.get_event_loop()
            loop.run_until_complete(asyncio.gather(
                face_reco(rgb, face_location),
                # face_emo(),
            ))

        else:
            time.sleep(0.03)
            pass


async def face_reco(rgb, face_location, ):
    start = time.time()
    with open("face/face_list.json", "r") as f:
        face_list = json.load(f)
    
    known_face_names = list(face_list.keys()) # list
    known_face_encodings = np.array(list(face_list.values())) # numpy.ndarray
    # print(known_face_names)
    # print(known_face_encodings)
    
    ##불러온 파일 이용해서 인코딩 구한다
    face_encoding = face_recognition.face_encodings(rgb, face_location)
    # print("face_encoding: ",face_encoding[0])
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding[0])
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding[0])
    # print("face_distances: ", face_distances)
    # print("matches: ", matches)
    best_match_index = np.argmin(face_distances)

    if matches[best_match_index]:
        name = known_face_names[best_match_index]
    else:
        name = "unknown"
    
    # print(name)
    # print("face_reco time: ",time.time() - start)
    # print(" ")

# async def face_emo(rgb, face_location, model,):
#     gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    
#     for (top, right, bottom, left) in face_location:
#         roi_gray = gray_for_emotion[top:bottom, left:right]
#         cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
#         prediction = model.predict(cropped_img)
#         # cv2.imwrite('cropped.png', roi_gray)
        
#         if len(prediction) != 0:
#             prediction = prediction[0]
#             prediction = np.rint(prediction/sum(prediction)*100) # %
#             print(prediction)

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

if __name__ == "__main__":
    try:
        # model = model()
        # model.load_weights('model/model.h5')

        HEIGHT = Value('i', 320)
        WIDTH = Value('i', 480)

        save_img()
        frame = SharedNDArray((HEIGHT.value, WIDTH.value, 3))
        face_location = SharedNDArray((1,4))
        # print(face_location.array)
        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

        view_running = Value('i', 1)
        face_tracking_running = Value('i', 1)
        recognition_running = Value('i', 1)
        is_detected = Value('i', 0)
        emotion = Value('i', 4)

        view = Process(target=view, args=(frame, HEIGHT, WIDTH, face_location ,view_running, ))
        face_tracking = Process(target=face_tracking, args=(face_location, face_tracking_running, ))
        recognition = Process(target=recognition, args=(frame, face_location, emotion, is_detected, ))

        view.start()
        face_tracking.start()
        # face_tracking_move.start()
        recognition.start()

        while True:
            # is_running 제어하기
            pass
        
    finally:
        frame.unlink()
        face_location.unlink()
