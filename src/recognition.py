import time
import cv2
import json
import face_recognition
import numpy as np
import model.model as md

def recognition(frame, face_location_, name_index, emotion, is_detected, ):
    model = md.model()
    model.load_weights('model/model.h5')
    # 아는 
    while True:
        if is_detected.value == 1:
            start = time.time()

            rgb = frame.array.astype(np.uint8)
            bgr = rgb[:,:,::-1]
            face_location = face_location_.array.astype(np.int16)

            face_reco(rgb, face_location, name_index, )
            face_emo(rgb, face_location, model, emotion, )
            # 여기 왜 비동기 처리가 안되는가
            
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