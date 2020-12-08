import cv2
import numpy as np
import time

capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('../haar/haarcascade_frontalface_alt2.xml')
is_detected = 0
count = 0

while True:
    start = time.time()

    ret, frame = capture.read()
    if not ret: break
    gray = frame
    gray = cv2.flip(gray, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, dsize=(400, 300), interpolation=cv2.INTER_AREA)


    # if count % 3 == 0:

    # elif count % 3 == 1:
    
    # elif count % 3 == 2:

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # faces = []

    # 아니면 이것처럼 그냥 해상도를 낮추는 것도 한가지 방법

    height, width = gray.shape
    matrix_l = cv2.getRotationMatrix2D((width/2, height/2), 30, 1)
    gray_l = cv2.warpAffine(gray, matrix_l, (width, height))
    matrix_r = cv2.getRotationMatrix2D((width/2, height/2), -30, 1)
    gray_r = cv2.warpAffine(gray, matrix_r, (width, height))
    faces_l = face_cascade.detectMultiScale(gray_l, 1.3, 5)
    faces_r = face_cascade.detectMultiScale(gray_r, 1.3, 5)
    # 연산이 3배니까 연산 속도가 3배로 줄어든다
    # 정방향에서 인식이 안될 경우에만 오른쪽, 그 다음 왼쪽 계산하도록 하면 어떨까
    # 아니면 멀티 프로세싱?
    # 아니면 랜덤하게 1/3 확률로 검출?
    # 또는 3번 나눠서 계산? 매 카운트마다 돌아가면서, 이러면 위치 정보는 0.1초마다 들어오지만, 프레임 자체는 매 카운트마다 저장된다.
    # 3번 카운트동안 모두 검출이 안되면 is_detected 를 0, 한번이라도 검출이 되면 1 을 유지
    # location 값도 초기화 안함


    if len(faces)>1:
        face_list = []
        for face in faces:
            face_list.append(face[2])
        faces = np.array([faces[np.argmax(np.array(face_list))]])

    if len(faces)==1:
        is_detected = 1 
        [x, y, w, h] = faces[0]
        face_locations = np.array([[y, x+w, y+h, x]])

        (top, right, bottom, left) = (y, x+w, y+h, x)
        print(top, right, bottom, left)
        cv2.rectangle(gray, (left, top), (right, bottom), (0,0,255), 2)

    else:
        is_detected = 0

    cv2.imshow("frame", gray)
    if cv2.waitKey(1) == ord('q'): break

    print("process time: ", time.time()-start)

    # print(is_detected)

capture.release() 
cv2.destroyAllWindows()