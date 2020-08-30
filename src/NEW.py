from multiprocessing import Process, Value, Array
import time, cv2
import numpy as np
####RPi####
# import motordrive
# import RPi.GPIO as GPIO

# 웹캠에서는 얼굴 인식하고 변수에 저장하는 것만
def webcam(
    # O_isDetected, O_rgb_for_face, O_gray_for_emotion, O_face_locations
    ):
    HEIGHT = 360
    WIDTH =  480

    ## 윈도는 0, 라즈베리파이는 -1 ##
    # capture = cv2.VideoCapture(-1)
    capture = cv2.VideoCapture(0)

    time.sleep(1)

    capture.set(3, WIDTH)
    capture.set(4, HEIGHT)
    capture.set(10, 60) #brightness
    capture.set(11, 60) #contrast
    capture.set(21, 0.25) #auto exposure
    #capture.set(5, 60)

    hor_error_Sum = 0
    hor_error_Prev = 0
    ver_error_Sum = 0
    ver_error_Prev = 0
    past_dc = 4
    ##########

    face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt2.xml')
    info = ''
    font = cv2.FONT_HERSHEY_SIMPLEX

    isDetected = False
    cycle_time = 0.05 #1프레임당 시간

    while True:
        start = time.time()
        ret, frame = capture.read()
        if not ret: break

        # 자르지 않은 이미지 전체 (흑백: 얼굴인식,  컬러: 표정 인식)
        # 생각해보면 사람도 얼굴만 따로 떼놓고 인식하지 않으니 타당함
        rgb_for_face = frame[::]
        gray_for_emotion = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray_for_emotion, 1.3, 5) 
        #문제는 이부분이 수평만 인지한다, 기울어진 얼굴도 인식하도록

        cv2.putText(frame, info, (5, 15), font, 0.5, (255, 0, 255), 1)
        #피클 파일에 쓰는 부분을 없애고, 프로세스간에 변수를 공유하도록 한다.
        if len(faces)>1:
            face_list = []

            for face in faces:
                face_list.append(face[2])
            #face[2] 가 얼굴의 width, width가 제일 큰 id 를 뽑아서, faces 에서 그 id에 해당하는 face를 nparray 로 다시 묶음
            faces = np.array([faces[np.argmax(np.array(face_list))]])

        if len(faces)==1:
            if isDetected == False:
                isDetected = True   #
            # for (x, y, w, h) in faces:
            [x, y, w, h] = faces[0]
            
            face_locations = np.array([[y, x+w, y+h, x]])

            ##밖으로 변수 빼주기
            # O_face_locations.send(face_locations)

            (top, right, bottom, left) = (y, x+w, y+h, x)
            cv2.rectangle(frame, (left, top), (right, bottom), (0,0,255), 2)
        else:     # No face detected
            if isDetected==True:
                # motordrive.headsleep()
                isDetected = False #사람없으면 True

        frame = cv2.flip(frame, 1)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'): break

        # print("time :", time.time() - start)
        if (time.time() - start) < cycle_time:
            time.sleep(cycle_time - (time.time() - start))
        
        # O_isDetected.send(isDetected)
        # O_isDetected.close

        # O_rgb_for_face.send(rgb_for_face)
        # O_rgb_for_face.close
        
        # O_gray_for_emotion.send(gray_for_emotion)
        # O_gray_for_emotion.close
        
        print(f"isDetected: {type(int(isDetected))}")
        print(f"rgb_for_face: {type(list(rgb_for_face))}  len: {len(list(rgb_for_face))}")
        print(f"gray_for_emotion: {type(list(gray_for_emotion))}  len: {len(list(gray_for_emotion))}")
        if len(faces)==1:
            print(f"face locations: {type(list(face_locations))}  len: {len(list(face_locations))}")
        print('\n')


def face_tracking():
    pass
def armMove():  
    pass 
def test():
    while True:
        time.sleep(1)     
        print("test")

if __name__ == '__main__':
    # Value 나 Array 로 공유하는 데이터들은 모두 _ 를 앞에 붙이겠다
    _isDetected = Value('i', 0)
    _rgb_for_face = Array('d', [0 for i in range(360)])
    _gray_for_emotion = Array('d', [0 for i in range(360)])
    _face_locations = Array('d', [0])
    Process(target=webcam, args=( )).start()
    # Process(target=test, args=()).start()
    
    # print(parent_conn.recv())
    # Process(target=test, args=(isDetected)).start()
    # print(isDetected)

    # Process(target=test3).start()
