from multiprocessing import Process, Value
import multiprocessing
from shared_ndarray import SharedNDArray
import numpy as np
import time
import cv2
# import face_recognition

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

# face_tracking 함수는 얼굴 위치를 받아서 모터 제어까지(기존의 )
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
    print(face_location.array)
    x_pos = x + w/2
    y_pos = y + h/2 
    # 모터 제어 파트 추가 


def recognition(frame):
    while True:
        pass
  

if __name__ == "__main__":
    try:
        frame = SharedNDArray((480, 640, 3))

        view_running = Value('i', 1)
        face_tracking_running = Value('i', 1)
        is_detected = Value('i', 0)
        
        view = Process(target=view, args=(frame, view_running, ))
        face_tracking = Process(target=face_tracking, args=(frame, face_tracking_running, is_detected, ))
        
        view.start()
        face_tracking.start()

        view.join()
        face_tracking.join()
        
    finally:
        frame.unlink()
