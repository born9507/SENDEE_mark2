import cv2
import numpy as np

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

            # flip 추가!!!!!!!!!!
            frame_ = cv2.flip(frame_, 0)
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