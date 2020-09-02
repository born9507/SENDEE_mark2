import cv2
from picamera import PiCamera
import time

# camera = PiCamera()
# camera.start_preview()
# time.sleep(5)
# camera.stop_preview()
# time.sleep(3)
HEIGHT = 360
WIDTH =  480

capture = cv2.VideoCapture(-1)
capture.set(3, WIDTH)
capture.set(4, HEIGHT)
# capture.set(10, 0.5) #brightness
# capture.set(11, 0.5) #contrast


while True:
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'): break

