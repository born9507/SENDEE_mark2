from multiprocessing import Process
import cv2
import random
import time

def emo2reaction(name, emotion):
    if emotion == 'Neutral':
        if name != "unknown": #아는 사람
            if random.randrange(100) < 50:
                display("neutral1", name, emotion) #무표정
            else:
                display("neutral2", name, emotion) #눈깜박
                display("neutral1", name, emotion)

        else:  #모르는 사람
            if random.randrange(100) < 50:
                display("neutral1", name, emotion)
            else:
                display("neutral3", name, emotion) #물음표
                display("neutral1", name, emotion)
    

    elif emotion == 'Angry':
        if name != "unknown": #아는 사람
            if random.randrange(100) < 50:
                display("fear2", name, emotion) #인상쓰기
            else:
                display("angry2", name, emotion) #극대노

        else:  #모르는 사람
            if random.randrange(100) < 80:
                display("angry1", name, emotion) #쫄음
            else:
                display("neutral1", name, emotion)


    elif emotion == 'Sad':
        if random.randrange(100) < 50:
            display("sad1", name, emotion)  #한숨
        else:
            display("sad2", name, emotion)  #울음


    ## 멋쩍은 웃음 만들어줘라!!!
    elif emotion == 'Happy':
        #아는 사람
        if name != "unknown": 
            display("happy1", name, emotion)  #많이 행복

        #모르는 사람
        else: 
            if random.randrange(100) < 80:
                display("neutral3", name, emotion) #happy3 어색한 웃음으로 대체
            else:
                display("neutral3", name, emotion)  #물음표

    elif emotion == 'Surprised':
        if name != "unknown": #아는 사람
            display("surprised2", name, emotion)

        else:  #모르는 사람
            display("surprised1", name, emotion)

    elif emotion == 'Fearful':
        display("fear1", name, emotion)


def display(filename, name, emotion): 
    cap = cv2.VideoCapture(f"display/{filename}.gif")

    while True:
        ret, frame = cap.read()
        if ret==False: break

        #뉴트럴만 좀 빨리 재생하도록
        #프레임당 33ms 기다리고 다음 프레임 재생
        if filename == "neutral1":
            cv2.waitKey(11)
        else:
            cv2.waitKey(33)
    
        ##인식 되고, 알때
        ##인식 되고, 모를때
        ##인식 안될때
        font = cv2.FONT_HERSHEY_DUPLEX

        if name =="noone":
            if emotion == "fun":
                cv2.putText(frame, f"Anybody there..?", (30, 450), font, 1.0, (0, 0, 0), 1)
            else:
                cv2.putText(frame, f"Anybody there..?", (30, 450), font, 1.0, (0, 0, 0), 1)
        
        elif name == "unknown": 
            cv2.putText(frame, f"Who are you?? You look {emotion}", (30, 450), font, 1.0, (0, 0, 0), 1)
        
        else:
            name_ = name.capitalize()
            cv2.putText(frame, f"{name_}, you look {emotion}", (30, 450), font, 1.0, (0, 0, 0), 1)
        
        #재생 되는 순간
        cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.imshow("window", frame)



def noface():
    name = "noone"
    emotion = "noone"
    rand = random.randrange(100)
    if rand < 50:     
        display("neutral1", name, emotion) #무표정
    elif 50<=rand<60:
        emotion = "fun"
        display("happy2", name, emotion)  #비웃음
    else:
        display("neutral2", name, emotion) #눈깜박
        display("neutral1", name, emotion) #무표정

def randMove(pi, randomNumber):
    # if not isMove:
    #     isMove = True
    if randomNumber%8 == 1:
        left_arm(pi)
    elif randomNumber%8 == 2:
        right_arm(pi)
    elif randomNumber%8 == 3:
        doridori(pi)
    elif randomNumber%8 == 4:
        nodnodnod(pi)
        # isMove = False

def left_arm(pi): 
    la = 13 
    ra = 19  
    
    right_mindc = 1700
    right_maxdc = 950
    right_interval = (right_maxdc - right_mindc)/40

    left_mindc = 950
    left_maxdc = 1900
    left_interval = (left_maxdc - left_mindc)/40

    for j in range(0, 3):
        for i in range(1, 41):
            rduty = i * right_interval + right_mindc
            lduty = i * left_interval + left_mindc
            pi.set_servo_pulsewidth(la, lduty)
            pi.set_servo_pulsewidth(ra, rduty)
            time.sleep(0.01)

        for i in range(41, 1, -1):
            rduty = i * right_interval + right_mindc
            lduty = i * left_interval + left_mindc
            pi.set_servo_pulsewidth(la, lduty)
            pi.set_servo_pulsewidth(ra, rduty)
            time.sleep(0.01)

    pi.set_servo_pulsewidth(la, 0)
    pi.set_servo_pulsewidth(ra, 0)

def right_arm(pi): 

    la = 13 
    ra = 19  

    right_mindc = 1700
    right_maxdc = 950
    right_interval = (right_maxdc - right_mindc)/40

    left_mindc = 950
    left_maxdc = 1900
    left_interval = (left_maxdc - left_mindc)/40

    for j in range(0, 3):
        for i in range(1, 41):
            rduty = i * right_interval + right_mindc
            lduty = i * -left_interval + left_maxdc
            pi.set_servo_pulsewidth(la, lduty)
            pi.set_servo_pulsewidth(ra, rduty)
            time.sleep(0.01)

        for i in range(1, 41):
            rduty = i * -right_interval + right_maxdc
            lduty = i * left_interval + left_mindc
            pi.set_servo_pulsewidth(la, lduty)
            pi.set_servo_pulsewidth(ra, rduty)
            time.sleep(0.01)

    pi.set_servo_pulsewidth(la, 0)
    pi.set_servo_pulsewidth(ra, 0)


def doridori(pi): 
    bm = 5

    body_mindc = 600
    body_maxdc = 2400
    body_interval = (body_maxdc - body_mindc)/80

    for j in range(0, 2):
        # for i in range(36, 56):
        #     bduty = round(i * body_interval + body_mindc)
        #     pi.set_servo_pulsewidth(bm, bduty)
        #     time.sleep(0.1)
        # time.sleep(0.4)

        for i in range(56, 36, -1):
            bduty = round(i * body_interval + body_mindc)
            pi.set_servo_pulsewidth(bm, bduty)
            print(pi.get_servo_pulsewidth(bm))
            time.sleep(0.1)
        time.sleep(0.4)

    pi.set_servo_pulsewidth(bm, 0)

def nodnodnod(pi):
    hm = 6

    head_mindc = 1200
    head_maxdc = 1700
    head_interval = (head_maxdc - head_mindc)/40

    for j in range(0, 3):
        for i in range(11, 21):
            hduty = i * head_interval + head_mindc
            pi.set_servo_pulsewidth(hm, hduty)
            time.sleep(0.01)

        for i in range(21, 11, -1):
            hduty = i * head_interval + head_mindc
            pi.set_servo_pulsewidth(hm, hduty)
            time.sleep(0.01)

    pi.set_servo_pulsewidth(hm, 0)