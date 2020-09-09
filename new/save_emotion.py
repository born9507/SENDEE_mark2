import time
import requests
import urllib.request
import numpy as np
import json
from datetime import datetime
from ftplib import FTP

URL_google = "8.8.8.8"
URL = "http://sangw.iptime.org"

def internet_on():
    try:
        urllib.request.urlopen(URL_google, timeout=1)
        return True
    except urllib.request.URLError as err: 
        return False

def server_on():
    try:
        urllib.request.urlopen(URL, timeout=1)
        return True
    except urllib.request.URLError as err: 
        return False


def save_emotion(is_detected, emotion, name_index,known_face_names, ):
    with open("emotions/emotions.json", "r") as f:
        time_total_emotions = json.load(f)
    
    total_emotions = {}
    for name in known_face_names:
        total_emotions[name] = [0,0,0,0,0,0,0]

    check_sec = time.gmtime(time.time())
    check_min = time.gmtime(time.time())
    webtime = time.time()
    # emotion_total 은 서버로 전송 후 초기화
    # 서버에서는 받아서 합산하도록

    # print(total_emotions)

    while True:
        try:
            if time.gmtime(time.time()).tm_sec != check_sec.tm_sec: # 1 
                if is_detected.value == 1:
                    name = known_face_names[name_index.value]
                    emotion_sum = np.array(total_emotions.get(name))
                    emotion_sum = emotion_sum + emotion.array[:]
                    total_emotions[name] = emotion_sum.tolist()[0]
                    # print(total_emotions)

                check_sec = time.gmtime(time.time())

            if time.gmtime(time.time()).tm_min != check_min.tm_min:
                now_min = datetime.today().strftime("%Y%m%d%H%M")
                print(now_min)
                time_total_emotions[now_min] = total_emotions
                
                # 1 분마다 따로 저장하므로, 1분 지나면 total_emotions 는 다시 초기화 해줘야 다음 시간에 다시 0부터 합
                total_emotions = {}
                for name in known_face_names:
                    total_emotions[name] = [0,0,0,0,0,0,0]
                
                with open("emotions/emotions.json", "w") as f:
                    json.dump(time_total_emotions, f, indent=2)

                for name in known_face_names:
                    total_emotions[name] = [0,0,0,0,0,0,0]
                
                check_min = time.gmtime(time.time())
                
            
            # 5분마다 전송 시도
            if time.time() - webtime > 30: 
                # 네트워크 연결 확인    
                if internet_on():
                    print("Internet Connected")
                    if server_on():
                        print("Server Ready, Send Json ...")
                        res = requests.post(URL, data=json.dumps(time_total_emotions))
                        if res.status_code != 200:
                            print("error code")
                            webtime = time.time() - 5
                        else:
                            print("Success!")
                            # time_total_emotions={}
                            webtime = time.time()
                    else:
                        print("No server connection")
                        webtime = time.time() - 5
                else:
                    print("No network")
                    webtime = time.time() - 5

        except ConnectionError:
            pass
        except KeyboardInterrupt:
            break
        except:
            pass