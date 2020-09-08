import time
import requests
import urllib.request
import numpy as np
import json
from datetime import datetime
from ftplib import FTP

URL = "http://sangw.iptime.org"

def internet_on():
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
                
                with open("emotions/emotions.json", "w") as f:
                    json.dump(time_total_emotions, f, indent=2)

                for name in known_face_names:
                    total_emotions[name] = [0,0,0,0,0,0,0]
                
                check_min = time.gmtime(time.time())
                
            
            # 5분마다 전송 시도
            if time.time() - webtime > 2: 
                # 네트워크 연결 확인    
                if internet_on() == True:
                    print("Internet Connected")
                    # print(json.dumps(time_total_emotions))
                    requests.post(URL, data=json.dumps(time_total_emotions))
                        
                    webtime = time.time()
                
                # 전송 실패시 5초 뒤에 다시 시도
                else:
                    print("No network")
                    webtime = time.time() - 5
                pass
        except ConnectionError:
            pass
        except KeyboardInterrupt:
            break
        except:
            pass