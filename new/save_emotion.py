import time
import requests
import urllib.request

URL = "http://sangw.iptime.orag"

def internet_on():
    try:
        urllib.request.urlopen(URL, timeout=1)
        return True
    except urllib.request.URLError as err: 
        return False

def save_emotion(is_detected, emotion, emotion_total, ):
    now = time.gmtime(time.time())
    webtime = time.time()
    # 1분 동안의 정보를 다 합산하여 1분에 한번씩 저장 - 나중에 구현 
    # 우선은 몇초동안 각각의 감정이 나타났는지만 저장, emotion_total 변수 구현
    # emotion_total 은 서버로 전송 후 초기화
    # 서버에서는 받아서 합산하도록
    
    while True:
        if time.gmtime(time.time()).tm_sec != now.tm_sec: # 1 
            if is_detected.value == 1:
                print("detected")
                # year = now.tm_year
                # month = now.tm_mon 
                # day = now.tm_mday
                # hour = now.tm_hour
                # minute = now.tm_min
                # sec = now.tm_sec
                # wday = now.tm_wday
                emotion_total.array[:] = emotion_total.array[:] + emotion.array[:]
                # print("now", emotion.array)
                # print("total: ",emotion_total.array)
            now = time.gmtime(time.time()) 
        
        # 5분마다 전송 시도
        if time.time() - webtime > 10: 
            # 네트워크 연결 확인    
            if internet_on() == True:
                print("Internet Connected")
                # r = requests.post(URL, data={})
                webtime = time.time()
            
            # 전송 실패시 5초 뒤에 다시 시도
            else:
                print("No network, Connect to WiFi")
                webtime = time.time() - 5
            pass