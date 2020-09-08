import time
# print(now.tm_year, now.tm_mon, now.tm_mday)
# print(now.tm_hour, now.tm_min, now.tm_sec)

now = time.gmtime(time.time())

while True:
    if time.gmtime(time.time()).tm_sec != now.tm_sec:
        print(now)
        now = time.gmtime(time.time())
