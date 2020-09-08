import pigpio
import time

# 프로세스
def face_tracking(face_location, is_running, pi, ):
    
    bm = 5
    hm = 6

    head_mindc = 1200
    head_maxdc = 1700
    head_interval = (head_maxdc - head_mindc)/40

    body_mindc = 600
    body_maxdc = 2400
    body_interval = (body_maxdc - body_mindc)/40

    hor_error_Sum = 0
    hor_error_Prev = 0
    ver_error_Sum = 0
    ver_error_Prev = 0
    past_hor_dc = 1700
    past_ver_dc = 1600
    # 모터 제어 파트 추가
    while True:
        for (top, right, bottom, left) in face_location.array:
            x = left
            w = right - left
            y = top
            h = bottom - top
            x_pos = (x + w/2 - 240)/240
            y_pos = (y + h/2 - 180)/180
            #print("x: ", x_pos,"y:", y_pos)
            
            # time.sleep(0.1)

        
        hor_error_Sum = hor_error_Sum + x_pos
        ver_error_Sum = ver_error_Sum + y_pos
        past_ver_dc = headServo(y_pos, 0.01, past_ver_dc, ver_error_Sum, ver_error_Prev, head_mindc, head_maxdc, head_interval, pi, hm, )
        past_hor_dc = bodyServo(x_pos, 0.01, past_hor_dc, hor_error_Sum, hor_error_Prev, body_mindc, body_maxdc, body_interval, pi, bm)
        hor_error_Prev = x_pos
        ver_error_Prev = y_pos
    

def headServo(error_Now, waittime, past_dc, error_Sum, error_Prev, head_mindc, head_maxdc, head_interval, pi, hm, ):
    Kp = 0.5
    Ki = 0
    Kd = 0
    
    error = error_Now
    error_sum = error_Sum + error
    error_diff = (error-error_Prev)/waittime
    
    ctrlval = -(Kp*error + Ki*error_sum*waittime + Kd*error_diff)
    
    if abs(ctrlval) < 0.008:
        ctrlval = 0
    ctrlval = round(ctrlval, 1)
           
    head_duty = round(past_dc - head_interval * ctrlval)
    
    if head_duty < head_mindc:
        head_duty = head_mindc
        
    elif head_duty > head_maxdc:
        head_duty = head_maxdc
    
    if head_duty == past_dc:
        print(head_duty, past_dc,'steady')
        head_duty = past_dc
        pi.set_servo_pulsewidth(hm, 0)
    else:
        print(head_duty, past_dc,'move')
        pi.set_servo_pulsewidth(hm, head_duty)
        time.sleep(waittime)
        pi.set_servo_pulsewidth(hm, 0)

    return head_duty

def bodyServo(error_Now, waittime, past_dc, error_Sum, error_Prev, body_mindc, body_maxdc, body_interval, pi, bm, ):    
    Kp = 0.15
    Ki = 0
    Kd = 0.02
    
    error = error_Now
    error_sum = error_Sum + error
    error_diff = (error-error_Prev)/waittime
    
    ctrlval = -(Kp*error + Ki*error_sum*waittime + Kd*error_diff)
    
    if abs(ctrlval) < 0.008:
        ctrlval = 0
    ctrlval = round(ctrlval, 1)
           
    body_duty = round(past_dc - body_interval * ctrlval)
    
    if body_duty < body_mindc:
        body_duty = body_mindc
        
    elif body_duty > body_maxdc:
        body_duty = body_maxdc
    
    if body_duty == past_dc:
        print(body_duty, past_dc,'steady')
        body_duty = past_dc
        pi.set_servo_pulsewidth(bm, 0)
    else:
        print(body_duty, past_dc,'move')
        pi.set_servo_pulsewidth(bm, body_duty)
        time.sleep(waittime)
        pi.set_servo_pulsewidth(bm, 0)

    return body_duty