import numpy as np
import cv2
import threading
from copy import deepcopy
import time
from gps import *

src = cv2.imread('./static/map.jpg')

#配置GPS方法函数
# def getPositionData(gps):
#     nx = gpsd.next()
#     # For a list of all supported classes and fields refer to:
#     # https://gpsd.gitlab.io/gpsd/gpsd_json.html
#     if nx['class'] == 'TPV':
#         latitude = getattr(nx,'lat', "Unknown")
#         longitude = getattr(nx,'lon', "Unknown")
#         # print("Your position: lon = " + str(longitude) + ", lat = " + str(latitude))
#         return "longitude " + str(longitude) + "; latitude " + str(latitude)
# #配置gps device
# gpsd = gps(mode=WATCH_ENABLE|WATCH_NEWSTYLE)

class Video_task(threading.Thread):
    def __init__(self, camera_id, img_height, img_width):
        super(Video_task, self).__init__()
        self.camera_id = camera_id
        self.img_height = img_height
        self.img_width = img_width
        self.num = 0
        self.open_cam = True
        self.thread_lock = threading.Lock()
        self.thread_exit = False
        self.frame = np.zeros((self.img_height, self.img_width, 1), np.uint8) + 200
        cv2.putText(self.frame, 'NO VIDEO', (110, 260), cv2.FONT_HERSHEY_SIMPLEX, 3, 50, 5)
        self.close_jpeg = cv2.imencode('.jpg', self.frame)[1].tobytes()

    def get_frame(self):
        self.thread_lock.acquire()
        frame = deepcopy(self.frame)
        self.thread_lock.release()
        # 因为opencv读取的图片并非jpeg格式，因此要用motion JPEG模式需要先将图片转码成jpg格式图片
        jpeg = cv2.imencode('.jpg', frame)[1]
        return jpeg.tobytes()

    def video_feed(self):
        num = self.num
        while num == self.num:
            if self.open_cam == True:
                frame = self.get_frame()
                # 使用generator函数输出视频流， 每次请求输出的content类型是image/jpeg
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
                # 降低帧率，减少cpu占用
                cv2.waitKey(50)
            else:
                # 使用generator函数输出视频流， 每次请求输出的content类型是image/jpeg
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + self.close_jpeg + b'\r\n\r\n')
                # 降低帧率，减少cpu占用
                cv2.waitKey(500)

    def gps_feed(self):
        while True:
            # text = getPositionData(gpsd)
            # AddText = src.copy()
            cv2.putText(src, "testtesttest", (0, 80), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 0), 6)
            jpeg = cv2.imencode('.jpg', src)[1]
            # 使用generator函数输出视频流， 每次请求输出的content类型是image/jpeg
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            # 降低帧率，减少cpu占用
            cv2.waitKey(50)

    def run(self):
        cap = cv2.VideoCapture(0)
        while not self.thread_exit:
            if self.open_cam == True:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (self.img_width, self.img_height))
                    frame = cv2.flip(frame,-1)
                    self.thread_lock.acquire()
                    self.frame = frame
                    self.thread_lock.release()
                    cv2.waitKey(10)
                else:
                    self.thread_exit = True
            else:
                cv2.waitKey(10)
        cap.release()
