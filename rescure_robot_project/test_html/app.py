#######################################（树莓派端）读取视频流显示到网页上,后期需要改变显示的位置和大小，设计界面##########################################
#Import necessary libraries
from flask import Flask, render_template, Response
import cv2
# from my_openvino import YOLOV7_OPENVINO
import numpy as np
#Initialize the Flask app
app = Flask(__name__)

camera = cv2.VideoCapture(0)

# yolov7_detector=YOLOV7_OPENVINO("E://yolov5-7.0//weights//yolov5s.onnx",'CPU', False, False)#初始化检测模型
'''
for ip camera use - rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' 
for local webcam use cv2.VideoCapture(0)
'''

def gen_frames():
    while True:
        success, frame = camera.read()  # read the camera frame 

        #result=yolov7_detector.infer_image(frame)#process the image
        result = cv2.resize(frame, (600,480))#reset the shape of the image to adapt the screen size

        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', result)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/')
def index():
    return render_template('index.html')#网页的基本构造

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)