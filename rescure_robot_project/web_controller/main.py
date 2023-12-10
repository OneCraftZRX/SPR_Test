# -*- coding: utf-8 -*-
from flask import Flask, render_template, Response
from flask.sessions import NullSession
from flask import request
import time
# import serial
import camera
import time

app = Flask(__name__)
# 配置摄像头
video_task = camera.Video_task(0,480,640)
# 配置串口
# ser = serial.Serial("/dev/ttyS3", 9600)
crab=0

def ButtonControl(mode,action=0):
    global crab
    #数据协议：[0]头0xff,[1]抛线器启用信号,[2]控制前后平移,[3]控制左右平移，[4]控制左右旋转（默认值10度），[5]控制形态变更，[6]控制物资盖子，[7]控制钻孔机上下,[8]校验位,[9]尾0xfe
    #0x01代表正1，0x02代表负1，正1代表前进或左方向或打开，负1代表后退或右方向或关闭
    servdict={'throw':1,'ws':2,'ad':3,'rotate':4,'shape':5,'cover':6,'drill':7}
    servdata=[0xff,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xfe]
    if mode=='stop':
        # cmd=bytearray(servdata)
        # ser.write(servdata)
        print(servdata)
        print("stop")
    else:
        if action<0:
            action=-action+1
            servdata[servdict[mode]]+=action
            print("button_mode=",mode,'and action = ',action-3)
        elif action==0:
            print("action=0, no command")
        else:
            if mode=='shape':
                if crab==1:
                    action=0
                    crab=0
                else:
                    action=1
                    crab=1
            if mode=='ad':
                servdata[3]+=crab
            print("crabmode = ",crab)
            servdata[servdict[mode]]+=action
            print("button_mode=",mode,'and action = ',action)
        servdata[8]=servdata[1]+servdata[2]+servdata[3]+servdata[4]+servdata[5]+servdata[6]+servdata[7]
        # cmd=bytearray(servdata)
        # ser.write(servdata)
        print('\ncmd = ',servdata,"\n")
    time.sleep(0.05)

def GamepadControl(operation=[]):
    #数据协议：[0]头0xff,[1]抛线器启用信号,[2]控制前后平移,[3]控制左右平移，[4]控制左右旋转（默认值10度），[5]控制形态变更，[6]控制物资盖子，[7]控制钻孔机上下,[8]校验位,[9]尾0xfe
    #0x01代表正1，0x02代表负1，正1代表前进或左方向或打开，负1代表后退或右方向或关闭
    servdata=[0xff,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xfe]
    for i in range(len(operation)):
        if operation[i]<0:
            operation[i]=-operation[i]+1
    if len(operation)!=7:
        print("operation length error")
    else:
        for i in range(len(operation)):
            servdata[i+2]=operation[i]
    
    servdata[8]=servdata[1]+servdata[2]+servdata[3]+servdata[4]+servdata[5]+servdata[6]+servdata[7]
    
    # cmd=bytearray(servdata)
    # ser.write(servdata)
    print('gamepad_mode\ncmd = ',servdata,"\n")
    time.sleep(0.05)

Servo_9 = 90
gamepad_flag = 0

@app.route('/')  # 主页
def index():
    # jinja2模板，具体格式保存在index.html文件中
    return render_template('index.html')

@app.route('/video_feed/')  # 这个地址返回视频流响应
def video_feed():
    video_task.num += 1
    return Response(video_task.video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/gps_feed/')
def gps_feed():
    return Response(video_task.gps_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_button/', methods=['GET'])
def get_button():
    if request.method == 'GET':
        button_num = request.args.get('num')
        global gamepad_flag
        if gamepad_flag == 0:
            if button_num == "1":
                ButtonControl('ws', 1)
            elif button_num == "2":
                ButtonControl('ws', -1)
            elif button_num == "3":
                ButtonControl('ad', 1)
            elif button_num == "4":
                ButtonControl('ad', -1)
            elif button_num == "5":
                ButtonControl('rotate', 1)
            elif button_num == "6":
                ButtonControl('rotate', -1)
            elif button_num == "7":
                ButtonControl('shape', 1)
            elif button_num == "8":
                ButtonControl('drill', 1)
            elif button_num == "9":
                ButtonControl('throw', 1)
            elif button_num == "10":
                ButtonControl('drill', -1)
            elif button_num == "11":
                ButtonControl('cover', -1)
            elif button_num == "12":
                ButtonControl('cover', 1)
            elif button_num == "13":
                ButtonControl('stop', 1)
    return "ok"

video_task.open_cam = False

@app.route('/get_switch/', methods=['GET'])
def get_switch():
    if request.method == 'GET':
        switch_num = request.args.get('num')
        switch_state = request.args.get('state')
        global gamepad_flag
        
        if switch_num == "1":
            print("switch_state=", switch_state)
            camera_control(switch_state)

        if switch_num == "2":
            if switch_state == "true":
                gamepad_flag = 1
            else:
                gamepad_flag = 0

        print(switch_num + switch_state)
    return "ok"

@app.route('/get_gamepad/', methods=['GET'])
def get_gamepad():
    if request.method == 'GET':
        ws, ad, rotate, shape, cover, drill = 0, 0, 0, 0, 0, 0
        
        button_0 = int(request.args.get('button_0'))
        button_2 = int(request.args.get('button_2'))
        drill = abs(button_0-button_2)

        button_1 = int(request.args.get('button_1'))
        button_3 = int(request.args.get('button_3'))
        if button_1 == button_3:
            rotate = 0
        else:
            if button_1 == 1:
                rotate = -1
            elif button_3 == 1:
                rotate = 1

        button_4 = int(request.args.get('button_4'))
        button_5 = int(request.args.get('button_5'))
        if button_4 == button_5:
            cover = 0
        else:
            if button_4 == 1:
                cover = 1
            elif button_5 == 1:
                cover = -1
        button_6 = int(request.args.get('button_6'))
        button_7 = int(request.args.get('button_7'))

        button_8 = int(request.args.get('button_8'))
        shape = button_8

        button_9 = int(request.args.get('button_9'))
        throw = button_9

        button_10 = int(request.args.get('button_10'))
        button_11 = int(request.args.get('button_11'))

        l_x = request.args.get('l_x')
        l_y = request.args.get('l_y')

        if l_y=='1':
            ws=-1
        
        if l_y=='-1':
            ws=1
        
        if l_x=='1':
            ad=-1
        
        if l_x=='-1':
            ad=1
        
        GamepadControl([throw,ws,ad,rotate,shape,cover,drill])
    return "ok"

def camera_control(state):
    if state == "true":
        # temp=1
        video_task.open_cam = True
    if state == "false":
        # temp=0
        video_task.open_cam = False

if __name__ == '__main__':
    video_task.start()
    app.run(host="0.0.0.0", port=5000, debug=False)
    video_task.join()