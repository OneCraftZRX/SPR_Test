import time
import matplotlib
import matplotlib.pyplot as plt
import argparse
import numpy as np
import cv2
from openvino.runtime import Core, Model
from typing import Tuple, Dict
import random
from ultralytics.yolo.utils import ops
import torch
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import colors
from ultralytics.yolo.utils import ROOT, yaml_load
from ultralytics.yolo.utils.checks import check_yaml
from WebcamVideoStream import WebcamVideoStream#双线程读取视频类
from Predict.kalmanfilter import MovingAverageFilter,KalmanFilter_low,KalmanFilter#卡尔曼预测类
from position_solver import PositionSolver
import math
# from my_serial import SerialPort
from inverse_transformation import CameraCalibration
matplotlib.use('TkAgg')

#储存pitch角度
pitch=[]

#标签和关键点
label_map = []
kpt_shape: Tuple = []

#可视化处理结果
show_result = True

#储存滤波结果的
t_matrix2=[]
t_matrix2_lvbo=[]
t_matrix2_kal=[]
dx_list=[]
dx_list_lvbo=[]
world_position=[]
anti_top=[]


#接收上一帧数据的列表
last_center=0
last_x=[]
last_z=[]


#储存速度列表
speed_x=[]
speed_z=[]
speed_x_lvbo_kal=[]
speed_x_lvbo_low=[]
speed_absolute=[]
speed_gimbal=[]


#接受时间的列表
time_list=[]

#跟随标志位
flag1=0#预测
flag2=0#跟随

#电控发来的角度
gimbal_angle_list=[]

#储存前哨战装甲板两帧位置的列表
outpost_armor_list=[]
outpost_center=[0,0]



# 初始化空的数据列表
yaw_data = []
iou_data = []
# 创建初始的Matplotlib图形
fig, ax = plt.subplots()
ax.set_xlabel('X轴')
ax.set_ylabel('Y轴')
# 设置 y 轴范围
y_min = 1000# 设置最小值
y_max = 2000  # 设置最大值
ax.set_ylim(y_min, y_max)
line, = ax.plot(yaw_data, iou_data, marker='o', linestyle='-', markersize=5)
#设置储存的缓冲区大小
buffer_size = 10000


def update_plot(yaw, iou):
    yaw_data.append(yaw)
    iou_data.append(iou)

    # 如果数据超过缓冲区大小，移除最旧的数据
    if len(yaw_data) > buffer_size:
        yaw_data.pop(0)
        iou_data.pop(0)
    
    # 更新Matplotlib图形数据
    line.set_data(yaw_data, iou_data)
    
    # 重新设置图形的x轴范围，可以根据需要进行调整
    ax.relim()
    ax.autoscale_view()
    
    # 更新图形
    plt.draw()
    plt.pause(0.01)  # 稍微暂停以显示更新

# 示例函数接受两个参数并更新图形
def receive_and_plot(yaw, iou):
    update_plot(yaw, iou)

#计算两圆交点的函数
def calculate_circle_intersections(center1, center2, radius):
    d = np.linalg.norm(np.array(center2) - np.array(center1))

    if d > 2 * radius:
        return None  # 两个圆不相交
    else:
        a = (radius ** 2 - radius ** 2 + d ** 2) / (2 * d)
        h = np.sqrt(radius ** 2 - a ** 2)

        x2 = center1[0] + a * (center2[0] - center1[0]) / d
        y2 = center1[1] + a * (center2[1] - center1[1]) / d

        intersection1 = (x2 + h * (center2[1] - center1[1]) / d, y2 - h * (center2[0] - center1[0]) / d)
        intersection2 = (x2 - h * (center2[1] - center1[1]) / d, y2 + h * (center2[0] - center1[0]) / d)

        return intersection1, intersection2

#计算速度的函数
def calculate_speed(dx,dt):
    relative_speed=dx/1000/dt
    return relative_speed

#调用低通滤波器
def low_filter(filter,x):
    
    x=filter.smooth(x)
    return x

#调用卡尔曼滤波器
def kal_filter(filter,x):
    x=filter.filter(x)
    return x

#目前采取的后一帧代替前一帧的方法，后面可以改为删掉第一帧，最后加一帧
def update(list_,new_value,length):
    for i in range(length-1):
        list_[i]=list_[i+1]
    list_[length-1]=new_value
    

#针对每一个检测框，检测框的数量nms有关
def process_data_img(box: np.ndarray, keypoints:np.ndarray, img: np.ndarray, color: Tuple[int, int, int] = None, label: str = None, line_thickness: int = 5):
    """
    Helper function for drawing single bounding box on image
    Parameters:
        x (np.ndarray): bounding box coordinates in format [x1, y1, x2, y2]
         (no.ndarray): input imagea
        color (Tuple[int, int, int], *optional*, None): color in BGR format for drawing box, if not specified will be selected randomly
        mask (np.ndarray, *optional*, None): instance segmentation mask polygon in format [N, 2], where N - number of points in contour, if not provided, only box will be drawn
        label (str, *optonal*, None): box label string, if not provided will not be provided as drowing result
        line_thickness (int, *optional*, 5): thickness for box drawing lines
    """
    #开火指令
    fire=0
    #小陀螺标志位
    anti_top_flag=0
    #上一帧装甲板的中心点
    global last_center,outpost_center
    
    #储存关键点的数组,默认顺序是左上，右上，右下，左下
    points_2D=np.empty([0,2],dtype=np.float64)
      
    # #相机参数矩阵
    # cameraMatrix = np.array([[2075.23100, 0, 624.212611],
    #                         [ 0, 2073.82280, 514.674148],
    #                          [0, 0, 1]], dtype=np.float64)
    # distCoeffs=np.array([[-0.09584139, 1.10498131, -0.00723334675, -0.00165270614, -8.01363788]],dtype=np.float64)
    
    # #相机参数矩阵(shaobing)
    # cameraMatrix = np.array([[2087.421950, 0, 640.0],
    #                         [ 0, 2087.421950, 512.0],
    #                          [0, 0, 1]], dtype=np.float64)
    # distCoeffs=np.array([[-0.075394, 0.509417, -0.694177, 0.0, 0.0]],dtype=np.float64)
    
    
    #相机参数矩阵(bubing)
    cameraMatrix = np.array([[1550.135539, 0, 640.0],
                            [ 0, 1550.135539, 512.0],
                             [0, 0, 1]], dtype=np.float64)
    distCoeffs=np.array([[-0.088273, 0.55129, 0.005279, 0.0, 0.0]],dtype=np.float64)


    #画框
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    #cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    
    #解包标签信息
    if label:

        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        
    #解包keypoints信息，添加关键点到二维点
    for i, k in enumerate(keypoints):
            if(k.dim()!=0):
                color_k = color or [random.randint(0, 255) for _ in range(3)]
                x_coord, y_coord = k[0], k[1]#得到x,y的坐标
                #                                1280/416=3.07692  1024/416=2.461538
                points_keypoint=np.array([int(x_coord*3.07692),int(y_coord*2.461538)],dtype=np.float64)#作为二维点进行储存
                points_2D=np.vstack((points_2D,points_keypoint))
                


                if x_coord % img.shape[1] != 0 and y_coord % img.shape[0] != 0:
                    if len(k) == 3:
                        conf = k[2]
                        if conf < 0.5:
                            continue

    #在points_2D最后添加点作为中心点
    #points_2D=np.vstack((points_2D,np.array([int((points_2D[0][0]+points_2D[2][0])/2),int((points_2D[0][1]+points_2D[2][1])/2)],dtype=np.float64)))

    if(np.shape(points_2D)==(4,2)):
        
        left_up=points_2D[0]
        right_up=points_2D[1]
        right_down=points_2D[2]
        left_down=points_2D[3]
        
        #print("反投影前二维点",points_2D)
        #画装甲板灯条关键点
        cv2.circle(img, (int(left_up[0]/3.07692), int(left_up[1]/2.461538)), 2, (255,0,0), -1,lineType=cv2.LINE_AA)
        cv2.circle(img, (int(right_up[0]/3.07692), int(right_up[1]/2.461538)), 2, (255,0,0), -1,lineType=cv2.LINE_AA)
        cv2.circle(img, (int(right_down[0]/3.07692), int(right_down[1]/2.461538)), 2, (255,0,0), -1,lineType=cv2.LINE_AA)
        cv2.circle(img, (int(left_down[0]/3.07692), int(left_down[1]/2.461538)), 2, (255,0,0), -1,lineType=cv2.LINE_AA)
        cv2.circle(img,(int((left_up[0]+right_down[0])/2/3.07692),int((left_up[1]+right_down[1])/2/2.461538)),2,(255,255,255),-1,lineType=cv2.LINE_AA)
        cv2.circle(img,(208,208),2,(0,255,0),-1,lineType=cv2.LINE_AA)
        # cv2.line(img,(208,0),(208,416),(0,255,0),2,lineType=cv2.LINE_AA) 

        #画装甲板灯条构成的矩形
        cv2.rectangle(img,(int(left_up[0]/3.07692),int(left_up[1]/2.461538)),(int(right_down[0]/3.07692),int(right_down[1]/2.461538)),(255,0,0),1,lineType=cv2.LINE_AA)

        #筛选大小装甲板
        #print("灯条比值",(points_2D[1][0]-points_2D[0][0])/(points_2D[2][1]-points_2D[1][1]))
        if((points_2D[1][0]-points_2D[0][0])/(points_2D[2][1]-points_2D[1][1])>=5):
            #print("大装甲板")
            #大装甲板尺寸
            points_3D=np.array([[-114.0, -26.0, 0],
                            [114.0,-26.0, 0],
                            [114.0,26.0, 0],
                            [-114.0,26.0, 0]], dtype=np.float64)
            
        elif((points_2D[1][0]-points_2D[0][0])/(points_2D[2][1]-points_2D[1][1])<5):
            #print("小装甲板")
            #小装甲板尺寸,默认顺序是左上，右上，右下，左下,中心
            points_3D = np.array([[-65.0, -26.5, 0],
                            [65.0,-26.5, 0],
                            [65.0,26.5, 0],
                            [-65.0,26.5, 0]], dtype=np.float64)
        
        # print("击打策略",port.enemy_color)
        
        # #此时识别到前哨站但不会发送开火指令和目标信息
        # if((label=="outpost_red" or label=="outpost_blue") and (port.enemy_color==0 or port.enemy_color==1)):
        #     #print("不击打前哨站") 
        #     pass
        
            
        #此时会正常发送所有识别到的目标信息，识别到谁打谁   
        # else:
            #print("击打前哨站和敌方目标")
                      
# ######################################################进行坐标结算###################################################################

        P_matrix,T_matrix=po.my_pnp(points_3D,points_2D,cameraMatrix,distCoeffs)
        #print("测距",T_matrix[2])
        """#########################################################反前哨站算法(反投影版)###########################################
        
        #P_matrix,T_matrix=po.my_pnp(points_3D,points_2D,cameraMatrix,distCoeffs)
        #目前yaw角度给的0，yaw应为自变量，yaw的取值影响反投影结果
        #三个值依次为pitch，yaw，roll
        rotation_vector = np.array([(-15 * math.pi / 180),0.0,0.0], dtype=np.float32)
        translation_vector = T_matrix
            
        # 设置参数
        calibrator.set_parameters(points_3D, cameraMatrix, rotation_vector, translation_vector)
        
        # # 获取投影后的图像点
        #image_points = calibrator.project_points(0)
        # print("反投影后维点",image_points)
        
        best_yaw,best_iou=calibrator.find_maximum_iou_yaw(points_2D)
        image_points = calibrator.project_points(best_yaw)

        #print("iou",best_yaw)
        #对最佳yaw角度进行滤波
        best_yaw=low_filter(filter_x,best_yaw)


        receive_and_plot(time.time(),best_yaw)

        # print("最佳yaw",best_yaw)
        # print("最佳iou",best_iou)

        
        #画出反投影点构成的矩形
        inverse_c1=(int(image_points[0][0]/3.07692), int(image_points[0][1]/2.461538))
        inverse_c2=(int(image_points[2][0]/3.07692), int(image_points[2][1]/2.461538))
        cv2.rectangle(img,inverse_c1,inverse_c2,(0,255,0),1,lineType=cv2.LINE_AA)
        """

        #######################################################反前哨站算法(物理预测版)#########################################
        P_matrix,T_matrix=po.my_pnp(points_3D,points_2D,cameraMatrix,distCoeffs)

        #目前只对x和z进行滤波，用于测试装甲板物理建模效果
        #T_matrix[0]=low_filter(filter_x,float(T_matrix[0]))
        #T_matrix[2]=low_filter(filter_z,float(T_matrix[2]))

        outpost_armor=[int(T_matrix[0]),int(T_matrix[2])]

        angle_yaw=math.atan2(float(T_matrix[0]),float(T_matrix[2]))
        #print(angle_yaw)
        # outpost_armor_list.append(outpost_armor)
        #print(outpost_armor_list)

        # #可视化数据观察稳定性
        # receive_and_plot(time.time(),float(T_matrix[2]))

        #收集前哨站装甲板位置，time数据
        time_new=time.time()
        if(np.shape(time_list)[0]<15 and angle_yaw>-0.05 and angle_yaw<0.05):
            time_list.append(float(time_new))#赋值时间列表
            outpost_armor_list.append(outpost_armor)#赋值前哨站装甲板列表
            

        if (np.shape(time_list)[0]==15 and angle_yaw>-0.05 and angle_yaw<0.05):
            
            #计算当前帧和前10帧相对位置，x和z两个维度都进行计算，如果相对波动小于某个值，就认为是稳定的,130是装甲板尺寸
            dx_ratio=abs((outpost_armor_list[0][0]-outpost_armor_list[14][0])/130)
            dz_ratio=abs((outpost_armor_list[0][1]-outpost_armor_list[14][1])/130)
            #print("x相对差值",dx_ratio)
            #print("z相对差值",dz_ratio)

            #此时认为装甲板处于旋转状态，采集到的运动数据是有效的
            if(dx_ratio>0.1 and dz_ratio>0.1):
                intersections_data = calculate_circle_intersections(outpost_armor_list[0], outpost_armor_list[14], 50)
                if(np.shape(intersections_data)!=()):
                    for i in range (2):
                        #前哨站的中心比装甲板远
                        if(intersections_data[i][1]>outpost_armor_list[0][1] and intersections_data[i][1]>outpost_armor_list[14][1]):
                            outpost_center=intersections_data[i]
                            print("armor1",outpost_armor_list[0])
                            print("armor2",outpost_armor_list[14])
                            print("前哨站中心点",outpost_center)
                            #receive_and_plot(outpost_center[0],outpost_center[1])
            
            #print(np.shape(outpost_armor_list))
            update(time_list,time_new,len(time_list))
            update(outpost_armor_list,outpost_armor,len(outpost_armor_list))
            receive_and_plot(outpost_armor[0],outpost_armor[1])

        #receive_and_plot(outpost_armor[0],outpost_armor[1])
        #receive_and_plot(outpost_center[0],outpost_center[1])
        
        
        
        
        
        # fire=1#更新开火指令
        
        # #计算yaw角度，目前由于相机标定的问题，yaw轴稍微偏左，需要额外补偿一个量(20是正常补偿相机标定误差)
        # #angle_yaw=math.atan2(float(T_matrix[0]),float(T_matrix[2]))
        # angle_yaw=math.atan2(float(T_matrix[0]),float(T_matrix[2]))
        # #print("发送前yaw",angle_yaw)
        
        # #计算装甲板高度,瞄上沿
        # armor_h=406-float(world_position_matrix[1])+52
        # #print("装甲板",armor_h)
        
        # #斜抛运动结算pitch角度
        # #print("高度差",float((480-armor_h)/1000))
        # #print("测距",math.fabs(float(T_matrix[2])/1000))
        # angle_pitch=po.solve_pitch(float((480-armor_h)/1000),math.fabs(float(T_matrix[2])/1000))
        # #print("pitch",angle_pitch)
        # pitch.append(angle_pitch)

        
        # #时间列表填满，进行速度结算,并且进行数据更新
        # if (np.shape(time_list)[0]==10):
            
        #     angle_yaw=math.atan2(float(world_position_matrix[0]),float(world_position_matrix[2]))
        #     t_matrix2_lvbo.append(float(T_matrix[0])+100+50)
    
        #     angle_list=[angle_yaw,angle_pitch,fire]
            
        #     # #发送串口数据
        #     # port.port_clean()
        #     # port.sendAngle(angle_list,fire)
            
        #     #更新,注意最后赋值的时候赋w的是滤波后的还是滤波前
        #     update(time_list,time_new,len(time_list))
        #     update(last_x,float(world_position_matrix[0]),len(last_x))
        #     update(last_z,float(world_position_matrix[2]),len(last_z))
        #     update(anti_top,float((int((left_up[0]+right_down[0])/2/3.07692)-last_center)/((right_up[0]-left_up[0])/3.07692)),len(anti_top))
        #     last_center=int((left_up[0]+right_down[0])/2/3.07692)

        # else:
        #     angle_list=[angle_yaw,angle_pitch]
        #     # #发送串口数据
        #     # port.port_clean()
        #     # port.sendAngle(angle_list,fire)

        
        
        
        
    
        

        """自瞄结算
        t_matrix2.append(float(T_matrix[0]))
        print("相机",T_matrix)
        
        #得到相机坐标系下的坐标
        po.position2cam()
        po.cam2imu()
        inetia_position_matrix=po.imu2inetia(port.gimbal_pitch_angle)
        print("惯性系下的坐标",inetia_position_matrix)
        
        #惯性系向世界系进行转换
        #print("电控yaw角度",port.gimbal_yaw_angle)
        world_position_matrix=po.inetia2world(port.gimbal_yaw_angle)
        #print("世界系下的坐标",world_position_matrix)
        
        
        #对世界坐标系下的x和z进行滤波
        world_position_matrix[0]=low_filter(filter_x,float(world_position_matrix[0]))
        world_position_matrix[2]=low_filter(filter_z,float(world_position_matrix[2]))
        #print(world_position_matrix[0])
        world_position.append(world_position_matrix[0])
        
        
        #收集x,z,time数据
        time_new=time.time()
        if(np.shape(time_list)[0]<10):
            time_list.append(float(time_new))#赋值时间列表
            last_x.append(float(world_position_matrix[0]))#赋值上一次x滤波后的坐标
            last_z.append(float(world_position_matrix[2]))
        
        #收集小陀螺数据
        if(len(anti_top)<30):
            #                    两帧中心点差值                                          变成相对量，不然没法观测
            anti_top.append((int((left_up[0]+right_down[0])/2/3.07692)-last_center)/((right_up[0]-left_up[0])/3.07692))#收集小陀螺的数据
        
        #开始判断小陀螺
        if(len(anti_top)==30):
            for i in range(30):
                if(anti_top[i]<-2 or anti_top[i]>2):
                    anti_top_flag+=1

# #############################################小陀螺状态，不预测#####################################################################        
        if(anti_top_flag>=1):
            print("小陀螺状态")
            
            fire=1#更新开火指令
        
            #计算yaw角度，目前由于相机标定的问题，yaw轴稍微偏左，需要额外补偿一个量(20是正常补偿相机标定误差)
            #angle_yaw=math.atan2(float(T_matrix[0]),float(T_matrix[2]))
            angle_yaw=math.atan2(float(world_position_matrix[0]),float(world_position_matrix[2]))
            #print("发送前yaw",angle_yaw)
            
            #计算装甲板高度,瞄上沿
            armor_h=406-float(world_position_matrix[1])+52
            #print("装甲板",armor_h)
            
            #斜抛运动结算pitch角度
            #print("高度差",float((480-armor_h)/1000))
            #print("测距",math.fabs(float(T_matrix[2])/1000))
            angle_pitch=po.solve_pitch(float((480-armor_h)/1000),math.fabs(float(T_matrix[2])/1000))
            #print("pitch",angle_pitch)
            pitch.append(angle_pitch)

            
            #时间列表填满，进行速度结算,并且进行数据更新
            if (np.shape(time_list)[0]==10):
                
                angle_yaw=math.atan2(float(world_position_matrix[0]),float(world_position_matrix[2]))
                t_matrix2_lvbo.append(float(T_matrix[0])+100+50)
        
                angle_list=[angle_yaw,angle_pitch,fire]
                
                # #发送串口数据
                # port.port_clean()
                # port.sendAngle(angle_list,fire)
                
                #更新,注意最后赋值的时候赋w的是滤波后的还是滤波前
                update(time_list,time_new,len(time_list))
                update(last_x,float(world_position_matrix[0]),len(last_x))
                update(last_z,float(world_position_matrix[2]),len(last_z))
                update(anti_top,float((int((left_up[0]+right_down[0])/2/3.07692)-last_center)/((right_up[0]-left_up[0])/3.07692)),len(anti_top))
                last_center=int((left_up[0]+right_down[0])/2/3.07692)

            else:
                angle_list=[angle_yaw,angle_pitch]
                # #发送串口数据
                # port.port_clean()
                # port.sendAngle(angle_list,fire)

    
        else:
            print("非小陀螺状态")
            #计算yaw角度，目前由于相机标定的问题，yaw轴稍微偏左，需要额外补偿一个量(20是正常补偿相机标定误差)
            #angle_yaw=math.atan2(float(T_matrix[0]),float(T_matrix[2]))
            angle_yaw=math.atan2(float(world_position_matrix[0]),float(world_position_matrix[2]))
            #print("发送前yaw",angle_yaw)
            
            #计算装甲板高度,瞄上沿
            armor_h=406-float(world_position_matrix[1])+52
            #print("装甲板",armor_h)
            
            #斜抛运动结算pitch角度
            #print("高度差",float((480-armor_h)/1000))
            #print("测距",math.fabs(float(T_matrix[2])/1000))
            angle_pitch=po.solve_pitch(float((480-armor_h)/1000),math.fabs(float(T_matrix[2])/1000))
            #print("pitch",angle_pitch)
            pitch.append(angle_pitch)
    
    
            #时间列表填满，进行速度结算,并且进行数据更新
            if (np.shape(time_list)[0]==10):
                #print("last_x",last_x)
                #print("inetia_position_matrix",inetia_position_matrix)
                
                dx=float(world_position_matrix[0])-last_x[0]#当前滤波后的数据减10帧之前的滤波数据,已经结算到了世界坐标系      
                dz=float(world_position_matrix[2])-last_z[0]
                dx_list.append(dx)# import time
                
                #dx=low_filter(filter_x,dx)
                #对dx进行低通滤波，减少识别带来的抖动
                #dx_list_lvbo.append(dx)
                #print("dx",dx)

                #计算相对速度
                relative_speed_x=calculate_speed(dx,(time_new-time_list[0]))
                relative_speed_z=calculate_speed(dz,(time_new-time_list[0]))
                speed_x.append(relative_speed_x)
                #speed_z.append(relative_speed_x)

                #对速度进行低通滤波
                speed_after1=low_filter(filter_vx,relative_speed_x)
                speed_after2=low_filter(filter_vz,relative_speed_z)
                speed_x_lvbo_low.append(speed_after1)
                #print("相对速度x",speed_after1)
                #print("相对速度z",speed_after2)
                
                #计算子弹飞行时间
                # fly_time=float(T_matrix[2])/1000/30/math.cos(angle_pitch)
                fly_time=float(T_matrix[2])/1000/27/math.cos(angle_pitch)   
                #print("飞行时间",fly_time)
                fire=1#更新开火指令
                
                #计算修正量
                x_fix=fly_time*speed_after1*1000
                z_fix=fly_time*speed_after2*1000
                
                #print("修正量x",x_fix)
                #print("修正量z",z_fix)rectanglerld_position_matrix[2]))
                #t_matrix2_lvbo.append(float(T_matrix[0])+100+50+x_fix*1.3)
                
                angle_yaw=math.atan2(float(world_position_matrix[0])+x_fix*1.6,float(world_position_matrix[2])+z_fix*1.6)
                t_matrix2_lvbo.append(float(T_matrix[0])+100+50)
        
                angle_list=[angle_yaw,angle_pitch,fire]
                
                # #发送串口数据
                # port.port_clean()
                # port.sendAngle(angle_list,fire)
                
                #更新列表,注意最后赋值的时候赋w的是滤波后的还是滤波前
                update(time_list,time_new,len(time_list))
                update(last_x,float(world_position_matrix[0]),len(last_x))
                update(last_z,float(world_position_matrix[2]),len(last_z))
                update(anti_top,float((int((left_up[0]+right_down[0])/2/3.07692)-last_center)/((right_up[0]-left_up[0])/3.07692)),len(anti_top))
                last_center=int((left_up[0]+right_down[0])/2/3.07692)

            else:
                angle_list=[angle_yaw,angle_pitch]
                #发送串口数据
                #fire=1
                # port.port_clean()
                # port.sendAngle(angle_list,fire)
            """
    
    return img

#得到目标信息，更新目标状态，判断对面是否是小陀螺状态，进而区分击打策略  
def analyse_results(results: Dict, source_image: np.ndarray, label_map: Dict):
    """
    Helper function for drawing bounding 
    # plt.subplot(313)
    # plt.plot(pitch,label = "pitch")
    # plt.xlabel("time")
    # plt.ylabel("speed")
    # plt.ylim(-1,1)
    # plt.legend() 
    # plt.show() boxes on image
    Parameters:
        image_res (np.ndarray): detection predictions in format [x1, y1, x2, y2, score, label_id]
        source_image (np.ndarray): input image for drawing
        label_map; (Dict[int, str]): label_id to class name mapping
    Returns:

    """
    boxes = results["det"]
    keypoints = results["keypoints"]
    
    h, w = source_image.shape[:2]
    
    for idx, (*xyxy, conf, lbl) in enumerate(boxes): 
        
        #让每帧图像只有一个输出结果
        if idx == 1:
            break 
        if(np.shape(keypoints)!=[]):
            if(np.shape(keypoints[0]>idx)):
                if(np.shape(keypoints[idx])!=torch.Size([])):#防止keypoints和boxes之间数量不匹配造成的报错
                    #label = f'{label_map[int(lbl)]} {conf:.2f}'
                    label = f'{label_map[int(lbl)]}'
                    *keypoints, = keypoints[idx]
                    #进行最终结算
                    source_image = process_data_img(
                        xyxy, keypoints, source_image, label=label, color=colors(int(lbl)), line_thickness=1)
                    

    return source_image


def letterbox(img: np.ndarray, new_shape: Tuple[int, int] = (416, 416), color: Tuple[int, int, int] = (114, 114, 114), auto: bool = False, scale_fill: bool = False, scaleup: bool = False, stride: int = 32):
    """
    Resize image and padding for detection. Takes image as input,
    resizes image to fit into new shape with saving original aspect ratio and pads it to meet stride-multiple constraints

    Parameters:
      img (np.ndarray): image for preprocessing
      new_shape (Tuple(int, int)): image size after preprocessing in format [height, width]
      color (Tuple(int, int, int)): color for filling padded area
      auto (bool): use dynamic input size, only padding for stride constrins applied
      scale_fill (bool): scale image to fill new_shape
      scaleup (bool): allow scale image if it is lower then desired input size, can affect model accuracy
      stride (int): input padding stride
    Returns:
      img (np.ndarray): image after preprocessing
      ratio (Tuple(float, float)): hight and width scaling ratio
      padding_size (Tuple(int, int)): height and width padding size
    """
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / \
            shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def preprocess_image(img0: np.ndarray):
    """
    Preprocess image according to YOLOv8 input requirements.
    Takes image in np.array format, resizes it to specific size using letterbox resize and changes data layout from HWC to CHW.

    Parameters:
      img0 (np.ndarray): image for preprocessing
    Returns:
      img (np.ndarray): image after preprocessing
    """
    # resize
    img = letterbox(img0)[0]

    # Convert HWC to CHW
    if(np.shape(img)==(416,416,3)):
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
    
    return img


def image_to_tensor(image: np.ndarray):
    """
    Preprocess image according to YOLOv8 input requirements.
    Takes image in np.array format, resizes it to specific size using letterbox resize and changes data layout from HWC to CHW.

    Parameters:
      img (np.ndarray): image for preprocessing
    Returns:
      input_tensor (np.ndarray): input tensor in NCHW format with float32 values in [0, 1] range
    """
    input_tensor = image.astype(np.float32)  # uint8 to fp32
    input_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0

    # add batch dimension
    if input_tensor.ndim == 3:
        input_tensor = np.expand_dims(input_tensor, 0)
    return input_tensor


def postprocess(
    preds: np.ndarray,
    input_hw: Tuple[int, int],
    orig_img: np.ndarray,
    min_conf_threshold: float = 0.1,
    nms_iou_threshold: float = 0.7,
    agnosting_nms: bool = False,
    max_detections: int = 300,
):        # parser.add_argument('--model', default="/home/rc-cv/yolov8-face-main/best_openvino_model/outpost.xml",
    """
    YOLOv8 model postprocessing function. Applied non maximum supression algorithm to detections and rescale boxes to original image size
    Parameters:
        preds (np.ndarray): model output prediction boxes and keypoints in format [x1, y1, x2, y2, score, label, keypoints_x, keypoints_y, keypoints_visible]
        input_hw (np.ndarray): preprocessed image
        orig_image (np.ndarray): image before preprocessing
        min_conf_threshold (float, *optional*, 0.25): minimal accepted confidence for object filtering
        nms_iou_threshold (float, *optional*, 0.45): minimal overlap score for removing objects duplicates in NMS
        agnostic_nms (bool, *optiona*, False): apply class agnostinc NMS approach or not
        max_detections (int, *optional*, 300):  maximum detections after NMS
    Returns:
       pred (List[Dict[str, np.ndarray]]): list of dictionary with det - detected boxes in format [x1, y1, x2, y2, score, label] and segment - segmentation polygons for each element in batch
    """
    nms_kwargs = {"agnostic": agnosting_nms, "max_det": max_detections}
    preds = ops.non_max_suppression(
        torch.from_numpy(preds),
        min_conf_threshold,
        nms_iou_threshold,
        nc=len(label_map),
        **nms_kwargs
    )

    results = []
    for i, pred in enumerate(preds):
        shape = orig_img[i].shape if isinstance(
            orig_img, list) else orig_img.shape
        if not len(pred):
            results.append({"det": [], "keypoints": []})
            continue
        else:
            pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
            pred_kpts = pred[:, 6:].view(
                len(pred), *kpt_shape) if len(pred) else pred[:, 6:]
            pred_kpts = ops.scale_coords(input_hw, pred_kpts, shape)
            results.append(
                {"det": pred[:, :6].numpy(), "keypoints": pred_kpts})
    return results


def detect(image: np.ndarray, model: Model):
    """
    OpenVINO YOLOv8 model inference function. Preprocess image, runs model inference and postprocess results using NMS.
    Parameters:
        image (np.ndarray): input image.
        model (Model): OpenVINO compiled model.
    Returns:
        detections (np.ndarray): detected boxes in format [x1, y1, x2, y2, score, label]
    """
    preprocessed_image = preprocess_image(image)
    input_tensor = image_to_tensor(preprocessed_image)
    result = model(input_tensor)
    preds = result[model.output(0)]
    input_hw = input_tensor.shape[2:]
    detections = postprocess(
        preds=preds, input_hw=input_hw, orig_img=image)
    return detections


def main(openvino_model, cap):

    global label_map, kpt_shape
    # Load YOLOv8 model for label map, if you needn't display labels, you can remove this part
    # if(port.enemy_color==1):
    yolo_model = YOLO("E:\\yolo_hero\\red_fina.pt")
    # else:
        # yolo_model = YOLO("/home/rc-cv/yolov8-face-main/red_fina.pt")
        
           
    label_map = yolo_model.model.names
    kpt_shape = yolo_model.model.kpt_shape

    core = Core()

    # Load a model
    # Path to the model's XML file
    model = core.read_model(model=openvino_model) 
    compiled_model = core.compile_model(model=model, device_name="CPU")
    input_layer = compiled_model.input(0)
    
    while(1):
        #time1=time.time()
        input_image=cap.read()
        #data=share_queue.shared_queue_back.get()
        #print("读取的图像数据",data)
        #input_image=cap.read()
        # if(np.shape(input_image)==(416,416,3)):

        input_image=cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        # input_image = cv2.imread(input_path)
        #input_image = np.array(Image.open(input_path))
        detections = detect(input_image, compiled_model)[0]

        boxes = detections["det"]
        keypoints = detections["keypoints"]

        #自瞄处理程序
        if(show_result):
            if (boxes!=[] and keypoints!=[]):
                image_with_boxes = analyse_results(detections, input_image, label_map)
                #可视化处理结果
                # Image.fromarray(image_with_boxes).show()
                image_with_boxes=cv2.cvtColor(image_with_boxes,cv2.COLOR_RGB2BGR)
                cv2.imshow(" ",image_with_boxes)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            else:
                image_with_boxes = analyse_results(detections, input_image, label_map)
                image_with_boxes=cv2.cvtColor(image_with_boxes,cv2.COLOR_RGB2BGR)
                cv2.imshow(" ",image_with_boxes)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                


        #time2=time.time()
                 
        #print("运行帧率：",1/(time2-time1))
        #print(anti_top_flag)



#初始化坐标结算器
po=PositionSolver()

calibrator = CameraCalibration()

#初始化滤波器
filter_x=MovingAverageFilter(5)#测距滑动平均滤波器
filter_z=MovingAverageFilter(5)#测距滑动平均滤波器
filter_vx=MovingAverageFilter(4)#速度滑动平均滤波器
filter_vz=MovingAverageFilter(4)#速度滑动平均滤波器


if __name__ == '__main__':
    # port=SerialPort().start()
    #初始化相机    
    cap=WebcamVideoStream().start()
    #cap=cv2.VideoCapture(1)
    
    parser = argparse.ArgumentParser()
    
    # print("敌方颜色",port.enemy_color)
    # if(port.enemy_color==1 or port.enemy_color==3):
    #     parser.add_argument('--model', default="/home/rc-cv/yolov8-face-main/best_openvino_model/blue_fina.xml",
    #                 help='Input your openvino model.')
        
    # else:
    parser.add_argument('--model', default="E:\\yolo_hero\\best_openvino_model\\red_fina.xml",
                    help='Input your openvino model.')

    args = parser.parse_args()
    


    #主程序
    main(args.model, cap)

    # # 可视化数据 rectangle
    # plt.plot(best_yaw_list, label="Yaw vs IoU")
    # plt.xlabel("IoU")
    # plt.ylabel("Yaw")
    # plt.ylim(-2, 2)
    # plt.legend()
    # plt.show()
    
