# import cv2
# import torch
# import time
# from imutils.video import WebcamVideoStream#双线程读取视频类
# # from Predict.kalmanfilter import KalmanFilter#卡尔曼预测类
# from position_solver import PositionSolver
# import numpy as np
# import math
# from my_serial import SerialPort


# def draw(list_temp, image_temp):
#         for temp in list_temp:
#             name = temp[6]      # 取出标签名
#             temp = temp[:4].astype('int')   # 转成int加快计算``
#             cv2.rectangle(image_temp, (temp[0]+2, temp[1]), (temp[2]-2, temp[3]), (0, 0, 255), 4)  # 框出识别物体
        

#             #测试坐标结算,数据要根据迈德威视采集大小进行比例缩放，实际点不变，像素点放大
#             points_2D= np.array([[(temp[0]+6.5)*2, temp[1]*2], [(temp[2]-6.5)*2, temp[1]*2],  [(temp[0]+6.5)*2, temp[3]*2],[(temp[2]-6.5)*2, temp[3]*2],[(temp[0]+temp[2]),(temp[1]+temp[3])]], 
#             dtype=np.float64)

#             #cv2.circle(image_temp,[temp[2]-4, temp[3]],3,(0,255,0),-1)

#             points_3D = np.array([[-112.5, -63.5, 0],
#                         [112.5,-63.5, 0],
#                         [-112.5,63.5, 0],
#                         [112.5,63.5, 0],[0,0,0]], dtype=np.float64)


#             cameraMatrix = np.array([[2062.395647170194, 0, 726.628746293859],
#                                      [ 0, 2055.621660346987, 425.6643777153183],
#                                      [0, 0, 1]], dtype=np.float64)

#             distCoeffs=np.array([[-0.113416547025808, 0.05173691222453672, -0.007565418959597135, 0.008806078192922203, 0]],dtype=np.float64)
#             #distCoeffs=None

#             #测距得到旋转矩阵和平移矩阵
#             P_matrix,T_matrix=po.my_pnp(points_3D,points_2D,cameraMatrix,distCoeffs)
#             #print(T_matrix)

#             #坐标结算到相机坐标系，只考虑平移矩阵
#             po.position2cam()

#             #向惯性系进行转换
#             inertia_=po.cam2inertia()
#             print(inertia_[0])

#             #结算角度
#             yaw,pitch=po.AngleSolve()
#             angle_list=[yaw,pitch]

#             # #发送串口数据
#             # port.port_clean()
#             # port.sendAngle(angle_list)


#             #cv2.putText(image_temp, "x: " + str(T_matrix[0]+20), (30, 150),cv2.FONT_ITALIC, 1, (255, 255, 255), 2)

#             #显示角度
#             cv2.putText(image_temp, "x: " + str(math.atan2((T_matrix[0]+20),T_matrix[2])), (30, 150),cv2.FONT_ITALIC, 1, (255, 255, 255), 2)

#             #print(T_matrix)

#             #显示类别和平移矩阵信息
#             #cv2.putText(frame, "z: " + str(temp_x), (30, 150),cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
#             # cv2.putText(frame, "y: " + str(temp_y), (30, 200),cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
#             #cv2.putText(image_temp, name, (int(temp[0]-10), int(temp[1]-10)), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)
#             #cv2.putText(image_temp, name, (int(temp[0]-10), int(temp[1]-10)), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)



# # 检测识别, 加载模型
# # 项目名, custom自定义 path权重路径
# model = torch.hub.load('E:/yolov5-7.0', 'custom', path='E:/yolov5-7.0/weights/best_armor.pt', source='local')
# #model=torch.load("E:/yolov5-7.0/weights/yolov5s.pt",map_location=torch.device('cpu'))
# #model = torch.hub.load('E:/yolov5-7.0', 'yolov5s', device='cpu')
# # 置信度阈值
# model.conf = 0.6


# # 加载摄像头
# cap = WebcamVideoStream().start()
# #cap = cv2.VideoCapture("D://MindVision//bubing_test.mp4")

# #加载坐标结算
# po=PositionSolver()

# #加载串口
# # port=SerialPort()
# # port.open_port()

# i=0
# while (1):
#     #_ok,frame = cap.read()
#     frame = cap.read()
#     # 翻转图像
#     # 转换rgb
#     img_cvt = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     # 记录推理消耗时间   FPS
#     start_time = time.time()
#     # 推理image.png
#     results = model(img_cvt)
#     pd = results.pandas().xyxy[0]
#     #print(pd)

#     # 取出对应标签的list
#     car_list = pd[pd['name'] == 'car'].to_numpy()
#     armor_red_list = pd[pd['name'] == '1_armor_red'].to_numpy()

#     #print("carlist",car_list)
#     #判断目标是否出现在视野范围内
#     if(len(car_list)==0 and len(armor_red_list)==0):
#         print("视野范围中没有目标出现")

#     else:
        
#         # 框出物体
#         #draw(car_list, frame)
#         draw(armor_red_list, frame)


#     end_time = time.time()
#     fps_text = 1/(end_time-start_time)
#     print("帧率",fps_text)
#     # if( fps_text<30):
#     cv2.putText(frame, "FPS: " + str(round(fps_text, 2)), (30, 50),cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
#     #print("图像大小：",frame.shape[1])
#     cv2.imshow('test', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

#     i+=1


# # # 做一些清理工作，释放缓存,关闭串口
# cv2.destroyAllWindows()
# # port.close_port()
# cap.stop()
# # cap.stop()





from my_openvino import YOLOV7_OPENVINO


#自瞄主函数
if __name__ == "__main__":

    # #初始化opevino加速
    yolov7_detector=YOLOV7_OPENVINO("C:\\Users\\53429\\Desktop\\test_html\\oil_fence.onnx",'CPU', False, False)
    #调用openvino推理
    #yolov7_detector=YOLOV7_OPENVINO("E:\yolov5-7.0\weights\yolov5s.onnx",'CPU', False, False)
    #yolov7_detector.infer_image(frame)
    yolov7_detector.infer_cam()
    
