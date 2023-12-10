import numpy as np
import math
import cv2

import time
import multiprocessing
from threading import Thread
#进行数据平滑的滤波
class KalmanFilter_low:
   
    def __init__(self):

        # plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
        # plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
        
        # 状态转移矩阵，上一时刻的状态转移到当前时刻
        self.A = np.array([[1,1],[0,1]])
        
        # 过程噪声w协方差矩阵Q，P(w)~N(0,Q)，噪声来自真实世界中的不确定性，Q越小越相信观测值，平滑效果越好，需要的时间越长
        self.Q = np.array([[0.0001,0],[0,0.0001]])
        
        #测量噪声协方差矩阵R，P(v)~N(0,R)，噪声来自测量过程的误差,R越大，更相信模型的值，平滑动效果越好，但需要的时间越长，＜0.1时(过小)，可能导致无法收敛
        self.R = np.array([[0.5,0],[0,0.5]])
        
        # 传输矩阵/状态观测矩阵H
        self.H = np.array([[1,0],[0,1]])
        
        # 控制输入矩阵B
        self.B = None
        
        # 初始位置和速度,x设置成0,v设置成1
        X0 = np.array([[0],[1]])
        
        # 状态估计协方差矩阵P初始化
        P =np.array([[1,0],[0,1]])
        
           
        #---------------------初始化-----------------------------
        #真实值初始化 再写一遍np.array是为了保证它的类型是数组array
        self.X_true = np.array(X0)
        #后验估计值Xk的初始化
        self.X_posterior = np.array(X0)
        #第k次误差的协方差矩阵的初始化
        self.P_posterior = np.array(P)
    
        #创建状态变量的真实值的矩阵 状态变量1：速度 状态变量2：位置
        self.speed_true = []
        self.position_true = []
    
        #创建测量值矩阵
        self.speed_measure = []
        self.position_measure = []
    
        #创建状态变量的先验估计值
        self.speed_prior_est = []
        self.position_prior_est = []
    
        #创建状态变量的后验估计值
        self.speed_posterior_est = []
        self.position_posterior_est = []
        
        self.x1=1
        self.x2=0
    
    
    # def start(self):#启动卡尔曼计算线程
    #     #start the thread to read data
    #     t = Thread(target=self.filter, name="filter", args=()) 
    #     #t=multiprocessing.Process(target=self.filter,name="kalmanfilter_x",args=())
    #     t.daemon=True
    #     t.start()
        
    #     return self
        
    def filter(self,speed):
        
        #print("进入卡尔曼计算线程")

        # 过程噪声w协方差矩阵Q，P(w)~N(0,Q)，噪声来自真实世界中的不确定性，Q越小越相信观测值，平滑效果越好，需要的时间越长
        self.Q = np.array([[0.0005,0],[0,0.0005]])
        
        #测量噪声协方差矩阵R，P(v)~N(0,R)，噪声来自测量过程的误差,R越大，更相信模型的值，平滑动效果越好，但需要的时间越长，＜0.1时(过小)，可能导致无法收敛
        self.R = np.array([[0.1,0],[0,0.1]])

        # 真实值X_true 得到当前时刻的状态;之前我一直在想它怎么完成从Xk-1到Xk的更新，实际上在代码里面直接做迭代就行了，这里是不涉及数组下标的！！！
        #dot函数用于矩阵乘法，对于二维数组，它计算的是矩阵乘积
        self.X_true = np.dot(self.A, self.X_true)

        # 速度的真实值是speed_true 使用append函数可以把每一次循环中产生的拼接在一起，形成一个新的数组speed_true

        self.speed_true.append(self.X_true[1,0])
        self.position_true.append(self.X_true[0,0])
        #print(speed_true)


        # # --------------------生成观测值-----------------------------
        # # 生成过程噪声
        # R_sigma = np.array([[math.sqrt(self.R[0,0]),self.R[0,1]],[self.R[1,0],math.sqrt(self.R[1,1])]])
        # #v = np.array([[gaussian_distribution_generator(R_sigma[0,0])],[gaussian_distribution_generator(R_sigma[1,1])]])

        # 生成观测值Z_measure 取H为单位阵
        X_measure=np.array([[speed],[0]],dtype=np.float64)
        Z_measure = np.dot(self.H, X_measure)
        
        self.speed_measure.append(Z_measure[1,0])
        #print(speed_measure)
        self.position_measure.append(Z_measure[0,0])

        # --------------------进行先验估计-----------------------------
        # 开始时间更新
        # 第1步:基于k-1时刻的后验估计值X_posterior建模预测k时刻的系统状态先验估计值X_prior
        # 此时模型控制输入U=0
        X_prior = np.dot(self.A, self.X_posterior)
        
        # print(self.X_posterior)
        
        # 把k时刻先验预测值赋给两个状态分量的先验预测值 speed_prior_est = X_prior[1,0];position_prior_est=X_prior[0,0]
        # 再利用append函数把每次循环迭代后的分量值拼接成一个完整的数组
        self.speed_prior_est.append(X_prior[1,0])
        self.position_prior_est.append(X_prior[0,0])

        # 第2步:基于k-1时刻的误差ek-1的协方差矩阵P_posterior和过程噪声w的协方差矩阵Q 预测k时刻的误差的协方差矩阵的先验估计值 P_prior
        P_prior_1 = np.dot(self.A, self.P_posterior)
        P_prior = np.dot(P_prior_1, self.A.T) + self.Q

        # --------------------进行状态更新-----------------------------
        # 第3步:计算k时刻的卡尔曼增益K
        k1 = np.dot(P_prior, self.H.T)
        k2 = np.dot(self.H, k1) + self.R
        #k3 = np.dot(np.dot(H, P_prior), H.T) + R  k2和k3是两种写法，都可以
        K = np.dot(k1, np.linalg.inv(k2))

        # 第4步:利用卡尔曼增益K 进行校正更新状态，得到k时刻的后验状态估计值 X_posterior
        X_posterior_1 = Z_measure -np.dot(self.H, X_prior)
        self.X_posterior = X_prior + np.dot(K, X_posterior_1)
        
        # 把k时刻后验预测值赋给两个状态分量的后验预测值 speed_posterior_est = X_posterior[1,0];position_posterior_est = X_posterior[0,0]
        self.speed_posterior_est.append(self.X_posterior[1,0])
        self.position_posterior_est.append(self.X_posterior[0,0])

        # step5:更新k时刻的误差的协方差矩阵 为估计k+1时刻的最优值做准备
        P_posterior_1 = np.eye(2) - np.dot(K, self.H)
        self.P_posterior = np.dot(P_posterior_1, P_prior) 
        self.X_true=self.X_posterior
        
    
        X_posterior_temp= np.dot(self.A, self.X_posterior)
        X_posterior_temp=np.dot(self.A,X_posterior_temp )
        X_posterior_temp=np.dot(self.A,X_posterior_temp )
        X_posterior_temp=np.dot(self.A,X_posterior_temp )
        X_posterior_temp=np.dot(self.A,X_posterior_temp )
        X_posterior_temp=np.dot(self.A,X_posterior_temp )
        
        return X_posterior_temp[0,0]
   
    # def show_result(self):
        
    # # ---------------------再从step5回到step1 其实全程只要搞清先验后验 k的自增是隐藏在循环的过程中的 然后用分量speed和position的append去记录每一次循环的结果-----------------------------
    #     #print(self.X_posterior)
    #     # 画出1行2列的多子图
    #     fig, axs = plt.subplots(1,2)
    #     #速度
    #     axs[0].plot(self.speed_true,"-",color="blue",label="速度真实值",linewidth="1")
    #     axs[0].plot(self.speed_measure,"-",color="grey",label="速度测量值",linewidth="1")
    #     axs[0].plot(self.speed_prior_est,"-",color="green",label="速度先验估计值",linewidth="1")
    #     axs[0].plot(self.speed_posterior_est,"-",color="red",label="速度后验估计值",linewidth="1")
    #     axs[0].set_title("speed")
    #     axs[0].set_xlabel('k')
    #     axs[0].legend(loc = 'upper left')


    #     #位置
    #     axs[1].plot(self.position_true,"-",color="blue",label="位置真实值",linewidth="1")
    #     axs[1].plot(self.position_measure,"-",color="grey",label="位置测量值",linewidth="1")
    #     axs[1].plot(self.position_prior_est,"-",color="green",label="位置先验估计值",linewidth="1")
    #     axs[1].plot(self.position_posterior_est,"-",color="red",label="位置后验估计值",linewidth="1")
    #     axs[1].set_title("position")
    #     axs[1].set_xlabel('k')
    #     axs[1].legend(loc = 'upper left')

    #     #     调整每个子图之间的距离
    #     plt.tight_layout()
    #     plt.figure(figsize=(60, 40))
    #     plt.show()


# #进行数据平滑的滤波
# class KalmanFilter_low_y:
   
#     def __init__(self):

#         plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
#         plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
        
#         #定义状态过程噪声协方差矩阵Q和测量过程噪声协方差矩阵R的值，数据不同，值也不同
#         self.Q_value=0.0001
#         self.R_value=0.5
        
#         # 状态转移矩阵，上一时刻的状态转移到当前时刻
#         self.A = np.array([[1,1],[0,1]])
        
#         # 过程噪声w协方差矩阵Q，P(w)~N(0,Q)，噪声来自真实世界中的不确定性，Q越小越相信观测值，平滑效果越好，需要的时间越长
#         self.Q = np.array([[self.Q_value,0],[0,self.Q_value]])
        
#         #测量噪声协方差矩阵R，P(v)~N(0,R)，噪声来自测量过程的误差,R越大，更相信模型的值，平滑动效果越好，但需要的时间越长，＜0.1时(过小)，可能导致无法收敛
#         self.R = np.array([[self.R_value,0],[0,self.R_value]])
        
#         # 传输矩阵/状态观测矩阵H
#         self.H = np.array([[1,0],[0,1]])
        
#         # 控制输入矩阵B
#         self.B = None
        
#         # 初始位置和速度,x设置成0,v设置成1
#         X0 = np.array([[0],[1]])
        
#         # 状态估计协方差矩阵P初始化
#         P =np.array([[1,0],[0,1]])
        
           
#         #---------------------初始化-----------------------------
#         #真实值初始化 再写一遍np.array是为了保证它的类型是数组array
#         self.X_true = np.array(X0)
#         #后验估计值Xk的初始化
#         self.X_posterior = np.array(X0)
#         #第k次误差的协方差矩阵的初始化
#         self.P_posterior = np.array(P)
    
#         #创建状态变量的真实值的矩阵 状态变量1：速度 状态变量2：位置
#         self.speed_true = []
#         self.position_true = []
    
#         #创建测量值矩阵
#         self.speed_measure = []
#         self.position_measure = []
    
#         #创建状态变量的先验估计值
#         self.speed_prior_est = []
#         self.position_prior_est = []
    
#         #创建状态变量的后验估计值
#         self.speed_posterior_est = []
#         self.position_posterior_est = []
        
#         #滤波函数需要的中间变量,x1是需要滤波的数据，x2是滤波的速度值
#         self.x1=0
#         self.x2=0
    
    
#     def start(self):#启动卡尔曼计算线程
#         #start the thread to read data
#         #t = Thread(target=self.filter, name="filter", args=()) 
#         t=multiprocessing.Process(target=self.filter,name="kalmanfilter_y",args=())
#         t.daemon=True
#         t.start()
        
#         return self
        
#     def filter(self):
#         #print("进入卡尔曼计算线程")

#         #main loop
#         while(True):

#             #从共享队列中读取数据
#             data=share_queue.shared_queue_y.get()
#             print("接收数据",data)
            
#             # 过程噪声w协方差矩阵Q，P(w)~N(0,Q)，噪声来自真实世界中的不确定性，Q越小越相信观测值，平滑效果越好，需要的时间越长
#             self.Q = np.array([[data[0],0],[0,data[0]]])
            
#             #测量噪声协方差矩阵R，P(v)~N(0,R)，噪声来自测量过程的误差,R越大，更相信模型的值，平滑动效果越好，但需要的时间越长，＜0.1时(过小)，可能导致无法收敛
#             self.R = np.array([[data[1],0],[0,data[1]]])

#             # 真实值X_true 得到当前时刻的状态;之前我一直在想它怎么完成从Xk-1到Xk的更新，实际上在代码里面直接做迭代就行了，这里是不涉及数组下标的！！！
#             #dot函数用于矩阵乘法，对于二维数组，它计算的是矩阵乘积
#             self.X_true = np.dot(self.A, self.X_true)

#             # 速度的真实值是speed_true 使用append函数可以把每一次循环中产生的拼接在一起，形成一个新的数组speed_true

#             self.speed_true.append(self.X_true[1,0])
#             self.position_true.append(self.X_true[0,0])
#             #print(speed_true)


#             # # --------------------生成观测值-----------------------------
#             # # 生成过程噪声
#             # R_sigma = np.array([[math.sqrt(self.R[0,0]),self.R[0,1]],[self.R[1,0],math.sqrt(self.R[1,1])]])
#             # #v = np.array([[gaussian_distribution_generator(R_sigma[0,0])],[gaussian_distribution_generator(R_sigma[1,1])]])

#             # 生成观测值Z_measure 取H为单位阵
#             X_measure=np.array([[data[2]],[data[3]]],dtype=np.float64)
#             Z_measure = np.dot(self.H, X_measure)
            
#             self.speed_measure.append(Z_measure[1,0])
#             #print(speed_measure)
#             self.position_measure.append(Z_measure[0,0])

#             # --------------------进行先验估计-----------------------------
#             # 开始时间更新
#             # 第1步:基于k-1时刻的后验估计值X_posterior建模预测k时刻的系统状态先验估计值X_prior
#             # 此时模型控制输入U=0
#             X_prior = np.dot(self.A, self.X_posterior)
            
#             # print(self.X_posterior)
            
#             # 把k时刻先验预测值赋给两个状态分量的先验预测值 speed_prior_est = X_prior[1,0];position_prior_est=X_prior[0,0]
#             # 再利用append函数把每次循环迭代后的分量值拼接成一个完整的数组
#             self.speed_prior_est.append(X_prior[1,0])
#             self.position_prior_est.append(X_prior[0,0])

#             # 第2步:基于k-1时刻的误差ek-1的协方差矩阵P_posterior和过程噪声w的协方差矩阵Q 预测k时刻的误差的协方差矩阵的先验估计值 P_prior
#             P_prior_1 = np.dot(self.A, self.P_posterior)
#             P_prior = np.dot(P_prior_1, self.A.T) + self.Q

#             # --------------------进行状态更新-----------------------------
#             # 第3步:计算k时刻的卡尔曼增益K
#             k1 = np.dot(P_prior, self.H.T)
#             k2 = np.dot(self.H, k1) + self.R
#             #k3 = np.dot(np.dot(H, P_prior), H.T) + R  k2和k3是两种写法，都可以
#             K = np.dot(k1, np.linalg.inv(k2))

#             # 第4步:利用卡尔曼增益K 进行校正更新状态，得到k时刻的后验状态估计值 X_posterior
#             X_posterior_1 = Z_measure -np.dot(self.H, X_prior)
#             self.X_posterior = X_prior + np.dot(K, X_posterior_1)
            
#             # 把k时刻后验预测值赋给两个状态分量的后验预测值 speed_posterior_est = X_posterior[1,0];position_posterior_est = X_posterior[0,0]
#             self.speed_posterior_est.append(self.X_posterior[1,0])
#             self.position_posterior_est.append(self.X_posterior[0,0])

#             # step5:更新k时刻的误差的协方差矩阵 为估计k+1时刻的最优值做准备
#             P_posterior_1 = np.eye(2) - np.dot(K, self.H)
#             self.P_posterior = np.dot(P_posterior_1, P_prior) 
#             self.X_true=self.X_posterior
            
#             #将处理后的数据写入队列
#             share_queue.shared_queue_back.put(self.X_posterior[0,0])
        
        
#     def show_result(self):
        
#     # ---------------------再从step5回到step1 其实全程只要搞清先验后验 k的自增是隐藏在循环的过程中的 然后用分量speed和position的append去记录每一次循环的结果-----------------------------
#         #print(self.X_posterior)
#         # 画出1行2列的多子图
#         fig, axs = plt.subplots(1,2)
#         #速度
#         axs[0].plot(self.speed_true,"-",color="blue",label="速度真实值",linewidth="1")
#         axs[0].plot(self.speed_measure,"-",color="grey",label="速度测量值",linewidth="1")
#         axs[0].plot(self.speed_prior_est,"-",color="green",label="速度先验估计值",linewidth="1")
#         axs[0].plot(self.speed_posterior_est,"-",color="red",label="速度后验估计值",linewidth="1")
#         axs[0].set_title("speed")
#         axs[0].set_xlabel('k')
#         axs[0].legend(loc = 'upper left')


#         #位置
#         axs[1].plot(self.position_true,"-",color="blue",label="位置真实值",linewidth="1")
#         axs[1].plot(self.position_measure,"-",color="grey",label="位置测量值",linewidth="1")
#         axs[1].plot(self.position_prior_est,"-",color="green",label="位置先验估计值",linewidth="1")
#         axs[1].plot(self.position_posterior_est,"-",color="red",label="位置后验估计值",linewidth="1")
#         axs[1].set_title("position")
#         axs[1].set_xlabel('k')
#         axs[1].legend(loc = 'upper left')

#         #     调整每个子图之间的距离
#         plt.tight_layout()
#         plt.figure(figsize=(60, 40))
#         plt.show()


# #进行数据平滑的滤波
# class KalmanFilter_low_z:
   
#     def __init__(self):

#         plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
#         plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
        
#         #定义状态过程噪声协方差矩阵Q和测量过程噪声协方差矩阵R的值，数据不同，值也不同
#         self.Q_value=0.0001
#         self.R_value=0.5
        
#         # 状态转移矩阵，上一时刻的状态转移到当前时刻
#         self.A = np.array([[1,1],[0,1]])
        
#         # 过程噪声w协方差矩阵Q，P(w)~N(0,Q)，噪声来自真实世界中的不确定性，Q越小越相信观测值，平滑效果越好，需要的时间越长
#         self.Q = np.array([[self.Q_value,0],[0,self.Q_value]])
        
#         #测量噪声协方差矩阵R，P(v)~N(0,R)，噪声来自测量过程的误差,R越大，更相信模型的值，平滑动效果越好，但需要的时间越长，＜0.1时(过小)，可能导致无法收敛
#         self.R = np.array([[self.R_value,0],[0,self.R_value]])
        
#         # 传输矩阵/状态观测矩阵H
#         self.H = np.array([[1,0],[0,1]])
        
#         # 控制输入矩阵B
#         self.B = None
        
#         # 初始位置和速度,x设置成0,v设置成1
#         X0 = np.array([[0],[1]])
        
#         # 状态估计协方差矩阵P初始化
#         P =np.array([[1,0],[0,1]])
        
           
#         #---------------------初始化-----------------------------
#         #真实值初始化 再写一遍np.array是为了保证它的类型是数组array
#         self.X_true = np.array(X0)
#         #后验估计值Xk的初始化
#         self.X_posterior = np.array(X0)
#         #第k次误差的协方差矩阵的初始化
#         self.P_posterior = np.array(P)
    
#         #创建状态变量的真实值的矩阵 状态变量1：速度 状态变量2：位置
#         self.speed_true = []
#         self.position_true = []
    
#         #创建测量值矩阵
#         self.speed_measure = []
#         self.position_measure = []
    
#         #创建状态变量的先验估计值
#         self.speed_prior_est = []
#         self.position_prior_est = []
    
#         #创建状态变量的后验估计值
#         self.speed_posterior_est = []
#         self.position_posterior_est = []
        
#         #滤波函数需要的中间变量,x1是需要滤波的数据，x2是滤波的速度值
#         self.x1=0
#         self.x2=0
    
    
#     def start(self):#启动卡尔曼计算线程
#         #start the thread to read data
#         #t = Thread(target=self.filter, name="filter", args=()) 
#         t=multiprocessing.Process(target=self.filter,name="kalmanfilter_z",args=())
#         t.daemon=True
#         t.start()
        
#         return self
        
#     def filter(self):
#         #print("进入卡尔曼计算线程")

#         #main loop
#         while(True):

#             #从共享队列中读取数据
#             data=share_queue.shared_queue_z.get()
#             print("接收数据",data)
            
#             # 过程噪声w协方差矩阵Q，P(w)~N(0,Q)，噪声来自真实世界中的不确定性，Q越小越相信观测值，平滑效果越好，需要的时间越长
#             self.Q = np.array([[data[0],0],[0,data[0]]])
            
#             #测量噪声协方差矩阵R，P(v)~N(0,R)，噪声来自测量过程的误差,R越大，更相信模型的值，平滑动效果越好，但需要的时间越长，＜0.1时(过小)，可能导致无法收敛
#             self.R = np.array([[data[1],0],[0,data[1]]])

#             # 真实值X_true 得到当前时刻的状态;之前我一直在想它怎么完成从Xk-1到Xk的更新，实际上在代码里面直接做迭代就行了，这里是不涉及数组下标的！！！
#             #dot函数用于矩阵乘法，对于二维数组，它计算的是矩阵乘积
#             self.X_true = np.dot(self.A, self.X_true)

#             # 速度的真实值是speed_true 使用append函数可以把每一次循环中产生的拼接在一起，形成一个新的数组speed_true

#             self.speed_true.append(self.X_true[1,0])
#             self.position_true.append(self.X_true[0,0])
#             #print(speed_true)


#             # # --------------------生成观测值-----------------------------
#             # # 生成过程噪声
#             # R_sigma = np.array([[math.sqrt(self.R[0,0]),self.R[0,1]],[self.R[1,0],math.sqrt(self.R[1,1])]])
#             # #v = np.array([[gaussian_distribution_generator(R_sigma[0,0])],[gaussian_distribution_generator(R_sigma[1,1])]])

#             # 生成观测值Z_measure 取H为单位阵
#             X_measure=np.array([[data[2]],[data[3]]],dtype=np.float64)
#             Z_measure = np.dot(self.H, X_measure)
            
#             self.speed_measure.append(Z_measure[1,0])
#             #print(speed_measure)
#             self.position_measure.append(Z_measure[0,0])

#             # --------------------进行先验估计-----------------------------
#             # 开始时间更新
#             # 第1步:基于k-1时刻的后验估计值X_posterior建模预测k时刻的系统状态先验估计值X_prior
#             # 此时模型控制输入U=0
#             X_prior = np.dot(self.A, self.X_posterior)
            
#             # print(self.X_posterior)
            
#             # 把k时刻先验预测值赋给两个状态分量的先验预测值 speed_prior_est = X_prior[1,0];position_prior_est=X_prior[0,0]
#             # 再利用append函数把每次循环迭代后的分量值拼接成一个完整的数组
#             self.speed_prior_est.append(X_prior[1,0])
#             self.position_prior_est.append(X_prior[0,0])

#             # 第2步:基于k-1时刻的误差ek-1的协方差矩阵P_posterior和过程噪声w的协方差矩阵Q 预测k时刻的误差的协方差矩阵的先验估计值 P_prior
#             P_prior_1 = np.dot(self.A, self.P_posterior)
#             P_prior = np.dot(P_prior_1, self.A.T) + self.Q

#             # --------------------进行状态更新-----------------------------
#             # 第3步:计算k时刻的卡尔曼增益K
#             k1 = np.dot(P_prior, self.H.T)
#             k2 = np.dot(self.H, k1) + self.R
#             #k3 = np.dot(np.dot(H, P_prior), H.T) + R  k2和k3是两种写法，都可以
#             K = np.dot(k1, np.linalg.inv(k2))

#             # 第4步:利用卡尔曼增益K 进行校正更新状态，得到k时刻的后验状态估计值 X_posterior
#             X_posterior_1 = Z_measure -np.dot(self.H, X_prior)
#             self.X_posterior = X_prior + np.dot(K, X_posterior_1)
            
#             # 把k时刻后验预测值赋给两个状态分量的后验预测值 speed_posterior_est = X_posterior[1,0];position_posterior_est = X_posterior[0,0]
#             self.speed_posterior_est.append(self.X_posterior[1,0])
#             self.position_posterior_est.append(self.X_posterior[0,0])

#             # step5:更新k时刻的误差的协方差矩阵 为估计k+1时刻的最优值做准备
#             P_posterior_1 = np.eye(2) - np.dot(K, self.H)
#             self.P_posterior = np.dot(P_posterior_1, P_prior) 
#             self.X_true=self.X_posterior
            
#             #将处理后的数据写入队列
#             share_queue.shared_queue_back.put(self.X_posterior[0,0])
        
        
#     def show_result(self):
        
#     # ---------------------再从step5回到step1 其实全程只要搞清先验后验 k的自增是隐藏在循环的过程中的 然后用分量speed和position的append去记录每一次循环的结果-----------------------------
#         #print(self.X_posterior)
#         # 画出1行2列的多子图
#         fig, axs = plt.subplots(1,2)
#         #速度
#         axs[0].plot(self.speed_true,"-",color="blue",label="速度真实值",linewidth="1")
#         axs[0].plot(self.speed_measure,"-",color="grey",label="速度测量值",linewidth="1")
#         axs[0].plot(self.speed_prior_est,"-",color="green",label="速度先验估计值",linewidth="1")
#         axs[0].plot(self.speed_posterior_est,"-",color="red",label="速度后验估计值",linewidth="1")
#         axs[0].set_title("speed")
#         axs[0].set_xlabel('k')
#         axs[0].legend(loc = 'upper left')


#         #位置
#         axs[1].plot(self.position_true,"-",color="blue",label="位置真实值",linewidth="1")
#         axs[1].plot(self.position_measure,"-",color="grey",label="位置测量值",linewidth="1")
#         axs[1].plot(self.position_prior_est,"-",color="green",label="位置先验估计值",linewidth="1")
#         axs[1].plot(self.position_posterior_est,"-",color="red",label="位置后验估计值",linewidth="1")
#         axs[1].set_title("position")
#         axs[1].set_xlabel('k')
#         axs[1].legend(loc = 'upper left')

#         #     调整每个子图之间的距离
#         plt.tight_layout()
#         plt.figure(figsize=(60, 40))
#         plt.show()


class MovingAverageFilter:
    def __init__(self,window_size):
        self.window_size=window_size
        self.data=[]
        
    def smooth(self,value):
        self.data.append(value)
        if len(self.data)>self.window_size:
            self.data=self.data[1:]
        
        # print("滤波数组",self.data)
        return sum(self.data)/len(self.data)
    def reset(self):
        self.data=[]
    
    
# class KalmanFilter:
#     #实例属性
#     kf = cv2.KalmanFilter(2, 1)                                                                              #其值为4，因为状态转移矩阵transitionMatrix有4个维度
#                                                                                                              #需要观测的维度为2
#     kf.measurementMatrix = np.array([[1, 0]], np.float32)                                #创建测量矩阵
#     kf.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)     #创建状态转移矩阵

#     def predict(self, coordX):                      #实例方法，自己实现一个predict
#         ''' This function estimates the position of the object'''
#         measured = np.array([[np.float32(coordX)]])   
#         self.kf.correct(measured)                           #结合观测值更新状态值，correct为卡尔曼滤波器自带函数
#         predicted = self.kf.predict()                       #调用卡尔曼滤波器自带的预测函数
#         x= int(predicted[0])                            #得到预测后的坐标值
#         return x


#https://pysource.com/2021/10/29/kalman-filter-predict-the-trajectory-of-an-object/
# import cv2
# import numpy as np

# class KalmanFilter:
#     #实例属性
#     kf = cv2.KalmanFilter(4, 2)                                                                              #其值为4，因为状态转移矩阵transitionMatrix有4个维度
#                                                                                                              #需要观测的维度为2
#     kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)                                #创建测量矩阵
#     kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)     #创建状态转移矩阵

#     def predict(self, coordX, coordY):                      #实例方法，自己实现一个predict
#         ''' This function estimates the position of the object'''
#         measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])   
#         self.kf.correct(measured)                           #结合观测值更新状态值，correct为卡尔曼滤波器自带函数
#         predicted = self.kf.predict()                       #调用卡尔曼滤波器自带的预测函数
#         x, y = int(predicted[0]), int(predicted[1])         #得到预测后的坐标值
#         return x, y
import cv2
import numpy as np

class KalmanFilter:
    #实例属性
    kf = cv2.KalmanFilter(4, 2)                                                                              #其值为4，因为状态转移矩阵transitionMatrix有4个维度
                                                                                                             #需要观测的维度为2
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)                                #创建测量矩阵
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)     #创建状态转移矩阵
    
    predicted=np.array([[0],[0]], np.float32)
    def predict(self, coordX, coordY):                      #实例方法，自己实现一个predict，
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])   
        self.kf.correct(measured)                           #结合观测值更新状态值，correct为卡尔曼滤波器自带函数
        predicted = self.kf.predict() 
        #print(type(predicted))
        
        # for i in range (3):
        #     #print("predic","次数",predicted,i)
        #     measured = np.array([[predicted[0]], [predicted[1]]])   
        #     self.kf.correct(measured)                           #结合观测值更新状态值，correct为卡尔曼滤波器自带函数
        #     predicted = self.kf.predict() 
        x, y = (predicted[0]), (predicted[1])         #得到预测后的坐标值
        return x, y


# data_list=[2493.237385946949, 2475.851288354743, 2472.113980830808, 2463.5573537326723, 2464.5062296282113, 2461.2661952928333, 2464.739194814679, 2464.85163390415, 2476.9671607008595, 2483.2464198831676, 2473.805830124269, 2478.5793982742975, 2491.6279399330383, 2487.5240220898586, 2477.014936579656, 2485.8499402191756, 2479.570681036867, 2461.764677019782, 2454.109068746651, 2452.8288510778957, 2444.2395010074215, 2438.5126495334007, 2436.442306309845, 2442.536921406666, 2446.0796569137487, 2449.0978076762067, 2455.9661692427867, 2475.5708545857187, 2477.131847762029, 2480.463626287847, 2491.1725746856127, 2496.063739235343, 2491.3257475509936, 2494.2684354718135, 2499.1832246178706, 2493.036005035362, 2493.0557179519706, 2487.5656210704133, 2475.246336372569, 2462.8842250358402, 2462.8842250358402, 2462.8645121192308, 2456.8961577973223, 2464.7117613980363, 2472.7233703515667, 2473.955800788018, 2473.955800788018, 2473.7655719829236, 2471.4396297433887, 2475.790132126587, 2480.7049212726442, 2487.1007538930794, 2493.0534011696013, 2489.2847283123156, 2487.1966537350745, 2472.9578196875764, 2466.930817020885, 2459.887689302964, 2461.184851561529, 2458.499357988741, 2460.7320813799097, 2446.882736805962, 2453.567513611505, 2459.138712919521, 2459.35584283587, 2460.8528679819983, 2480.7292152226373, 2481.489050133676, 2476.117713713224, 2474.3948927645943, 2491.0532263552127, 2472.7698745796038, 2464.7907251194297, 2470.9493082464332, 2470.439105519557, 2459.7334192054614, 2458.2462673912746, 2470.614142274728, 2473.0140502327217, 2473.2756606870184, 2471.4455196398694, 2481.8723707597705, 2471.319742866403, 2462.761251781405, 2457.154999541101, 2459.1910764387308, 2464.270186122224, 2472.6384484021264, 2472.8286772072215, 2491.11202898283, 2487.4938014390837, 2479.0812446901814, 2470.7129824102785, 2473.06310918844, 2461.9990719619673, 2459.458716378711, 2461.430129599919, 2460.3038250354502, 2454.651299453016, 2453.427204960412, 2453.617433765507, 2459.313466497545, 2459.3615164726098, 2468.0679828359325, 2465.9694286755494, 2482.4375334610727, 2470.9790822576065, 2477.2384444623676, 2469.9198301866777, 2471.3678060765374, 2454.709472485919, 2453.239026012069, 2453.60320695179, 2461.0630422235085, 2463.2960658990132, 2467.2772999543276, 2468.5575176230827, 2471.5219471674054, 2471.9450129342354, 2477.1396809390376, 2477.280953113097, 2493.939286703715, 2495.7484253094212, 2498.148333267415, 2490.0195424239164, 2491.8496834710654, 2481.14399715697, 2470.634911646768, 2466.446622433312, 2461.9310728493574, 2460.2082519007286, 2456.795960207463, 2455.1709420224724, 2450.8007401509312, 2458.250412573582, 2454.0205862456883, 2467.5788525743237, 2478.2751837337737, 2488.5376847039565, 2484.856685138592, 2483.2316669536012, 2471.255551271082, 2466.1654723519373, 2460.264837309447, 2462.223015926183, 2466.3883896944303, 2465.763236307465, 2468.7811679151228, 2460.955634768363, 2462.678455716992, 2464.4999660614285, 2465.1251194483943, 2458.1259537854226, 2471.8521219746717, 2476.408560208351, 2477.99934155718, 2478.908899543555, 2477.419853188506, 2465.7498634990284, 2461.5949092640103, 2455.8324907925826, 2461.5432832752967, 2467.1548358597197, 2474.720907706017, 2468.6966394300625, 2472.487644680282, 2460.1197697968278, 2460.2270898953484, 2458.430372696038, 2462.223015926183, 2462.8393718215616, 2474.973475516754, 2476.696296465383, 2476.315664593004, 2470.7041120085805, 2470.476723929429, 2476.3543101121377, 2476.7557941107984, 2478.933143182488, 2484.652015865431, 2484.643679921583, 2461.179016900678, 2463.3833767104925, 2461.8223835341814, 2452.108391429486, 2445.7920323645394, 2455.1325415308615, 2452.5266977223864, 2453.523404135839, 2465.166153039644, 2463.9026325268765, 2461.830343493644, 2461.830343493644, 2462.188694406021, 2462.1886944060216, 2471.565291183046, 2485.6282531146044, 2483.656839893396, 2481.5659694796755, 2477.67903406383, 2471.520450936826, 2459.8076157834303, 2458.0888995664436, 2451.0091090536203, 2452.718695397777, 2457.0888972693174, 2452.695515859534, 2456.385645297729, 2465.762242074754, 2465.903514248813, 2465.655818606645, 2473.461491709694]

# data=[150,150,150,150,150,151,155,155,155,155,160,165,165,169,168]

# for value in data:
#     smoothed_value=filter.smooth(value)
#     print("origin value:",value,"smoothed",smoothed_value)


# def plot_points(data):
#     x_valuees=range(len(data))
#     plt.plot(x_valuees,data)
#     plt.show()
    
    
# x_data_before=[2250.2133096645694, 2240.8372787850576, 2230.1741263394238, 2232.181827691945, 2215.672707803298, 2219.1966433417106, 2236.3077570691694, 2242.6852977374265, 2236.9655560743463, 2212.9768139236417, 2215.672707803298, 2234.8145001662306, 2241.724652019955, 2241.724652019955, 2209.3508866142733, 2225.7790475996044, 2242.6852977374265, 2236.3077570691694, 2250.2133096645694, 2215.672707803298, 2228.235117883181, 2243.449373472964, 2238.9142635457642, 2242.6852977374265, 2272.5815671116184, 2217.467838616096, 2233.5211161764732, 2231.52555569309, 2241.724652019955, 2230.1741263394238, 2240.8372787850576, 2251.9641606612777, 2191.7261322058325, 2251.9641606612777, 2240.8372787850576, 2240.8372787850576, 2251.9641606612777, 2238.9142635457642, 2240.8372787850576, 2243.5210561416798, 2240.8372787850576, 2242.6852977374265, 2217.467838616096, 2238.9664264430885, 2241.724652019955, 2217.467838616096, 2244.0805595859297, 2241.724652019955, 2224.9396015172742, 2228.08940647515, 2233.5211161764732, 2230.1213203738225, 2230.1741263394238, 2221.1905841950324, 2232.181827691945, 2224.072271780954, 2240.8372787850576, 2240.8372787850576, 2239.8027899862295, 2251.9641606612777, 2241.724652019955, 2203.847136541692, 2240.8372787850576, 2248.4370696298292, 2240.8372787850576, 2241.724652019955, 2228.08940647515, 2230.1741263394238, 2222.0584133627062, 2253.6427670933213, 2251.9641606612777, 2230.1741263394238, 2242.6852977374265, 2215.7862904717326, 2232.1296381927045, 2193.437195272265, 2240.8372787850576, 2196.4902121092946, 2240.8372787850576, 2251.9641606612777, 2244.0805595859297, 2227.4191387350343, 2217.467838616096, 2240.8372787850576, 2215.672707803298, 2241.724652019955, 2226.644588247696, 2230.1741263394238, 2240.8372787850576, 2236.3077570691694, 2242.6852977374265, 2242.6852977374265, 2219.491517421203, 2240.8372787850576, 2240.8372787850576, 2239.8027899862295, 2241.724652019955, 2215.672707803298, 2203.847136541692, 2232.181827691945, 2240.8372787850576, 2230.1741263394238, 2241.724652019955, 2240.8372787850576, 2230.1741263394238]
# x_data_after=[2250.2133096645694, 2245.5252942248135, 2240.4082382630168, 2238.351635620249, 2233.815850056859, 2227.6125167922874, 2226.7066124491093, 2229.2088467287094, 2230.16559240519, 2229.626413629259, 2228.9216265215764, 2228.6229751409887, 2228.430845997494, 2229.382665186616, 2228.657479724742, 2230.6787476840036, 2232.2529071982426, 2231.1695282080855, 2232.8672597370087, 2234.1316239748135, 2234.6228380315288, 2234.7756531786363, 2235.2969544739553, 2233.7913520885268, 2245.173123950191, 2243.019668096774, 2241.0340166374754, 2239.5562750669405, 2239.3641459234464, 2230.8826577690074, 2235.5565458028004, 2239.245154699761, 2231.2852700023095, 2233.3331717305737, 2235.465802219701, 2235.465802219701, 2235.465802219701, 2244.903428487687, 2242.678052112443, 2243.214807583767, 2243.214807583767, 2241.3590349989972, 2237.0697500130636, 2236.6955795446697, 2236.3362987203245, 2231.6624106865324, 2231.941463056233, 2236.792825737005, 2233.987460751842, 2231.2604116428806, 2234.4710671549565, 2231.679219312535, 2229.369114176429, 2228.6193107119802, 2229.437794955339, 2227.5480260762356, 2229.6912177584827, 2231.8238482476095, 2235.5462894058487, 2239.502755999715, 2243.033232047516, 2235.635203598842, 2235.635203598842, 2237.362059527562, 2235.1366831523183, 2235.1366831523183, 2239.98513713901, 2237.8525066498833, 2232.5767753964587, 2235.137873058111, 2237.185774786376, 2237.6027187592304, 2240.1049530388314, 2238.8505284606363, 2234.547902680513, 2222.8425096027104, 2224.975140091837, 2215.736122966211, 2220.7463206288753, 2224.7132251225908, 2234.8418979853236, 2232.158269975319, 2236.353795276679, 2236.353795276679, 2229.095504705083, 2228.624323191888, 2228.469413094421, 2231.010670639086, 2231.010670639086, 2235.13768049226, 2235.329809635755, 2238.537951533701, 2236.4014297500566, 2236.4014297500566, 2237.307334093234, 2236.730832542995, 2236.5387033995003, 2235.7749414759196, 2228.3769130272462, 2226.645822808624, 2226.8527205683895, 2224.542615432283, 2229.753004275615, 2237.1510327242877, 2236.749492453783]
# #plot_points(x_data_before)
# plot_points(x_data_after)