#openvino加速代码，整合了相机包，目前两种检测模式，图像模式和相机连续采集模式
import cv2
import numpy as np
import random
import time
from openvino.runtime import Layout, AsyncInferQueue, PartialShape
import math
#Import necessary libraries
import cv2
from openvino.runtime import Core, Model





class YOLOV7_OPENVINO(object):
    def __init__(self, model_path, device, pre_api, nireq):

        #self.classes=["1_blue","1_red","2_blue","2_red","3_blue","3_red","4_blue","4_red","guard_blue"]

    #     self.classes = [
    #     "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    #     "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    #     "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    #     "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    #     "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    #     "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    #     "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    #     "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    #     "hair drier", "toothbrush"
    #    ]
    
        self.classes =["oil_leak"]


#########附加处理（串口，预测，坐标结算）
#########模型参数
        self.TARGET=False
        self.NUM_CLASSES = 8;  # Number of classes
        self.NUM_COLORS = 4;   # Number of color
        self.batchsize = 1   #n张图片打包成一个进行处理，一般设定为逐帧处理，即batchsize=1
        self.grid = True             #调整该参数可以在v7和v5模型之间切换，适合不同的输出形式（3feature map分别输出或是三合一输出）
        self.img_size = (640,640)   #输入图像的大小，v5模型为(640,640)
        self.conf_thres = 0.6     #置信度
        self.iou_thres = 0.1         #iou阈值，iou过大可能检测不到框，过小可能会检测到多个框，作为nms函数的参数
        self.class_num = 20          #类别数目
        #self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.classes]#每次画框的颜色
        self.stride = [8, 16, 32]    #图像每次卷积移动的步长，这一步决定了三个feature map的大小，每个feature map上都会画出框，最后合起来输出一个框
        #self.anchor_list = [[12, 16, 19, 36, 40, 28], [36, 75, 76, 55, 72, 146], [142, 110, 192, 243, 459, 401]]#每个feature map上的先验框，一共三个
        #self.anchor = np.array(self.anchor_list).astype(float).reshape(3, -1, 2)
        # area = self.img_size[0] * self.img_size[1]#图像的面积
####暂时还没理解
        # self.size = [int(area / self.stride[0] ** 2), int(area / self.stride[1] ** 2), int(area / self.stride[2] ** 2)]

####暂时还没理解
        #self.feature = [[int(j / self.stride[i]) for j in self.img_size] for i in range(3)]

        ie = Core()#初始化
        self.model = ie.read_model(model_path)#加载模型 
        self.input_layer = self.model.input(0)#设定输入层，跟读取的模型相关
        new_shape = PartialShape([self.batchsize, 3, self.img_size[0], self.img_size[1]])
        self.model.reshape({self.input_layer.any_name: new_shape})
        self.compiled_model = ie.compile_model(model=self.model, device_name=device)#编译模型
        self.infer_queue = AsyncInferQueue(self.compiled_model, nireq)#创建推理队列，读取模型创建一个推理请求
        self.pre_api=False


    def argmax(self,ptr,len):
        max_arg=0
        i=1
        while(i<len):
            i+=1
            if(ptr[i]>ptr[max_arg]):
                max_arg=i

        return max_arg 


    #归一化函数，v7模型中的三个输出张量需要进行归一化处理,适合上交的模型
    def sigmoid(self, x):
        
        return 1 / (1 + np.exp(x))
    

    def inv_sigmoid(self,x):

        return -np.log( 1 / x-1) 

        
#预处理函数，该函数计算输入图像尺寸和需求尺寸之间的比值，对图像大小进行重塑，并且给图像添加一个边界
    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
            new_unpad[1]  # wh padding

        # divide padding into 2 sides
        dw /= 2
        dh /= 2

        # resize
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        # add border
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        
        return img
       

    #decode部分函数，需要将v5/v7模型的xywh数据转成opencv的xyxy数据
    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y

        return y
    
    #decode部分函数，进行非极大值抑制，防止多个目标框的出现，只返回置信度最高的
    def nms(self, prediction, conf_thres, iou_thres):
        predictions = np.squeeze(prediction[0])

        # Filter out object confidence scores below threshold
        obj_conf = predictions[:, 4]
        predictions = predictions[obj_conf > conf_thres]
        obj_conf = obj_conf[obj_conf > conf_thres]

        # Multiply class confidence with bounding box confidence
        predictions[:, 5:] *= obj_conf[:, np.newaxis]

        # Get the scores
        scores = np.max(predictions[:, 5:], axis=1)

        # Filter out the objects with a low score
        valid_scores = scores > conf_thres
        predictions = predictions[valid_scores]
        scores = scores[valid_scores]

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 5:], axis=1)

        # Get bounding boxes for each object
        boxes = self.xywh2xyxy(predictions[:, :4])

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # indices = nms(boxes, scores, self.iou_threshold)
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thres, iou_thres)
        #print(indices)

        return boxes[indices], scores[indices], class_ids[indices]




    
    #画框函数,将会截取目标roi区域传入threshold函数进行精确，最终返回精确后的每个类别的左上角点和右下角点
    def plot_one_box(self, x, img,label, color=None, line_thickness=None): 
        
        # Plots one bounding box on image img
        tl = line_thickness or round(
            0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0][0]), int(x[0][1])), (int(x[0][2]), int(x[0][3]))
        # print(c1)
        # print(c2)

        #cv2.rectangle(img, c1, c2, color, thickness=1, lineType=cv2.LINE_AA)
        if label:
           
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(
                label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            #cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            #cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                        #[0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

 
    #调用plot_one_box函数进行画框,并且进行串口数据发送，卡尔曼预测函数
    def draw(self, img, boxinfo):
        cls_list=[]

        for xyxy, conf, cls in boxinfo:
            self.TARGET=True
            cls_list.append(cls)
            if(self.classes[int(cls)]!='banana'):    
                self.plot_one_box(xyxy, img, label=self.classes[int(cls)], color=(0,255,0), line_thickness=2)
            # #画出基准线
            # cv2.line(img, (0,320),(640,320), (0,0,255), 1)
            # cv2.line(img, (320,0),(320,640), (0,0,255), 1)
        if(cls_list==[]):
            self.TARGET=False
        #reset
        cls_list=[]

        self.fina_image=img

        # cv2.imshow("result",self.fina_image)
        # cv2.waitKey(1)
        


    #decode部分核心，后处理函数，对得到的输出信息进行整合，有关各种输出信息的解码
    def postprocess(self, infer_request, info):
        src_img_list, src_size = info
        for batch_id in range(self.batchsize):
            if self.grid:
                results = np.expand_dims(infer_request.get_output_tensor(0).data[batch_id], axis=0)
                
            else:
                continue
                # output = []
                # # Get the each feature map's output data
                # output.append(self.sigmoid(infer_request.get_output_tensor(2).data[batch_id].reshape(-1, self.size[0]*3, 5+self.class_num)))
                # output.append(self.sigmoid(infer_request.get_output_tensor(1).data[batch_id].reshape(-1, self.size[1]*3, 5+self.class_num)))
                # output.append(self.sigmoid(infer_request.get_output_tensor(0).data[batch_id].reshape(-1, self.size[2]*3, 5+self.class_num)))
                
                # # Postprocessing
                # grid = []
                # for _, f in enumerate(self.feature):
                #     grid.append([[i, j] for j in range(f[0]) for i in range(f[1])])

                # result = []
                # for i in range(3):
                #     src = output[i]
                #     xy = src[..., 0:2] * 2. - 0.5
                #     wh = (src[..., 2:4] * 2) ** 2
                #     dst_xy = []
                #     dst_wh = []
                #     for j in range(3):
                #         dst_xy.append((xy[:, j * self.size[i]:(j + 1) * self.size[i], :] + grid[i]) * self.stride[i])
                #         dst_wh.append(wh[:, j * self.size[i]:(j + 1) *self.size[i], :] * self.anchor[i][j])

                #     src[..., 0:2] = np.concatenate((dst_xy[0], dst_xy[1], dst_xy[2]), axis=1)
                #     src[..., 2:4] = np.concatenate((dst_wh[0], dst_wh[1], dst_wh[2]), axis=1)
                #     result.append(src)
                #results = np.concatenate(result, 1)
                
            boxes, scores, class_ids = self.nms(results, self.conf_thres, self.iou_thres)
            #img_shape = self.img_size
            # self.scale_coords(img_shape, src_size, boxes)

            # Draw the results
            self.draw(src_img_list[batch_id], zip(boxes, scores, class_ids))

    #推理函数，调用图像
    def infer_image(self, img_path):

        #print("<<<<start inference.......................")
        # Read image
        src_img =img_path 

        #调整图像大小
        src_img = cv2.resize(src_img, (640,640))
        src_img_list = []
        src_img_list.append(src_img)
        img = self.letterbox(src_img, self.img_size)
        src_size = src_img.shape[:2]
        img = img.astype(dtype=np.float32)

        #是否进行前处理
        if (self.pre_api == False):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB
            img /= 255.0
            img = img.transpose(2, 0, 1) # NHWC to NCHW
        input_image = np.expand_dims(img, 0)

        # 设置后处理函数为回调函数
        self.infer_queue.set_callback(self.postprocess)
        # 进行推理
        self.infer_queue.start_async({self.input_layer.any_name: input_image}, (src_img_list, src_size))
        self.infer_queue.wait_all()

        return self.TARGET,self.fina_image

        



    #推理函数，调用摄像头，目前采取的是opencv读取，后期整合迈德威视包
    def infer_cam(self):
        # Set callback function for postprocess
        self.infer_queue.set_callback(self.postprocess)
        # Capture camera source
        src_img_list = []
        cap=cv2.VideoCapture(2)
        while(1): 
            #frame = self.cap.read()
            ret,frame=cap.read() 
            img = self.letterbox(frame, self.img_size)
            img_copy=img.copy()
            #print(img.shape[ :2])
            src_size = frame.shape[:2]
            img = img.astype(dtype=np.float32)

            start_time = time.time()

            #是否进行前处理
            if (self.pre_api == False):
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB
                img /= 255.0
                img = img.transpose(2, 0, 1) # NHWC to NCHW

            input_image = np.expand_dims(img, 0)

            # # Batching
            # img_list.append(input_image)
            src_img_list.append(img_copy)
            # if (len(img_list) < self.batchsize):
            #     continue
            # img_batch = np.concatenate(img_list)

            # Do inference
            self.infer_queue.start_async({self.input_layer.any_name: input_image}, (src_img_list, src_size))

            #计算帧率
            #end_time = time.time()
            #fps = 1 / (end_time - start_time)
            #print("time",end_time-start_time)
            #print("throughput: {:.2f} fps".format(fps))

            #reset
            src_img_list = []
            
        self.cap.stop()
        #self.port.close_port()
        cv2.destroyAllWindows() 