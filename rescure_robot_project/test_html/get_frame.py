#（上位机端）从url链接拉取图像，需要树莓派开始图传但不访问，windows去网页上得到url然后拉取图像进行处理

import cv2
from my_openvino import YOLOV7_OPENVINO
from color import color_process

#初始化检测模型
yolov7_detector_leak=YOLOV7_OPENVINO("E:\\yolov5-7.0\\runs\\train\\exp12\\weights\\energy_conservation.onnx",'CPU', False, False)
#yolov7_detector_fence=YOLOV7_OPENVINO("E:\\yolov5-7.0\\runs\\train\\exp8\weights\\oil_fence.onnx",'CPU',["oil_fence"], False, False)


color_processor=color_process()
# 定义视频流的URL地址（示例）
ip_address = "192.168.139.187"
port_no = "5000"
url = f"http://{ip_address}:{port_no}/video_feed"


# 创建VideoCapture对象
cap = cv2.VideoCapture("C:\\Users\\53429\\Desktop\\1.mp4")
#cap = cv2.VideoCapture(url)

# 循环读取视频流中的每个帧
while True:
    ret, frame = cap.read()
    #frame=cv2.flip(frame,0)

    ok2,result_image=yolov7_detector_leak.infer_image(frame)

    cv2.imshow("result",result_image)
    cv2.waitKey(1)

    # ok,result_image=yolov7_detector_fence.infer_image(frame)

    # if(ret and ok):
      
    #     cv2.imshow("result",result_image)

    #     cv2.waitKey(1)

    # else:
    #     ok2,result_image=yolov7_detector_leak.infer_image(frame)

    #     cv2.imshow("result",result_image)
    #     cv2.waitKey(1)

# 释放资源
cap.release()
cv2.destroyAllWindows()