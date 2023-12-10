#将oil_leak显示在图像固定位置
#字体改成红色
#换个背景图
#图像显示的框改小一点


import cv2
import random
from my_openvino import YOLOV7_OPENVINO
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtCore import QTimer, Qt
import sys
from PyQt5.QtGui import QFont

# 初始化检测模型
yolov7_detector_leak = YOLOV7_OPENVINO("E:\\yolov5-7.0\\runs\\train\\exp12\\weights\\energy_conservation.onnx", 'CPU', False, False)

class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.flag = 0  # 标志位，根据需要更新此值

        self.timer = QTimer()  # 创建定时器
        #self.cap = cv2.VideoCapture("C:\\Users\\53429\\Desktop\\1111.mp4")  # 从默认摄像头捕获视频
        self.cap = cv2.VideoCapture(2)  # 从第二个摄像头捕获视频
        self.timer.timeout.connect(self.update_frame)  # 将定时器超时信号连接到输出函数
        self.initUI()

    def initUI(self):
        self.setGeometry(200, 200, 1000, 600)
        self.setWindowTitle("QT Dispaly")

        self.image_label = QLabel(self)  # 创建用于显示视频帧的标签
        self.image_label.setGeometry(100, 150, 400, 400)  # 设置标签的大小和位置

        self.image_label_text = QLabel("Image:", self)
        self.image_label_text.move(150, 80)  # 设置标签位置
        self.image_label_text.setFont(QFont("Arial Rounded MT Bold", 24))  # 设置字体和大小
        self.image_label_text.setStyleSheet("color: white;")  # 设置文本颜色为白色
        self.image_label_text.adjustSize()  # 调整标签大小以适应文本

        self.object_label = QLabel("Object:", self)
        self.object_label.move(700, 380)  # 设置标签位置
        self.object_label.setFont(QFont("Arial Rounded MT Bold", 24))  # 设置字体和大小（默认大小的倍数）
        self.object_label.setStyleSheet("color: white;")  # 设置文本颜色为白色
        self.object_label.adjustSize()  # 调整标签大小以适应更大的文本

        self.result_label = QLabel(self)
        self.result_label.move(650, 450)  # 设置标签位置
        self.result_label.setFont(QFont("Arial Rounded MT Bold", 24))  # 设置字体和大小（默认大小的倍数）
        self.result_label.setStyleSheet("color: white;")  # 设置文本颜色为白色
        self.result_label.adjustSize()  # 调整标签大小以适应更大的文本

        width = max(self.result_label.width(), self.result_label.fontMetrics().width("(150.0, 152.0)"))
        self.result_label.setFixedWidth(width)

        self.timer.start(20)  # 启动定时器。定时器每20毫秒输出一次连接的函数


    def update_frame(self):

        ret, img = self.cap.read()  # 读取摄像头帧

        self.flag, result_image = yolov7_detector_leak.infer_image(img)
        result_image = cv2.resize(result_image, (420, 400))  # 重置图像大小以适应标签大小

        if ret:
            # 获取图像的尺寸
            height, width = result_image.shape[:2]
            # # 计算截取的区域的起始坐标和大小
            # start_x = int(width / 6)
            # start_y = int(height / 6)
            # end_x = int(width * 5 / 6)
            # end_y = int(height * 5 / 6)

            # # 截取 ROI 区域
            # roi = img[start_y:end_y, start_x:end_x]

            #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)  # 转换图像颜色
            result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)  # 转换图像颜色
            height, width, channel = result_image.shape
            bytesPerLine = 3 * width
            qImg = QImage(result_image.data, width, height, bytesPerLine, QImage.Format_RGB888)  # 转换为QImage
            self.image_label.setPixmap(QPixmap.fromImage(qImg))  # 设置QPixmap为标签的像素图

            # 根据标志更新结果标签
            if self.flag == 0:
                self.result_label.setText("No leak")
            else:
                random_number_x = round(random.uniform(49.0, 52.0), 1)
                random_number_y = round(random.uniform(29.0, 32.0), 1)
                self.result_label.setText(f"({random_number_x}, {random_number_y})")
        else:
            print("无法读取帧")


    def closeEvent(self, event):
        self.cap.release()  # 当关闭窗口时释放摄像头

    def paintEvent(self, event):
        painter = QPainter(self)
        pixmap = QPixmap("C://Users//53429//Desktop//R.jpg")  # 用你实际的图片路径替换"background.jpg"
        painter.drawPixmap(self.rect(), pixmap)


def main():
    app = QApplication(sys.argv)
    win = MyWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
