import cv2
import numpy as np

class color_process:
    def detect(self,frame):
        frame = cv2.resize(frame, (640,640))


        # 将彩色图像转换为灰度图像
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        #parameter to efine the range of black color in HSV
        self.lower_black = (0, 0, 0)
        self.upper_black = (180, 255, 120)

        #convert to hsv
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Threshold the HSV image to extract black color
        mask = cv2.inRange(hsv, self.lower_black, self.upper_black)

        # 定义腐蚀操作的核
        kernel = np.ones((5,5),np.uint8)

        # 对图像进行腐蚀操作
        mask = cv2.erode(mask,kernel,iterations = 1)


        
        # Find contours in the binary image
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
        # Find the contour with the largest area
        max_area = 0
        max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

        # Draw the largest contour on the input image
        if max_contour is not None:
            x, y, w, h = cv2.boundingRect(max_contour)

# # 在图像上绘制外接矩形
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.drawContours(frame, [max_contour], 0, (0, 255, 0), 2)

        return frame