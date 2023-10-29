import cv2
import numpy as np

def thresholdplus(src,show):
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        ret, dst = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
        #第一个返回值是执行的结果和状态是否成功，第二个返回值才是真正的图片结果
        if show==1:
            cv2.imshow('threshold', dst)
            return dst
        else:
            return dst


img_ori=cv2.imread('./testM.jpg')
thresholdplus(img_ori,1)
# #src = cv.imread("./testM.jpg")
# rawImage = cv2.imread("./testM.jpg")
cv2.waitKey(0)
cv2.destroyAllWindows()