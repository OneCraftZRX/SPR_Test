import cv2
import os

# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)) 
def open(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    dst = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("open", dst)
    return dst

def close(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    dst = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("close", dst)
    return dst


class preproc1:    
    def __init__(self,path):
        self.imglist=[]
        self.path=path
        self.img = cv2.imread(self.path)
        # 转变成单通道
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # 二值化,第一个返回值是执行的结果和状态是否成功，第二个返回值才是真正的图片结果
        self.ret, self.binary = cv2.threshold(self.gray, 200, 255, cv2.THRESH_BINARY)
        self.binary=255-self.binary
        # cv2.imshow("binary",self.binary)
        # cv2.waitKey(0)
        # 轮廓查找,第一个返回值是轮廓，第二个是层级
        self.open_res=close(self.binary)
        # cv2.imshow("open",self.open_res)
        self.contours, self.h = cv2.findContours(self.open_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # cv2.drawContours(self.img, self.contours, -1, (0, 255, 0), 1)  # 改变的是img这张图
        # cv2.imshow("contours",self.img)

        print("len(self.contours):",len(self.contours))
        # self.xs=[]
        # self.ys=[]
        # self.squares=[]
        self.lastx=0
        REV_FLAG=False
        for i in range(len(self.contours)):#rect
            x, y, w, h = cv2.boundingRect(self.contours[i])
            square=max(w,h)
            if square>5:
                # self.xs.append(x)
                # self.ys.append(y)
                # self.squares.append(square)
                self.imglist.append(self.img[y:y+square, x:x+square])
                if x<self.lastx:
                    REV_FLAG=True
                self.lastx=x
            if REV_FLAG:
                self.imglist=self.imglist[::-1]
        # print("len(self.imglist):",len(self.imglist))
            
    def result(self):
        return self.imglist

# imagefolder=r"C:\Users\25176\OneDrive\Codes\A+B\samples\inputs"
# expfoloder=r"C:\Users\25176\OneDrive\Codes\A+B\samples\explanations"
# files = os.listdir(imagefolder)   # 读入文件夹
# num_png = len(files)       # 统计文件夹中的文件个数

# for i in range(773,num_png+1):
#     imagepath=os.path.join(imagefolder,str(i)+".jpg")
#     preproc=preproc1(imagepath)
#     list1=preproc.result()
#     for i in range(len(list1)):
#         cv2.imshow(str(i),list1[i])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()