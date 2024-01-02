import cv2

class preproc1:
    def __init__(self,path):
        self.imglist=[]
        self.path=path
        self.img = cv2.imread(self.path)
        # 转变成单通道
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # 二值化,第一个返回值是执行的结果和状态是否成功，第二个返回值才是真正的图片结果
        self.ret, self.binary = cv2.threshold(self.gray, 100, 255, cv2.THRESH_BINARY)
        # 轮廓查找,第一个返回值是轮廓，第二个是层级
        self.contours, self.hierarchy = cv2.findContours(self.binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # refCnts = myutils.sort_contours(self.contours, method="left-to-right")[0] #排序，从左到右，从上到下
        # print("len(self.contours):",len(self.contours))
        # self.contours=
        for i in range(1,len(self.contours)):
            #rect
            x, y, w, h = cv2.boundingRect(self.contours[i])
            square=max(w,h)
            self.imglist.append(self.img[y:y+square, x:x+square])
        # print("len(self.imglist):",len(self.imglist))
            
    def result(self):
        return self.imglist

# preproc1=preproc1()

# # 绘制轮廓
# cv2.drawContours(img, contours[1:3], -1, (0, 255, 0), 1)  # 改变的是img这张图

# list1=preproc1.result()
# for i in range(len(list1)):
#     cv2.imshow(str(i),list1[i])
# cv2.waitKey(0)