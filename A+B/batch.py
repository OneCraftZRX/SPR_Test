import os
import cv2
import cvpreproctest
import PIL.Image


imgout=r"C:\Users\25176\OneDrive\Codes\A+B\splited_img"
labelout=r"C:\Users\25176\OneDrive\Codes\A+B\splited_label"
folder=r"C:\Users\25176\OneDrive\Codes\A+B\samples"
imagefolder=r"C:\Users\25176\OneDrive\Codes\A+B\samples\inputs"
expfoloder=r"C:\Users\25176\OneDrive\Codes\A+B\samples\explanations"
files = os.listdir(imagefolder)   # 读入文件夹
num_png = len(files)       # 统计文件夹中的文件个数

def save(cvimg,path):
    pil_img = PIL.Image.fromarray(cvimg)
    pil_img = pil_img.resize((28,28))  # 输入尺寸与网络的输入保持一致
    pil_img = pil_img.convert('L')     # 转为灰度图，保持通道数与网络输入一致
    pil_img.save(path)

time=0
for i in range(1,num_png+1):
    imagepath=os.path.join(imagefolder,str(i)+".jpg")
    preproc=cvpreproctest.preproc1(imagepath)
    imglist=preproc.result()
    answerfile=open(os.path.join(expfoloder,str(i)+".txt"),"r")
    rightnums=[]
    xs=[]
    for i in range(2):
        line=answerfile.readline()
        # print(line.split(" "))
        num,x1,y1,x2,y2=line.split(" ")
        xs.append(int(x1))
        rightnums.append(int(num))
    if xs[0]<xs[1]:
        pass
    else:
        rightnums=rightnums[::-1]
    
    out1=os.path.join(imgout,str(i+time)+".jpg")
    save(imglist[0],out1)
    labelfile=open(os.path.join(labelout,str(i+time)+".txt"),"w")
    labelfile.write(str(rightnums[0]))
    labelfile.close()

    out2=os.path.join(imgout,str(i+time+1)+".jpg")
    save(imglist[1],out2)
    labelfile=open(os.path.join(labelout,str(i+time+1)+".txt"),"w")
    labelfile.write(str(rightnums[1]))
    labelfile.close()

    print("over")
    print(i+time,i+time+1)
    time+=2

# n1 x11 y11 x12 y12
# n2 x21 y21 x22 y22