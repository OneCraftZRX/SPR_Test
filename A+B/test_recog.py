import torch
from torchvision import transforms as transforms
import PIL.Image
from train import Net
import os
import cvpreproctest
import cv2

Infer_model = Net()  # 获得网络结构
Infer_model.load_state_dict(torch.load('./model_AB.pth')) # 加载最后训练的参数，在进行推理时，不需要优化器（optimizer），因为优化器只在训练时用于更新模型参数。
Infer_model.eval()  # 将模型转化为评估模型，此时虽然训练模型的batch_size是64，但是推理的时候可以只用一张图片
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

folder=r"C:\Users\25176\OneDrive\Codes\A+B\samples"
imagefolder=r"C:\Users\25176\OneDrive\Codes\A+B\samples\inputs"
expfoloder=r"C:\Users\25176\OneDrive\Codes\A+B\samples\explanations"
files = os.listdir(imagefolder)   # 读入文件夹
num_png = len(files)       # 统计文件夹中的文件个数

total_correct=0
acc=0
sumsum=0

for j in range(1,num_png+1):
    imagepath=os.path.join(imagefolder,str(j)+".jpg")
    preproc=cvpreproctest.preproc1(imagepath)
    imglist=preproc.result()
    answerfile=open(os.path.join(expfoloder,str(j)+".txt"),"r")
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
    # print(xs)
    print("rightnums:",rightnums)
    if len(imglist)!=2:
        print("len(imglist)!=2")
        print(j)
        break
    for i in range(2):
        pil_img = PIL.Image.fromarray(preproc.imglist[i])
        pil_img = pil_img.resize((28,28))  # 输入尺寸与网络的输入保持一致
        pil_img = pil_img.convert('L')     # 转为灰度图，保持通道数与网络输入一致
        # pil_img.show()
        pil_img = transform(pil_img)
        with torch.no_grad():
            output= Infer_model(pil_img)  # 得到推理结果
        # 返回函数的最大值的下标
        pred = torch.argmax((output))
        print('Prediction:', pred.item())
        if pred.item()==rightnums[i]:
            total_correct+=1
        sumsum+=1
    print("this : ",j," total correct:",total_correct)
    cv2.waitKey(0)

acc=total_correct/sumsum
print("total acc:",acc)