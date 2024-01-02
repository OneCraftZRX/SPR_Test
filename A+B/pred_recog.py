import torch
from torchvision import transforms as transforms
import PIL.Image
from train import Net
import os
import cvpreproctest

Infer_model = Net()  # 获得网络结构
Infer_model.load_state_dict(torch.load('./model_AB.pth'))
Infer_model.eval()  # 将模型转化为评估模型，此时虽然训练模型的batch_size是64，但是推理的时候可以只用一张图片
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

imagefolder=r"C:\Users\25176\OneDrive\Codes\A+B\inputs"
outfoloder=r"C:\Users\25176\OneDrive\Codes\A+B\outputs"
files = os.listdir(imagefolder)   # 读入文件夹
num_png = len(files)       # 统计文件夹中的文件个数

for j in range(1,num_png+1):
    imagepath=os.path.join(imagefolder,str(j)+".jpg")
    anspath=os.path.join(outfoloder,str(j)+".txt")
    preproc=cvpreproctest.preproc1(imagepath)
    imglist=preproc.result()
    res=[]

    if len(imglist)!=2:
        print("number",j,"len(imglist)!=2")
        break
    for i in range(2):
        pil_img = PIL.Image.fromarray(preproc.imglist[i])
        pil_img = pil_img.resize((28,28))  # 输入尺寸与网络的输入保持一致
        pil_img = pil_img.convert('L')     # 转为灰度图，保持通道数与网络输入一致
        pil_img = transform(pil_img)
        with torch.no_grad():
            output= Infer_model(pil_img)  # 得到推理结果
        # 返回函数的最大值的下标
        pred = torch.argmax((output))
        print('Prediction:', pred.item())
        res.append(pred.item())
    ans=res[0]+res[1]
    ansfile=open(os.path.join(outfoloder,str(j)+".txt"),"w")
    ansfile.write(str(ans))
    ansfile.close()
    print("number",j,"ans=",ans)