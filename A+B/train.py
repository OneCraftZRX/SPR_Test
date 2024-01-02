import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
import os
from PIL import Image

# Super parameter ------------------------------------------------------------------------------------
batch_size = 64
learning_rate = 0.01
momentum = 0.5
EPOCH = 10

imagefolder=r"C:\Users\25176\OneDrive\Codes\A+B\splited_img"
expfoloder=r"C:\Users\25176\OneDrive\Codes\A+B\splited_label"

testimagefolder=r"C:\Users\25176\OneDrive\Codes\A+B\splited_img_test"
testexpfoloder=r"C:\Users\25176\OneDrive\Codes\A+B\splited_label_test"
files1 = os.listdir(imagefolder)   # 读入文件夹
num_png = len(files1)       # 统计文件夹中的文件个数
files2 = os.listdir(testimagefolder)   # 读入文件夹
num_tests = len(files2)       # 统计文件夹中的文件个数


class CustomDatasetTrain(Dataset):
    def __init__(self):
        self.transform = transform
        self.image_paths = [os.path.join(imagefolder, f"{i}.jpg") for i in range(1,num_png+1)]
        self.label_paths = [os.path.join(expfoloder, f"{i}.txt") for i in range(1,num_png+1)]
        # print(len(self.image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        # 读取图像
        image = Image.open(img_path)
        transf = transforms.ToTensor()
        img_tensor = transf(image) 

        # 读取标签
        answerfile=open(label_path,"r")
        label = int(answerfile.read())

        return img_tensor, label
    
class CustomDatasetTest(Dataset):
    def __init__(self):
        self.transform = transform
        self.image_paths = [os.path.join(testimagefolder, f"{i}.jpg") for i in range(num_tests)]
        self.label_paths = [os.path.join(testexpfoloder, f"{i}.txt") for i in range(num_tests)]
        # print(len(self.image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        # 读取图像
        image = Image.open(img_path)
        transf = transforms.ToTensor()
        img_tensor = transf(image) 

        # 读取标签
        answerfile=open(label_path,"r")
        label = int(answerfile.read())

        return img_tensor, label

# Prepare dataset ------------------------------------------------------------------------------------
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# softmax归一化指数函数(https://blog.csdn.net/lz_peter/article/details/84574716),其中0.1307是mean均值和0.3081是std标准差

custom_dataset_train = CustomDatasetTrain()
custom_loader_train = DataLoader(custom_dataset_train, batch_size=batch_size, shuffle=True)
custom_dataset_test = CustomDatasetTest()
custom_loader_test = DataLoader(custom_dataset_test, batch_size=batch_size, shuffle=True)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50),
            torch.nn.Linear(50, 10),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)  # 一层卷积层,一层池化层,一层激活层(图是先卷积后激活再池化，差别不大)
        x = self.conv2(x)  # 再来一次
        x = x.view(batch_size, -1)  # flatten 变成全连接网络需要的输入 (batch, 20,4,4) ==> (batch,320), -1 此处自动算出的是320
        x = self.fc(x)
        return x  # 最后输出的是维度为10的，也就是（对应数学符号的0~9）

model = Net()

# Construct loss and optimizer ------------------------------------------------------------------------------
criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)  # lr学习率，momentum冲量


# Train and Test CLASS --------------------------------------------------------------------------------------
# 把单独的一轮一环封装在函数类里
def train(epoch):
    running_loss = 0.0  # 这整个epoch的loss清零
    running_total = 0
    running_correct = 0
    for batch_idx, data in enumerate(custom_loader_train, 0):
        inputs, target = data
        optimizer.zero_grad()

        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)

        loss.backward()
        optimizer.step()

        # 把运行中的loss累加起来，为了下面300次一除
        running_loss += loss.item()
        # 把运行中的准确率acc算出来
        _, predicted = torch.max(outputs.data, dim=1)
        running_total += inputs.shape[0]
        running_correct += (predicted == target).sum().item()

        if batch_idx % 300 == 299:  # 不想要每一次都出loss，浪费时间，选择每300次出一个平均损失,和准确率
            print('[%d, %5d]: loss: %.3f , acc: %.2f %%'
                  % (epoch + 1, batch_idx + 1, running_loss / 300, 100 * running_correct / running_total))
            running_loss = 0.0  # 这小批300的loss清零
            running_total = 0
            running_correct = 0  # 这小批300的acc清零

        torch.save(model.state_dict(), './model_AB.pth')
        torch.save(optimizer.state_dict(), './optimizer_AB.pth')


def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 测试集不用算梯度
        for data in custom_loader_test:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)  # dim = 1 列是第0个维度，行是第1个维度，沿着行(第1个维度)去找1.最大值和2.最大值的下标
            total += labels.size(0)  # 张量之间的比较运算
            correct += (predicted == labels).sum().item()
    acc = correct / total
    print('[%d / %d]: Accuracy on test set: %.1f %% ' % (epoch+1, EPOCH, 100 * acc))  # 求测试的准确率，正确数/总数
    return acc

# Start train and Test --------------------------------------------------------------------------------------
if __name__ == '__main__':
    acc_list_test = []
    for epoch in range(EPOCH):
        train(epoch)
        # if epoch % 10 == 9:  #每训练10轮 测试1次
        acc_test = test()
        acc_list_test.append(acc_test)
    plt.plot(acc_list_test)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy On TestSet')
    plt.show()