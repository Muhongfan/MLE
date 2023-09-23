#import torch
from torch.utils import data
import torchvision
from torchvision.datasets import mnist
from CNN.train import train
from CNN.test import test


# import numpy as np



def dataset(data_path : str , train_batch : int = 128, test_batch : int = 100, download : bool = True):
    # 数据集的预处理
    data_tf = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor( ),# 将原有数据转化成张量图像，值在(0,1)
            torchvision.transforms.Normalize([0.5],[0.5])# 将数据归一化到(-1,1)，参数（均值，标准差）。
        ]
    )
    # 数据集的获取
    train_data = mnist.MNIST(root=data_path,train=True,transform=data_tf,download=download)
    test_data = mnist.MNIST(root=data_path,train=False,transform=data_tf,download=download)

    # 获取数据batch
    train_loader = data.DataLoader(dataset=train_data,batch_size=train_batch,shuffle=True)# 训练数据，batch用于一次反向传播参数更新
    test_loader = data.DataLoader(dataset=test_data,batch_size=test_batch,shuffle=False)# 测试数据，shuffle是打乱数据的意思(多次取出batch_size)
    return (train_loader, test_loader)


# 数据集获取路径设置,你也可以用相对路径
data_path = r'/data'
model_save = r'/Users/amberm/PycharmProjects/MLE/MNISTcollection/my_model.pth'
image_path = r'/Users/amberm/PycharmProjects/MLE/data/image/num_8.png'  # 自己写的黑底数字
# image_path = r'./image/white3_1.jpg'  # 自己写的白底数字

# 加载数据
(train_loader, test_loader) = dataset(data_path=data_path, train_batch=64, test_batch=100)

# 模型训练
train(train_epoch=1, model_save=model_save, train_loader=train_loader, test_loader=test_loader)

# 模型训练效果测试
test(image_path=image_path, model_path=model_save)
