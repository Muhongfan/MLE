import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
import numpy as np
# # Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
# # Hyper parameters
# # num_epochs = 5
# # num_classes = 10
# # learning_rate = 0.001
# # batch_size = 100

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=100,
                                          shuffle=False)

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),  # 一般Kernel_size=5,padding=2
            nn.BatchNorm2d(16),  # make feature's mean_value=1,variance=1,learn or fit better from good distribution
            nn.ReLU(),  # standard activation fuction for cnn
            nn.MaxPool2d(kernel_size=2, stride=2))  # demension_reduce
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Linear(3 * 3 * 48, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = ConvNet(10).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
total_step = len(train_loader)
for epoch in range(5):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, 5, i + 1, total_step, loss.item()))

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))


# Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')
# torch.save(model.state_dict(), "my_model.pth")  # 只保存模型的参数
#torch.save(model, "./my_model.pth")  # 保存整个模型

# from PIL import Image

model_path = r'CNN/my_model.pth'

image_path = r'/data/img/num_8.png'  # 自己写的黑底数字

# 加载数据
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 读取图片，并指定模式为灰度
# 数据预处理
img = cv2.resize(img, (28, 28))  # 调整图片为28*28

# 白底黑字 -> 黑底白字
img = abs(255-img)

# Normalization
img = img/255
img = (img-0.5)/0.5

# dst = np.zeros(img.shape, dtype=np.float64) # 返回形状和img相同的，全部用0填充的numpy数组
# img = cv2.normalize(img, dst=dst, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX)

plt.imshow(img, cmap='gray')
plt.show()

# 图片转换与预测
img = torch.from_numpy(img).float()  # 将img 从numpy类型转为torch类型，并限定数据元类型为float
# [100, 1, 28, 28]
img = img.view(1, 1, 28, 28)  # 改变维度，用于对应神经网络的输入

img = img.to(device)  # 将img内容copy一份放到GPU上去


# 加载模型
net = torch.load(model_path)
net.to(device)


outputs = net(img)  # 将测试的图片输入网络中img

# 原始计算数据
probability, predicted = torch.max(outputs.data, 1)  # 0是列最大值，1是行最大值；return（每行最大值,最大值的索引）
print(outputs.to('cpu').detach().numpy().squeeze())  # 输出概率数组
print("the probability of this number being {} is {}"
      .format(predicted.to('cpu').numpy().squeeze(),
              probability.to('cpu').numpy().squeeze()))  # 预测结果

# 归一化
softmax = nn.Softmax(dim = 1)
prob_matrix = softmax(outputs)
prob_matrix = prob_matrix.to('cpu').detach().numpy().squeeze()
label = predicted.to('cpu').item()
print(prob_matrix)  # 输出概率数组
print("the probability of this number being {} is {:0.4f}%"
      .format(label, prob_matrix[label]*100))  # 预测结果




#
# # 白底黑字 -> 黑底白字
# img = abs(255-img)
#
#
#
# # Normalization

#