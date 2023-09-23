import torch
class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet,self).__init__()# super用于调用父类(超类)的构造函数
        self.conv1 = torch.nn.Sequential(
            # 二维卷积
            torch.nn.Conv2d(in_channels=1,# 输入图片的通道数
                            out_channels=16,# 卷积产生的通道数
                            kernel_size=3,# 卷积核尺寸
                            stride=2,# 步长,默认1
                            padding=1),# 补0数，默认0
            # 数据在进入ReLU前进行归一化处理，num_features=batch_size*num_features*height*width
            # 先分别计算每个通道的方差和标准差，然后通过公式更新矩阵中每个值，即每个像素。相关参数：调整方差，调整均值
            # 输出期望输出的(N,C,W,H)中的C (数量，通道，高度，宽度)
            # 实际上，该层输入输出的shape是相同的
            torch.nn.BatchNorm2d(16),
            # 设置该层的激活函数RELU()
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            # torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16,32,3,2,1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            # torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32,64,3,2,1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            # torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64,64,2,2,0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        # 全连接层参数设置
        self.mlp1 = torch.nn.Linear(2*2*64,100)# 为了输出y=xA^T+b,进行线性变换（输入样本大小，输出样本大小）
        self.mlp2 = torch.nn.Linear(100,10)
    def forward(self,x):# 前向传播
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mlp1(x.view(x.size(0),-1))# 将多维度的tensor展平成一维
        x = self.mlp2(x)
        return x
print(CNNnet())#输出模型结构

