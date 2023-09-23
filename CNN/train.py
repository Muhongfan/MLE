import torch
import matplotlib.pyplot as plt
from CNN.model import CNNnet
from CNN.device import device

def train(train_epoch : int, model_save : str, train_loader, test_loader) :
    # 模型
    model = CNNnet()
    model.to(device)
    # 定义损失和优化器
    # 使用交叉熵损失函数
    loss_func = torch.nn.CrossEntropyLoss( )
    # 使用Adam函数优化器,输入模型的参数与学习率lr
    # opt = torch.optim.Adam(model.parameters( ),lr=0.001)
    # SGD 就是随机梯度下降
    # opt = torch.optim.SGD(model.parameters( ), lr=0.001)
    # momentum 动量加速,在SGD函数里指定momentum的值即可
    #opt = torch.optim.SGD(model.parameters( ), lr=0.001, momentum=0.8)
    # RMSprop 指定参数alpha
    opt = torch.optim.RMSprop(model.parameters( ), lr=0.001, alpha=0.9)

    loss_count = []  # 定义一个list，用于存储损失函数数据
    for epoch in range(train_epoch):
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)  # torch,Size([128,1,28,28])将张量转为变量
            y = y.to(device)  # torch.Size([128])
            # 获取最后输出
            out = model(x)  # torch.Size([128,10])
            # 获取损失
            loss = loss_func(out, y)
            # 使用优化器优化损失
            opt.zero_grad()  # 清空上一步残余更新参数值,把模型中参数的梯度设为0
            loss.backward()  # 误差反向传播计算参数更新值,计算与图中叶子结点有关的当前张量的梯度
            opt.step()  # 将参数更新值施加到net的parameters上,根据梯度更新网络参数
            # 每20次输出一次当前的损失函数值，并保存模型权重
            if i % 20 == 0:
                loss_count.append(loss.item())
                print('{}:\t'.format(i), loss.item())
                # torch.save(model.state_dict(),model_save)
                torch.save(model, model_save)
            # 每100次输出一个当前softmax层的最大概率
            if i % 100 == 0:
                for a, b in test_loader:
                    a = a.to(device)
                    b = b.to(device)
                    out = model(a)
                    # print('test_out：\t',torch.max(out,1)[1])
                    # print('test_y:\t',test_y)
                    accuarcy = torch.max(out, 1)[1].to("cpu").numpy() == b.to("cpu").numpy()  # 获得当前softmax层最大概率对应的索引值
                    print('accuary：\t', accuarcy.mean())
                    break
    # 绘制损失函数曲线
    plt.figure('PyTorch_CNN_Loss')
    plt.plot(loss_count, label='Loss')
    plt.legend('Loss')  # 给图片加图例
    plt.show()
