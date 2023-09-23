import cv2
import torch
from CNN.device import device
import matplotlib.pyplot as plt

def test(image_path : str , model_path : str , show_image : bool = True):
    # 载入图片
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # 读取图片，并指定模式为灰度
    if img is not None:
        print('variable is not None')
        print(img.shape)
    else:
        print('variable is None')
    img = cv2.resize(img,(28, 28))  # 调整图片为28*28
    # 灰度化
    # img = np.float64(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY))
    # 归一化
    # dst = np.zeros(img.shape, dtype=np.float64) # 返回形状和img相同的，全部用0填充的numpy数组
    # img = cv2.normalize(img, dst=dst, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    img = img/255
    img = (img - 0.5)/0.5

    # 显示图片
    # if show_image == True:
    # cv2.namedWindow('input_image', cv2.WINDOW_AUTOSIZE)
    # cv2.resizeWindow("input_image", 640, 480);
    # cv2.imshow('input_image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    plt.imshow(img, cmap='gray')
    plt.show()


    # 加载模型
    net = torch.load(model_path)
    net.to(device)  # 放入GPU中

    # 图片转换与预测
    img = torch.from_numpy(img).float() # 将img 从numpy类型转为torch类型，并限定数据元类型为float
    img = img.view(1, 1, 28, 28)    # 改变维度，用于对应神经网络的输入
    img = img.to(device)    # 将img内容copy一份放到GPU上去
    outputs = net(img)  # 将测试的图片输入网络中
    probability, predicted = torch.max(outputs.data, 1)   # 0是列最大值，1是行最大值；return（每行最大值,最大值的索引）
    print(outputs.to('cpu').detach().numpy().squeeze())   # 输出概率数组
    print("the probability of this number being {} is {}"
        .format(predicted.to('cpu').numpy().squeeze(),
                probability.to('cpu').numpy().squeeze()))   # 预测结果

