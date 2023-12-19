import os

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config import DefaultConfig
from data import mnist


def show_info():
    opt = DefaultConfig()

    train_set = mnist(data_root=opt.train_data_root, train=True)
    test_set = mnist(data_root=opt.test_data_root, train=False)
    train_loader = DataLoader(train_set, opt.batch_size_train, shuffle=True)
    test_loader = DataLoader(test_set, opt.batch_size_test, shuffle=True)

    print(f"训练集大小：{len(train_set)}\n测试集大小：{len(test_set)}\n")

    examples = enumerate(train_loader) # 一个batch的训练集数据
    batch_idx, (example_data, example_targets) = next(examples)
    print(f"一批次的训练集数据大小：{len(example_data)}\n"
        f"一批次的训练集数据形状(批次大小, 通道数, 图像高度, 图像宽度)：{example_data.shape}\n")
    # print(example_targets)

    examples = enumerate(test_loader) # 一个batch的测试集数据
    batch_idx, (example_data, example_targets) = next(examples)
    print(f"一批次的测试集数据大小：{len(example_data)}\n"
        f"一批次的测试集数据形状(批次大小, 通道数, 图像高度, 图像宽度)：{example_data.shape}\n")
    # print(example_targets)

    img_path = './img'
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    h, w = 4, 8
    scale = 1.5
    figsize = (w * scale, h * scale)
    plt.figure(figsize=figsize, dpi=100)      # 创建一个图像窗口，设置大小和分辨率
    for i in range(h * w):
        plt.subplot(h, w, i + 1)                    # 2行5列, 第i+1个子图
        plt.tight_layout()                          # 自动调整子图参数, 使之填充整个图像区域
        plt.imshow(example_data[i][0], cmap='gray') # 画图，灰度图
        plt.title(f"{example_targets[i]}")          # 设置子图标题，真实值
        plt.xticks([])                              # 设置x轴刻度
        plt.yticks([])                              # 设置y轴刻度
    plt.savefig(img_path + '/test_set_example.png') # 保存图片
    plt.show()


def show_loss(train_losses, train_counter, test_losses, test_counter, 
              no_test=True, figheight=5, figwidth=15):
    fig = plt.figure()
    fig.set_figheight(figheight)
    fig.set_figwidth(figwidth)
    plt.plot(train_counter, train_losses, color='#fa7f6f')
    if no_test:
        plt.legend(['Train Loss'], loc='upper right')
    else:
        plt.plot(test_counter, test_losses, color='#2878b5', marker='*')
        plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('Number of Training Examples')
    plt.ylabel('Negative Log Likelihood Loss')
    plt.savefig('./img/loss.png')
    plt.show()