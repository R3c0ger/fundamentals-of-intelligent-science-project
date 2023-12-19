import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# 保证每次运行py文件时，生成的随机数相同，结果可以复现
random_seed = 2 
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

# 调用GPU
use_cuda = 1
torch.backends.cudnn.benchmark = True # 启用cudnn底层算法
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
print("Using device {}.".format(device))


# 设置超参数
batch_size_train = 100
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
num_epochs = 15   # 训练轮数
log_interval = 6  # 每隔6个batch输出一次训练日志

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))])
train_set = datasets.MNIST(root='./data/', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='./data/', train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size_train, shuffle=True)
test_loader = DataLoader(test_set, batch_size_test, shuffle=True)


examples = enumerate(test_loader) # 生成一个枚举对象, 里面包含了一个batch的数据
batch_idx, (example_data, example_targets) = next(examples)
print(f"一个batch的数据大小：{len(example_data)}\t"
      f"一个batch的数据形状(批次大小, 通道数, 图像高度, 图像宽度)：{example_data.shape}")
# 当前这个批次有 1000 个样本，每个样本是单通道的 28x28 像素的图像。
# 一个批次的tensor图像是一个形为 (B, C, H, W) 的张量。
# print(example_targets)


# 若没有img文件夹，先创建一个
import os
if not os.path.exists('./img'):
    os.makedirs('./img')

fig = plt.figure(figsize=(10, 4), dpi=100)                              # 创建一个图像窗口，设置大小和分辨率
for i in range(10):
    plt.subplot(2, 5, i + 1)                                            # 2行5列, 第i+1个子图
    plt.tight_layout()                                                  # 自动调整子图参数, 使之填充整个图像区域
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')   # 画图，不插值；[i][0]表示第i个样本的第0个通道
    plt.title(f"Ground Truth: {example_targets[i]}")                    # 设置子图标题，真实值
    plt.xticks([])                                                      # 设置x轴刻度
    plt.yticks([])                                                      # 设置y轴刻度
plt.savefig('./img/test_set_example.png')                               # 保存图片
# plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # 定义池化层（汇聚层）
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 定义 Dropout 层
        self.dropout = nn.Dropout(p=0.5)
        # 定义全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 300)
        self.fc2 = nn.Linear(300, 10)
    
    def forward(self, x):
        # 卷积层和池化层
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # 展平操作
        x = x.view(x.size(0),-1)
        # 全连接层
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        # 使用 log_softmax 作为输出层的激活函数
        return F.log_softmax(x, dim=1)


network = Net() # 实例化网络
network = network.to(device) # 将网络移动到 GPU 上
summary(network, input_size=(1, 28, 28), batch_size=batch_size_train) # 打印网络结构

# 定义优化器
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
print(optimizer.state_dict())


train_set_len = len(train_loader.dataset) # 训练集大小  # type: ignore
test_set_len  = len(test_loader.dataset) # 测试集大小  # type: ignore
train_losses  = [] # 记录训练损失
train_counter = [] # 记录训练的批次
test_losses   = [] # 记录测试损失
test_counter  = [i*train_set_len for i in range(num_epochs + 1)] # 记录测试的批次

def train(epoch, device=device):
    network.train() # 训练模式
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device) # 将数据移动到 GPU 上
        optimizer.zero_grad()               # 梯度清零
        output = network(data)              # 前向传播
        loss = F.nll_loss(output, target)   # 计算损失
        loss.backward()                     # 反向传播
        optimizer.step()                    # 更新参数
        
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{train_set_len} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
            
            train_losses.append(loss.item()) # 记录训练损失
            train_counter.append((batch_idx * batch_size_train) + ((epoch - 1) * train_set_len)) # 记录训练的批次

            # 保存训练模型
            torch.save(network.state_dict(), './model.pth')
            torch.save(optimizer.state_dict(), './optimizer.pth')

def test(device=device):
    network.eval() # 测试模式
    test_loss = 0
    correct = 0
    with torch.no_grad(): # 禁用梯度计算，节约内存
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # 将一批的损失相加
            pred = output.argmax(dim=1, keepdim=True) # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item() # 计算正确的数量

    test_loss /= test_set_len
    test_losses.append(test_loss) # 记录测试损失
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{test_set_len} "
          f"({100. * correct / test_set_len:.2f}%)\n")

# 多轮训练和测试
for epoch in range(1, num_epochs + 1):
    test()
    train(epoch)
test()

fig = plt.figure()
fig.set_figheight(5)
fig.set_figwidth(15)
plt.plot(train_counter, train_losses, color='#fa7f6f')
plt.plot(test_counter, test_losses, color='#2878b5', marker='*')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('Number of Training Examples')
plt.ylabel('Negative Log Likelihood Loss')
plt.savefig('./img/loss.png')
plt.show()