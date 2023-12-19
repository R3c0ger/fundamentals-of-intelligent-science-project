from torch import nn
import torch.nn.functional as F
from torchsummary import summary

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
    
    def print_net(self):
        print(self, "\n")
        summary(self, (1, 28, 28))