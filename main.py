import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse

from data.dataset import mnist
from config import DefaultConfig
from models import Net
from utils import show_info, show_loss
from train import train
from test import test

def load_config():
    opt = DefaultConfig()
    # TODO
    parser = argparse.ArgumentParser(description='PyTorch Example', conflict_handler='resolve', 
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-cpu', '--use_cpu', action='store_false', default=opt.use_cuda, dest='use_cuda', help='use cuda or not')
    parser.add_argument('-e', '--epochs', type=int, default=opt.num_epochs, dest='num_epochs', help='number of epochs to train')
    parser.add_argument('-c', '--checkpoint', type=int, default=opt.log_interval, dest='log_interval',
                        help='batches to wait before logging training status')
    parser.add_argument('-btr', '--batch_size_train', type=int, default=opt.batch_size_train, help='input batch size for training')
    parser.add_argument('-bte', '--batch_size_test', type=int, default=opt.batch_size_test, help='input batch size for testing')
    parser.add_argument('-lr', '--learning_rate', type=float, default=opt.learning_rate, help='learning rate')
    parser.add_argument('-m', '--momentum', type=float, default=opt.momentum, help='SGD momentum')
    
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0', help='show program\'s version number and exit')
    parser.add_argument('-si', '--show_info', action='store_true', default=False, help='show dataset information')
    parser.add_argument('-pn', '--print_net', action='store_true', default=False, help='print network')
    parser.add_argument('-otr', '--only_train', '-nte', '--no_test', action='store_true', default=False, help='only train')
    parser.add_argument('-te', '--test_once', action='store_true', default=False, help='test the existing model')
    parser.add_argument('-r', '--recognize', action='store_true', default=False, help='recognize image uploaded by user') # TODO
    parser.add_argument('-s', '--save', action='store_true', default=False, help='save model') # TODO

    parser.add_argument('-nsc', '--no_show_checkpoint', action='store_true', default=False, help='will not show checkpoint logs')
    parser.add_argument('-nsl', '--no_show_loss', action='store_true', default=False, help='will not show losses of train and test')
    
    args = parser.parse_args()
    opt.parse({k: v for k, v in vars(args).items() if k in opt.get_keys()})
    actions = {k: v for k, v in vars(args).items() if k not in opt.get_keys()}
    return opt, actions

def main():
    # 读取配置参数与命令行参数
    opt, actions = load_config()

    # 保证每次运行py文件时，生成的随机数相同，结果可以复现
    random_seed = 2
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    # 调用GPU
    torch.backends.cudnn.benchmark = True # 启用cudnn底层算法
    device = torch.device("cuda" if opt.use_cuda and torch.cuda.is_available() else "cpu")
    print("Using device \033[1;34m{}\033[0m.\n".format(device))

    if actions['show_info']:
        show_info()

    network = Net().to(device)
    if actions['print_net']:
        network.print_net()

    if actions['recognize']:
        # recognize(device, network) # TODO
        return

    if actions['test_once']: # TODO
        # 加载测试集
        test_loader = DataLoader(mnist(train=False).data_set, opt.batch_size_test, shuffle=True)
        # 测试
        test(device, network, test_loader, [], [], no_record=False)
        return
    
    # 加载数据集
    train_loader = DataLoader(mnist(train=True).data_set, opt.batch_size_train, shuffle=True)
    test_loader = DataLoader(mnist(train=False).data_set, opt.batch_size_test, shuffle=True)

    # 定义优化器
    optimizer = optim.SGD(network.parameters(), lr=opt.learning_rate, momentum=opt.momentum)

    # 数据记录
    train_set_len = len(train_loader.dataset) # 训练集大小  # type: ignore
    test_set_len  = len(test_loader.dataset) # 测试集大小  # type: ignore
    train_losses  = [] # 记录训练损失
    train_counter = [] # 记录训练的批次
    test_losses   = [] # 记录测试损失
    test_counter  = [i*train_set_len for i in range(opt.num_epochs + 1)] # 记录测试的批次

    # 多轮训练和测试
    for epoch in range(1, opt.num_epochs + 1):
        if actions['only_train'] is False:
            test_losses = test(device, network, test_loader, test_set_len, test_losses, no_record=actions['no_show_loss'])
        train_losses, train_counter = train(epoch, device, network, optimizer, train_loader, opt.log_interval, 
                                            opt.batch_size_train, train_set_len, train_losses, train_counter, 
                                            no_record=actions['no_show_loss'], no_show_log=actions['no_show_checkpoint'])
    if actions['only_train'] is False: 
        test_losses = test(device, network, test_loader, test_set_len, test_losses, no_record=actions['no_show_loss'])

    if actions['no_show_loss'] is False:
        show_loss(train_losses, train_counter, test_losses, test_counter, no_test=actions['only_train'])
            

if __name__ == '__main__':
    main()

    

