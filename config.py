import warnings

class DefaultConfig(object):
    env = 'default'
    train_data_root = './data/'
    test_data_root = './data/'
    load_model_path = None
    save_model_path = './checkpoints/model.pth'
    result_file = 'result.csv'

    # 超参数
    use_cuda = True
    batch_size_train = 100
    batch_size_test = 1000
    num_epochs = 15   # 训练轮数
    log_interval = 6  # 每隔6个batch输出一次训练日志
    learning_rate = 0.01
    momentum = 0.5

    def print_config(self):
        '''
        打印配置信息
        '''
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            # 不打印函数
            if not k.startswith('__') and not callable(v):
                space_num = 18 - len(k)
                space = ' ' * space_num
                print('\033[0;32m', k, '\033[0m', space, getattr(self, k))
        print()

    def parse(self, kwargs):
        '''
        根据字典 kwargs 更新 config 参数
        '''
        # 更新配置参数
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" %k)
            setattr(self, k, v)
        
        # 打印配置信息
        self.print_config()

    def get_keys(self):
        '''
        获取配置参数的键
        '''
        return [k for k, v in self.__class__.__dict__.items() if not k.startswith('__')]