from torchvision import datasets, transforms
from torch.utils.data import Dataset

class mnist(Dataset):
    def __init__(self, data_root='./data/', train=True):
        dataset_mean = (0.1307,)
        dataset_std = (0.3081,)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=dataset_mean, std=dataset_std)])
        self.data_set = datasets.MNIST(root=data_root, train=train, download=True, transform=self.transform)

    def __getitem__(self, index):
        return self.data_set[index]
    
    def __len__(self):
        return len(self.data_set)