import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import os

def file_name(file_dir):
    list = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                list.append(os.path.join(root, file))
    return list

class Dogcat_Dataset(Dataset):
    def __init__(self, train_dir, test_dir,transform=True, train=True):
        self.transform = transform
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.train = train


    def __getitem__(self, item):
        if self.train:
            path = file_name(self.train_dir)[item]
            image = read_image(path)
            label = self.find_label(path)
            sample = {"image": image, "label": label}
            return sample
        else:
            path = file_name(self.test_dir)[item]
            image = read_image(path)
            label = self.find_label(path)
            sample = {"image": image, "label": label}
            return sample

    # 回归一个dic 里面 image 为 numpy arr label 为 label
    def __len__(self):
        if self.train:
            return(len(file_name(self.train_dir)))
        else:
            return(len(file_name(self.test_dir)))
    # 回归一个长度

    def find_label(self, name):
        if name.count('dog') != 0:
            return 'dog'
        if name.count('cat') != 0:
            return 'cat'

