"""
Using Dogcat Dataset to Train Classification Pytorch Network
2021/04/20  MYZ
"""
import torch.nn as nn
from Ex_4.dataset.dataset import Dogcat_Dataset
from torch.utils.data import DataLoader

class DogCat_Net(nn.modules):
    def __init__(self):
        super(DogCat_Net, self).__init__()
        self.l1 = nn.Conv2d()