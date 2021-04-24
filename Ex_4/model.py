"""
Using Dogcat Dataset to Train Classification Pytorch Network
2021/04/20  MYZ
"""
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from Ex_4.dataset.dataset import Dogcat_Dataset
from torchvision.io import read_image
from torch.autograd import Variable



class DogCat_Net(nn.Module):
    def __init__(self):
        super(DogCat_Net, self).__init__()
        # image size 3,256,256
        in_channels = 3
        out_channels = 8
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cov1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(3,3),padding=1)
        self.l1 = nn.ReLU()
        self.cov2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=(3,3),padding=1)
        self.l2 = nn.ReLU()
        self.max1 = nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0)
        self.l3 = nn.Linear(64*64*8, 1000)
        self.l4 = nn.ReLU()
        self.l5 = nn.Linear(1000, 1000)
        self.l6 = nn.ReLU()
        self.l7 = nn.Linear(1000, 2)
        self.l8 = nn.ReLU()
        self.l9 = nn.Softmax(dim=1)
        # todo 可能要 batchnorm 一下方便训练
    def forward(self,x):
        x1 = self.cov1(x)
        x2 = self.l1(x1)
        x3 = self.cov2(x2)
        x4 = self.l2(x3)
        x5 = self.max1(x4)
        x6 = torch.flatten(x5, start_dim=1,end_dim=3)
        x7 = self.l3(x6)
        x8 = self.l4(x7)
        x9 = self.l5(x8)
        x10 = self.l6(x9)
        x11 = self.l7(x10)
        x12 = self.l8(x11)
        x13 = self.l9(x12)

        return x13






if __name__ == '__main__':
    model = DogCat_Net()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # image = torch.randn(1,3,256,256)
    # y = torch.tensor([0,1])
    # preds = model.forward(image)
    #
    # loss = loss_fn(preds, y)
    # print(preds)

    training_data = Dogcat_Dataset(train_dir='dataset/train_set', test_dir='dataset/test_set')
    test_data = Dogcat_Dataset(train_dir='dataset/train_set', test_dir='dataset/test_set', train=False)

    # train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
    # test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    # for batch, (X, y) in enumerate(train_dataloader):
    #     print(f'Batch is {batch}')
    #
    #     print(X.type(),X.size())

    # train_features, train_labels = next(iter(train_dataloader))
    # train_features, train_labels = Variable(train_features), Variable(train_labels)
    X = torch.zeros(1,3,128,128)
    y = torch.zeros(1,2)

    for img, label in training_data:
        X[0,:] = img[:]
        y[0,:] = label[:]
    # print(X.type(),X.size())
    preds = model.forward(X)
    loss = loss_fn(preds,y)