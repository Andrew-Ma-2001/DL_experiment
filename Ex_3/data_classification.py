import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class DC_net(nn.modules):

    def __init__(self):
        super(DC_net, self).__init__()
        self.l1 = nn.Linear(2, 15)
        self.l2 = nn.ReLU()
        self.l3 = nn.Linear(15, 2)
        self.l4 = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        x4 = self.l4(x3)
        return x4


# Creating dataset
data = torch.ones(100, 2)
x0 = torch.normal(2*data, 1)
x1 = torch.normal(-2*data, 1)
x = torch.cat((x0, x1))

y0 = torch.zeros(100)
y1 = torch.ones(100)
y = torch.cat((y0,y1)).type(dtype=torch.LongTensor)

# Using GPU to calculate
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
x = x.to(device)
y = y.to(device)
model = DC_net()
model = model.to(device)

# Setting Optimizer & Loss function
optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
loss_fun = nn.CrossEntropyLoss()
epoches = 100
# Training
# for i in range(epoches):
#     y_pre = model.forward(x)
#
#     loss = loss_fun(y_pre, y)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
