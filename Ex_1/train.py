"""
Using Pytorch to Predict XOR calculation
2021/03/27  MYZ
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score

# Using GPU to calculate
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n = 500
x = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
x = x.to(device)
y = torch.Tensor([[1,0] if k[0]+k[1]==1 else [0, 1] for k in x])
y = y.to(device)

class XOR_net(nn.Module):
    def __init__(self):
        super(XOR_net, self).__init__()
        self.l1 = nn.Linear(2, 20)
        self.l2 = nn.ReLU()
        self.l3 = nn.Linear(20, 2)
        self.l4 = nn.Sigmoid()

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        x4 = self.l4(x3)
        return x4


model = XOR_net()
model = model.to(device)
loss_fun = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

epochs = 5000

for i in range(epochs):
    y_pre = model.forward(x)

    loss = loss_fun(y_pre, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 1000 == 0:
        print(f'Epochs : {i}')
        print(f'Loss : {loss}')

path = 'ex1_model.pth'
torch.save(model, path)
out = model(x)

print(out)
print(y)
