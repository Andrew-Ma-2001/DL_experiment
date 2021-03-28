import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class DR_net(nn.Module):
    def __init__(self):
        super(DR_net, self).__init__()
        self.l1 = nn.Linear(1, 10)
        self.l2 = nn.ReLU()
        self.l3 = nn.Linear(10, 1)

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        return x3


# Create Dataset
x = torch.unsqueeze(torch.linspace(-np.pi, np.pi, 100), dim=1)
y = torch.sin(x) + 0.5 * torch.rand(x.size())

# Using GPU to calculate
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
x = x.to(device)
y = y.to(device)
model = DR_net()
model = model.to(device)

# Setting loss & optimizer
loss_fun = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Start training
epochs = 2000
for i in range(epochs):
    y_pre = model.forward(x)

    loss = loss_fun(y_pre, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 500 == 0:
        print(f'Epochs : {i}')
        print(f'Loss : {loss}')

path = 'Ex2_model.pth'
torch.save(model.state_dict(), path)
out = model(x)
