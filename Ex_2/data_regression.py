'''
Using Pytorch to do Data Regression
2021/03/29 MYZ
'''
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


if __name__ == '__main__':
    # Create Dataset
    x = torch.unsqueeze(torch.linspace(-np.pi, np.pi, 100), dim=1)
    y = torch.sin(x) + 0.5 * torch.rand(x.size())
    y_train = torch.sin(x)
    # Create Image
    plot_x = torch.squeeze(x.to('cpu')).numpy()
    plot_y_train = torch.squeeze(y.to('cpu')).numpy()

    # Using GPU to calculate
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    y = y.to(device)
    model = DR_net()
    model = model.to(device)

    # Setting loss & optimizer
    loss_fun = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Start training
    epochs = 2000
    for i in range(epochs):
        y_pre = model.forward(x)

        loss = loss_fun(y_pre, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 250 == 0:
            print(f'Epochs : {i}')
            print(f'Loss : {loss}')
            plot_y_pre = (torch.squeeze(y_pre.to('cpu'))).detach().numpy()
            plt.ion()
            plt.clf()
            plt.scatter(plot_x, plot_y_train)
            plt.plot(plot_x, plot_y_pre, 'r', linewidth=3)
            plt.pause(0.5)
            plt.ioff()

    path = 'Ex2_model.pth'
    torch.save(model.state_dict(), path)
    out = model(x)

    # todo matplotlib 制图换个颜色

