import torch
import numpy as np
import matplotlib.pyplot as plt
from Ex_3.train import DC_net

# Using GPU to calculate
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = DC_net()
path = 'Ex3_model.pth'

# Creating dataset
data = torch.ones(100, 2)
x0 = torch.normal(2 * data, 1)
x1 = torch.normal(-2 * data, 1)
x = torch.cat((x0, x1))
x_print = x
y0 = torch.zeros(100)
y1 = torch.ones(100)
y = torch.cat((y0, y1)).type(dtype=torch.LongTensor)
y_print = y
y_label = []

x = x.to(device)
y = y.to(device)

model.load_state_dict(torch.load(path))
model.to(device)
model.eval()
y_pre = model(x)

y_label = y_pre.to('cpu').detach()
y_prediction = (torch.max(y_label, 1)[1]).numpy()
target_y = y_print.numpy()
plt.scatter(x_print.numpy()[:, 0], x_print.numpy()[:, 1], c=y_prediction, s=100, cmap='RdYlGn')
acc = sum(y_prediction == target_y) / 200
plt.text(1.5, -4, f'Accuracy is {acc}', fontdict={'size': 20, 'color': 'red'})
plt.show()