import torch
import numpy as np
import matplotlib.pyplot as plt
from Ex_2.data_regression import DR_net


# Using GPU to calculate
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = DR_net()
x = torch.unsqueeze(torch.linspace(-np.pi, np.pi, 100), dim=1)
x_print = x.numpy()
x = x.to(device)

path = 'Ex2_model.pth'

model.load_state_dict(torch.load(path))
model.to(device)
model.eval()


out = model(x)
y_print = out.to('cpu').detach().numpy()
plt.plot(x_print, y_print,'r')
plt.show()