import torch
from Ex_1.train import XOR_net
# Using GPU to calculate
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

x = torch.Tensor([[0, 1]])
x = x.to(device)


path = 'Ex1_model.pth'
model = XOR_net()
model.load_state_dict(torch.load(path))
model.to(device)
model.eval()
out = model(x)
print(out)