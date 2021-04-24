import torch
import numpy as np
import matplotlib.pyplot as plt
from Ex_4.model import DogCat_Net
from Ex_4.dataset.dataset import Dogcat_Dataset
from torch.utils.data import DataLoader

def test_loop(dataloader, model, loss_fn, use_gpu=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.type(torch.FloatTensor)
            y = y.type(torch.LongTensor)
            y = torch.squeeze(y)
            if use_gpu == True:
                X = X.to(device)
                y = y.to(device)
                model.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct





batch_size = 10

model = DogCat_Net()
path = 'model/26_best_model.pth'
model.load_state_dict(torch.load(path))
model.eval()

training_data = Dogcat_Dataset(train_dir='dataset/train_set', test_dir='dataset/test_set')
test_data = Dogcat_Dataset(train_dir='dataset/train_set', test_dir='dataset/test_set', train=False)

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

if __name__ == '__main__':
    test_loop(test_dataloader,model,loss_fn=torch.nn.CrossEntropyLoss())
