from Ex_4.dataset.dataset import Dogcat_Dataset
from Ex_4.model import DogCat_Net
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_loop(dataloader, model, loss_fn, optimizer,use_gpu=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    size = len(dataloader.dataset)
    for batch, X, y in enumerate(dataloader):
        # Compute prediction and loss
        if use_gpu == False:
            pred = model(X)
            loss = loss_fn(pred, y)
        else:
            X.to(device)
            y.to(device)
            model.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, use_gpu=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            if use_gpu == True:
                X.to(device)
                y.to(device)
                model.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")





learning_rate = 1e-3
batch_size = 10
epochs = 5

model = DogCat_Net()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

training_data = Dogcat_Dataset(train_dir='dataset/train_set', test_dir='dataset/test_set')
test_data = Dogcat_Dataset(train_dir='dataset/train_set', test_dir='dataset/test_set', train=False)

train_dataloader = DataLoader(training_data, batch_size=10, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=10, shuffle=True)

for batch, (X, y) in enumerate(train_dataloader):
    print(X.size(),y.size())
    print(X.type(),y.type())
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")