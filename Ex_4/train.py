from Ex_4.dataset.dataset import Dogcat_Dataset
from Ex_4.model import DogCat_Net
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_loop(dataloader, model, loss_fn, optimizer,use_gpu=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.type(torch.FloatTensor)
        y = y.type(torch.LongTensor)
        y = torch.squeeze(y)
        if use_gpu == False:
            pred = model(X)
            loss = loss_fn(pred, y)
        else:
            X = X.to(device)
            y = y.to(device)
            model.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        loss, current = loss.item(), (batch+1) * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


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




if __name__ == '__main__':
    learning_rate = 1e-4
    batch_size = 10
    epochs = 50

    model = DogCat_Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    training_data = Dogcat_Dataset(train_dir='dataset/train_set', test_dir='dataset/test_set')
    test_data = Dogcat_Dataset(train_dir='dataset/train_set', test_dir='dataset/test_set', train=False)

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    torch.cuda.empty_cache()

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer,use_gpu=True)
        test_loop(test_dataloader, model, loss_fn,use_gpu=True)
    print("Done!")