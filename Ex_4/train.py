from Ex_4.dataset.dataset import Dogcat_Dataset
from Ex_4.model import DogCat_Net
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

def mkdir(path):
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        print(path + ' 目录已存在')
        return False

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

def train_loop(train_dataloader,model, loss_fn, optimizer,use_gpu=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    size = len(train_dataloader.dataset)
    for batch, (X, y) in enumerate(train_dataloader):
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








if __name__ == '__main__':
    learning_rate = 1e-4
    batch_size = 10
    epochs = 10
    best_acc = 0
    model = DogCat_Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    training_data = Dogcat_Dataset(train_dir='dataset/train_set', test_dir='dataset/test_set')
    test_data = Dogcat_Dataset(train_dir='dataset/train_set', test_dir='dataset/test_set', train=False)

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    torch.cuda.empty_cache()
    mkdir('model')
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer,use_gpu=True)
        acc = test_loop(test_dataloader, model, loss_fn,use_gpu=True)
        if float(acc) >= best_acc:
            best_acc = acc
            save_path = 'model/' + str(t) + '_best_model.pth'
            torch.save(model.state_dict(), save_path)
    print("Done!")