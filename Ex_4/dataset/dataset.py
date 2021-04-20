import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import matplotlib.pyplot as plt

def file_name(file_dir):
    list = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                list.append(os.path.join(root, file))
    return list

class Dogcat_Dataset(Dataset):
    def __init__(self, train_dir, test_dir,transform=True, train=True):
        self.transform = transform
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.train = train


    def __getitem__(self, item):
        if self.train == True:
            path = file_name(self.train_dir)[item]
            image = read_image(path)
            label = self.find_label(path)
            sample = {"image": image, "label": label}
            return sample
        else:
            path = file_name(self.test_dir)[item]
            image = read_image(path)
            label = self.find_label(path)
            sample = {"image": image, "label": label}
            return sample


    def __len__(self):
        if self.train:
            return(len(file_name(self.train_dir)))
        else:
            return(len(file_name(self.test_dir)))

    # Here we consider cat as 0 dog as 1
    def find_label(self, name):
        if name.count('dog') != 0:
            return 1
        if name.count('cat') != 0:
            return 0



if __name__ == '__main__':
    training_data = Dogcat_Dataset(train_dir='train_set', test_dir='test_set')
    test_data = Dogcat_Dataset(train_dir='train_set', test_dir='test_set', train=False)


    train_dataloader = DataLoader(training_data, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=16, shuffle=True)


    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]






    # Display image and label.
    # train_features, train_labels = next(iter(train_dataloader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")
    # img = train_features[0].squeeze()
    # label = train_labels[0]
    # plt.imshow(img, cmap="gray")
    # plt.show()
    # print(f"Label: {label}")