import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
# from torchvision.io import read_image
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

loader = transforms.Compose([transforms.ToTensor()])
def PIL_to_tensor(image):
    image = loader(image).unsqueeze(0)
    return image.to('cpu', torch.float)

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
            image = Image.open(path)
            image = PIL_to_tensor(image)
            image = torch.squeeze(image)
            label = self.find_label(path)

            return image,label
        else:
            path = file_name(self.test_dir)[item]
            image = Image.open(path)
            image = PIL_to_tensor(image)
            image = torch.squeeze(image)
            label = self.find_label(path)
            sample = [image, label]
            return sample


    def __len__(self):
        if self.train:
            return(len(file_name(self.train_dir)))
        else:
            return(len(file_name(self.test_dir)))

    # Here we consider cat as 0 dog as 1
    def find_label(self, name):
        if name.count('dog') != 0:
            return torch.tensor([1])
        if name.count('cat') != 0:
            return torch.tensor([0])



if __name__ == '__main__':
    labels_map = { 0: 'cat', 1: 'dog',}

    training_data = Dogcat_Dataset(train_dir='train_set', test_dir='test_set')
    test_data = Dogcat_Dataset(train_dir='train_set', test_dir='test_set', train=False)


    train_dataloader = DataLoader(training_data, batch_size=10, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=10, shuffle=True)

    # Testing
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]

    # numpy image: H x W x C
    # torch image: C x H x W
    # np.transpose( xxx,  (2, 0, 1))   # 将 H x W x C 转化为 C x H x W
    # img = torch.squeeze(img)
    img = np.transpose((img.numpy()), (1, 2, 0))
    plt.imshow(img)
    plt.title(label)
    plt.show()

    train_features, train_labels = next(iter(train_dataloader))
    print(train_features.type())






