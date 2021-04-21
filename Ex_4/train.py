from Ex_4.dataset.dataset import Dogcat_Dataset
from Ex_4.model import DogCat_Net
import torch
from torch.utils.data import DataLoader

training_data = Dogcat_Dataset(train_dir='dataset/train_set', test_dir='dataset/test_set')
test_data = Dogcat_Dataset(train_dir='dataset/train_set', test_dir='dataset/test_set', train=False)

train_dataloader = DataLoader(training_data, batch_size=10, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=10, shuffle=True)

sample_idx = torch.randint(len(training_data), size=(1,)).item()
img = training_data[sample_idx]['image']
label = training_data[sample_idx]['label']