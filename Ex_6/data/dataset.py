import torchvision as tv
import torchvision.transforms as transforms
import torch
from torchvision.transforms import ToPILImage

show = ToPILImage()
import matplotlib.pyplot as plt
import torchvision
import numpy as np


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
                                ])
# 训练集
train_set = tv.datasets.CIFAR10(root='./data',
                               train=True,
                               download=True,
                               transform=transform)

train_loader = torch.utils.data.DataLoader(train_set,
                                      batch_size=16,
                                      shuffle=True)
# 测试集
test_set = tv.datasets.CIFAR10(root='./data',
                              train=False,
                              download=True,
                              transform=transform)

test_loader = torch.utils.data.DataLoader(test_set,
                                     batch_size=16,
                                     shuffle=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5     # 将正则化后图片恢复
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 随取抽取训练集中的图片
images, labels = next(iter(train_loader))

# 显示图片
imshow(torchvision.utils.make_grid(images))
# 输出标签
print(' '.join('%5s' % classes[labels[j]] for j in range(16)))