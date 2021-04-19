"""
Data Pre-processing for Cat and Dog Dataset
2021/04/14 MYZ
"""
import cv2
import numpy as np
import os

# Dataset URl
url = 'https://www.kaggle.com/tongpython/cat-and-dog'

# Parameters
img_x = 256
img_y = 256
train_img_num = 140
test_img_num = 60

# Reading Images
cat_list = os.listdir('cat')
dog_list = os.listdir('dog')
cat_img = []
dog_img = []


def preprocess_img(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


for i in range(100):
    cat_name = 'cat.' + str(i) + '.jpg'
    cat_path = 'cat/' + cat_name
    dog_name = 'dog.' + str(i) + '.jpg'
    dog_path = 'dog/' + dog_name
    try:
        if cat_name in cat_list and dog_name in dog_list:
            cat_img.append(preprocess_img(cat_path))
            dog_img.append(preprocess_img(dog_path))
    except:
        print(f'Check the name of your dataset')

# Creating Dataset
# 这里就 140x256x256x3
train_set = np.zeros(train_img_num * img_x * img_y)
train_set = np.reshape(train_set, (train_img_num, img_x, img_y, 3))
test_set = np.zeros(test_img_num * img_x * img_y)
test_set = np.reshape(test_set, (test_img_num, img_x, img_y, 3))
