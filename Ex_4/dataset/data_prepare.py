"""
Data Pre-processing for Cat and Dog Dataset
2021/04/14 MYZ
"""
import cv2
import os

# Dataset URl
url = 'https://www.kaggle.com/tongpython/cat-and-dog'

# Parameters
img_x = 128
img_y = 128
train_img_num = 140
test_img_num = 60

# Reading Images
cat_list = os.listdir('cat')
dog_list = os.listdir('dog')
cat_img = []
dog_img = []


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


def file_name(file_dir):
    list = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                list.append(os.path.join(root, file))
    return list


cat_dir = 'cat'
dog_dir = 'dog'
val_dir = 'validation_set'

cat_path = file_name(cat_dir)
dog_path = file_name(dog_dir)
val_path = file_name(val_dir)


def preprocess_img(list):
    img = cv2.imread(list)
    img = cv2.resize(img, (img_x, img_y))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


mkdir('train_set')
mkdir('test_set')
for i in range(70):  # 70-100
    cat_name = 'cat.' + str(i) + '.jpg'
    cat_path = 'cat/' + cat_name
    dog_name = 'dog.' + str(i) + '.jpg'
    dog_path = 'dog/' + dog_name
    try:
        if cat_name in cat_list and dog_name in dog_list:
            cv2.imwrite('train_set/' + cat_name, preprocess_img(cat_path))
            cv2.imwrite('train_set/' + dog_name, preprocess_img(dog_path))
    except:
        print(f'Check the name of your dataset')

for i in range(70, 100):  # 70-100
    cat_name = 'cat.' + str(i) + '.jpg'
    cat_path = 'cat/' + cat_name
    dog_name = 'dog.' + str(i) + '.jpg'
    dog_path = 'dog/' + dog_name
    try:
        if cat_name in cat_list and dog_name in dog_list:
            cv2.imwrite('test_set/' + cat_name, preprocess_img(cat_path))
            cv2.imwrite('test_set/' + dog_name, preprocess_img(dog_path))
    except:
        print(f'Check the name of your dataset')
