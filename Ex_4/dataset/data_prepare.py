"""
Data Pre-processing for Cat and Dog Dataset
2021/04/14 MYZ
"""
import cv2
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
    img = cv2.resize(img, (256, 256))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

mkdir('cat/cat')
mkdir('dog/dog')
for i in range(100):
    cat_name = 'cat.' + str(i) + '.jpg'
    cat_path = 'cat/' + cat_name
    dog_name = 'dog.' + str(i) + '.jpg'
    dog_path = 'dog/' + dog_name
    try:
        if cat_name in cat_list and dog_name in dog_list:
            cv2.imwrite('cat/cat/' + cat_name, preprocess_img(cat_path))
            cv2.imwrite('dog/dog/' + dog_name, preprocess_img(dog_path))
    except:
        print(f'Check the name of your dataset')


# Creating Dataset
# 这里就 140x256x256x3
# train_set = np.zeros(train_img_num * img_x * img_y)
# train_set = np.reshape(train_set, (train_img_num, img_x, img_y, 3))
# test_set = np.zeros(test_img_num * img_x * img_y)
# test_set = np.reshape(test_set, (test_img_num, img_x, img_y, 3))




