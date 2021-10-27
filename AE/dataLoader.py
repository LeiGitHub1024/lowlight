import numpy as np
from sklearn.model_selection import train_test_split
from skimage.util import random_noise
import torch
import torch.utils.data as Data 
from tqdm import tqdm
import os
from PIL import Image

# 首先，从上述链接中下载完STL10的数据集后，开始对数据做预处理。STL10中都是96*96的RGB图片，训练集放在了train_X.bin的文件中，可以用做自编码器的无监督学习，下面是第数据的预处理：
def read_image(data_path):
    with open(data_path, "rb") as f:
        data1 = np.fromfile(f, dtype=np.uint8)
        # 塑形成[batch, c, h, w]
        images = np.reshape(data1, [-1, 3, 96, 96])
        # 图像转化为RGB(即最后一个维度是通道维度)的形式，方便使用matplotlib进行可视化
        images = np.transpose(images, [0, 3, 2, 1])
    return images / 255
def gaussian_gamma(images, sigma):
    """sigma: 噪声标准差, gamma: gamma矫正"""
    sigma2 = sigma**2 / (255 ** 2)   # 噪声方差
    res = np.zeros_like(images)
    for ii in tqdm(range(images.shape[0])):
        image = images[ii]
        # 使用skimage中的函数增加噪音,gamma,在1.5-4.5之间
        gamma = 3 #random.uniform(1-5)
        noise_im = random_noise(image, mode="gaussian", var=sigma2, clip=True)
        # 使用gamma来让图像变暗
        gamma_noise_im = np.power(noise_im, gamma)

        res[ii] = gamma_noise_im
    return res
def stl10_loader(device):
  data_path = "../data/stl10_binary/train_X.bin"
  images = read_image(data_path)
  images.shape

  # 下面定义一个函数，为干净的图片添加高斯噪音，这部分添加了噪音的数据，将成为自编码器的输入。其中的random_noise是属于skimage.util下的一个方法。
  images_noise = gaussian_gamma(images, 25)

  # 数据与处理完后，老规矩，构建loader，以便于训练时直接拿出一个batch的数据出来训练。
  # 首先将数据集切分为训练集和验证集，并转换为torch张量。

  # 数据集准备为PyTorch可用的形式，转化为[样本, 通道, 高, 宽]
  data_X = np.transpose(images_noise, (0, 3, 2, 1))
  data_Y = np.transpose(images, (0, 3, 2, 1))

  X_train, X_val, y_train, y_val = train_test_split(data_X, data_Y, test_size=0.2, random_state=123)
  # 转化为torch张量
  X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
  y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
  X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
  y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
  # 将X与y整合在一起
  train_data = Data.TensorDataset(X_train, y_train)
  val_data = Data.TensorDataset(X_val, y_val)
  return train_data,val_data



#读取mbllen的pairdata，整理成需要的格式，
def mbllen_npy():
    # image => npy
    # mbllenPath = '../datasets/mbllen/train'
    # mbllenLowPath = '../datasets/mbllen/train_lowlight'
    # train = []
    # train_lowlight = []
    # i=500
    # for file in tqdm(os.listdir(mbllenPath)):
    #     if(i==0):
    #         break
    #     i =i-1
    #     train_path = os.path.join(mbllenPath, file)
    #     if os.path.isfile(train_path) == True:
    #         im = np.array(Image.open(train_path))
    #         train.append(im)
    #     lowlight_path = os.path.join(mbllenLowPath, file)
    #     if os.path.isfile(lowlight_path) == True:
    #         im = np.array(Image.open(lowlight_path))
    #         train_lowlight.append(im)
    # np.save("../data/mb_train.npy", train) 
    # np.save("../data/mb_train_lowlight.npy", train_lowlight) 

    #load npy
    image_train = np.load('../data/mb_train.npy')
    image_train = np.reshape(image_train, [-1, 3, 256, 256])
    image_train = np.transpose(image_train, [0, 3, 2, 1])

    image_lowlight = np.load('../data/mb_train_lowlight.npy') 
    image_lowlight = np.reshape(image_lowlight, [-1, 3, 256, 256])   
    image_lowlight = np.transpose(image_lowlight, [0, 3, 2, 1])
    return image_train/255, image_lowlight/255

def mbllen_loader(device):
    train,train_lowlight = mbllen_npy()
    # 数据集准备为PyTorch可用的形式，转化为[样本, 通道, 高, 宽]
    data_X = np.transpose(train_lowlight, (0, 3, 2, 1))
    data_Y = np.transpose(train, (0, 3, 2, 1))

    X_train, X_val, y_train, y_val = train_test_split(data_X, data_Y, test_size=0.2, random_state=123)
    # 转化为torch张量
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
    # 将X与y整合在一起
    train_data = Data.TensorDataset(X_train, y_train)
    val_data = Data.TensorDataset(X_val, y_val)

    print("X_train.shape", X_train.shape)
    print("y_train.shape:", y_train.shape)
    print("X_val.shape:", X_val.shape)
    print("y_val.shape:", y_val.shape)
    return train_data,val_data



def lol_loader(device):
    # # image => npy
    # mbllenPath = '../datasets/lol/train/groundtruth'
    # mbllenLowPath = '../datasets/lol/train/input'
    # train = []
    # train_lowlight = []
    # # i=5000
    # for file in tqdm(os.listdir(mbllenPath)):
    #     # if(i==0):
    #     #     break
    #     # i =i-1
    #     train_path = os.path.join(mbllenPath, file)
    #     if os.path.isfile(train_path) == True:
    #         im = np.array(Image.open(train_path))
    #         train.append(im)
    #     lowlight_path = os.path.join(mbllenLowPath, file)
    #     if os.path.isfile(lowlight_path) == True:
    #         im = np.array(Image.open(lowlight_path))
    #         train_lowlight.append(im)
    # np.save("../data/lol_train.npy", train) 
    # np.save("../data/lol_train_lowlight.npy", train_lowlight) 

    #load npy
    image_train = np.load('../data/lol_train.npy')
    image_train = np.reshape(image_train, [-1, 3, 256, 256])
    image_train = np.transpose(image_train, [0, 3, 2, 1])

    image_lowlight = np.load('../data/lol_train_lowlight.npy') 
    image_lowlight = np.reshape(image_lowlight, [-1, 3, 256, 256])   
    image_lowlight = np.transpose(image_lowlight, [0, 3, 2, 1])
    return image_train/255., image_lowlight/255.
        
