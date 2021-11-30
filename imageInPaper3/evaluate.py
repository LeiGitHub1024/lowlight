import numpy as np
import matplotlib.image as mpimg # mpimg 用于读取图片
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error, normalized_root_mse
from sklearn.metrics import mean_absolute_error
import os
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--groundtruth', '-g',default='/home/mist/low-light/train_datasets/lol_800/Real_captured/test/groundtruth')
parser.add_argument('--res', '-r',default='/home/mist/low-light/imageInPaper/source/lol-v2/res/')
args = parser.parse_args()

# python evaluate.py -g './source/ablation1/groundtruth' -r './source/ablation1/res/'

im_num = 0
for file in os.listdir(args.groundtruth):
  im_num = im_num + 1

psnrDict = {}
ssimDict = {}
MSE = {}
NRMSE = {}
MAE = {}

for dirname in os.listdir(args.res):

  psnrDict[dirname]=0
  ssimDict[dirname]=0
  MSE[dirname]=0
  NRMSE[dirname]=0
  MAE[dirname]=0

for file in tqdm(os.listdir(args.groundtruth)):
  for dirname in psnrDict.keys():
    im1 = mpimg.imread(os.path.join(args.groundtruth, file))
    im2 = mpimg.imread(os.path.join(args.res+dirname,file))
    # print(dirname,im1.shape,im2.shape)
    # continue
    
    psnrDict[dirname] = psnrDict[dirname] + peak_signal_noise_ratio(im1, im2)/im_num
    ssimDict[dirname] = ssimDict[dirname] + structural_similarity(im1, im2,multichannel=True)/im_num
    MSE[dirname] = MSE[dirname] + mean_squared_error(im1, im2)/im_num
    NRMSE[dirname] = NRMSE[dirname] + normalized_root_mse(im1, im2)/im_num
    MAE[dirname] = MAE[dirname] + np.mean(np.abs(im1-im2))/im_num


    
for dirname in os.listdir(args.res):
  print(dirname)
  print('psnr: ',"%.3f"%psnrDict[dirname])
  print('ssim: ',"%.3f"%ssimDict[dirname])
  print('mse: ',"%.3f"%MSE[dirname])
  print('nrmse: ',"%.3f"%NRMSE[dirname])
  # print('mae: ',"%.3f"%MAE[dirname])
  print('\n')

  
# print('psnr: ',psnrDict)
# print('ssim: ',ssimDict)
# print('mse: ',MSE)
# print('nrmse: ',NRMSE)