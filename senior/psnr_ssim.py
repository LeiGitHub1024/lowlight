import numpy as np
import matplotlib.image as mpimg # mpimg 用于读取图片
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--groundtruth', '-g',default='./test')
parser.add_argument('--res', '-r',default='./res/')
args = parser.parse_args()



im_num = 0
for file in os.listdir(args.groundtruth):
  im_num = im_num + 1

psnrDict = {}
ssimDict = {}
for dirname in os.listdir(args.res):
  psnrDict[dirname]=0
  ssimDict[dirname]=0

for file in os.listdir(args.groundtruth):
  for dirname in psnrDict.keys():
    im1 = mpimg.imread(os.path.join(args.groundtruth, file))
    im2 = mpimg.imread(os.path.join(args.res+dirname,file))
    
    psnrDict[dirname] = psnrDict[dirname] + peak_signal_noise_ratio(im1, im2)/im_num
    ssimDict[dirname] = ssimDict[dirname] + structural_similarity(im1, im2,multichannel=True)/im_num


  
print(psnrDict)
print(ssimDict)