from glob import glob
from tqdm import tqdm
import numpy as np
import os
from natsort import natsorted
import cv2
from joblib import Parallel, delayed
import multiprocessing
import argparse
from skimage.transform import resize

def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])

parser = argparse.ArgumentParser(description='Generate patches from Full Resolution images')
parser.add_argument('--src_dir', default='/home/mist/low-light/train_datasets/lol_800/Real_captured/train', type=str, help='Directory for full resolution images')
parser.add_argument('--tar_dir', default='/home/mist/low-light/train_datasets/lol_800/train_dataset/train',type=str, help='Directory for image patches')
parser.add_argument('--ps', default=256, type=int, help='Image Patch Size')
parser.add_argument('--num_patches', default=2, type=int, help='Number of patches per image')
parser.add_argument('--num_cores', default=10, type=int, help='Number of CPU Cores')

args = parser.parse_args()

src = args.src_dir
tar = args.tar_dir
PS = args.ps
NUM_PATCHES = args.num_patches
NUM_CORES = args.num_cores

low_patchDir = os.path.join(tar, 'input')
high_patchDir = os.path.join(tar, 'groundtruth')

if os.path.exists(tar):
    os.system("rm -r {}".format(tar))

os.makedirs(low_patchDir)
os.makedirs(high_patchDir)

low_files = sorted(os.listdir(os.path.join(src, 'input')))
high_files = sorted(os.listdir(os.path.join(src, 'groundtruth')))

low_filenames = [os.path.join(src, 'input', x) for x in low_files if is_png_file(x)]
high_filenames = [os.path.join(src, 'groundtruth', x) for x in high_files if is_png_file(x)]

# print(low_filenames)
# print(high_filenames)

def Generate_patches(i):
    low_file, high_file = low_filenames[i], high_filenames[i]
    low_img = cv2.imread(low_file)
    high_img = cv2.imread(high_file)

    H = low_img.shape[0]
    W = low_img.shape[1]
    cnt = 0
    save_name = os.path.split(low_file)[-1]
    save_name = save_name.split('.')[0]  

    n = H // PS
    m = W // PS
    len1 = H // n
    len2 = W // m
    for ii in range(n):
        for jj in range(m):
            rr = len1*ii+np.random.randint(0, len1-PS)
            cc = len2*jj+np.random.randint(0, len2-PS)
            low_patch = low_img[rr:rr+PS, cc:cc+PS, :]
            high_patch = high_img[rr:rr+PS, cc:cc+PS, :]
            cnt = cnt + 1
            cv2.imwrite(os.path.join(low_patchDir, '{}_{}.png'.format(save_name, cnt)), low_patch)
            cv2.imwrite(os.path.join(high_patchDir, '{}_{}.png'.format(save_name, cnt)),  high_patch)
    
    #low_img = low_img.transpose((1,2,0))
    #high_img = high_img.transpose((1,2,0))
    low_patch = cv2.resize(low_img, (PS, PS))
    high_patch = cv2.resize(high_img, (PS, PS))
    #low_patch = low_patch.transpose((2,0,1))
    #high_patch = high_patch.transpose((2,0,1))
    cnt = cnt + 1
    cv2.imwrite(os.path.join(low_patchDir, '{}_{}.png'.format(save_name, cnt)), low_patch)
    cv2.imwrite(os.path.join(high_patchDir, '{}_{}.png'.format(save_name, cnt)),  high_patch)
    # for j in range(NUM_PATCHES):
    #     rr = np.random.randint(0, H-PS)
    #     cc = np.random.randint(0, W-PS)
    #     low_patch = low_img[rr:rr+PS, cc:cc+PS, :]
    #     high_patch = high_img[rr:rr+PS, cc:cc+PS, :]

    #     save_name = os.path.split(low_file)[-1]    
    #     cv2.imwrite(os.path.join(low_patchDir, '{}_{}'.format(j+1, save_name)), low_patch)
    #     cv2.imwrite(os.path.join(high_patchDir, '{}_{}'.format(j+1, save_name)),  high_patch)
    

Parallel(n_jobs=NUM_CORES)(delayed(Generate_patches)(i) for i in tqdm(range(len(low_filenames))))
