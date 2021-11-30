from glob import glob
from tqdm import tqdm
import numpy as np
import os
from natsort import natsorted
import cv2
from joblib import Parallel, delayed
import multiprocessing
import argparse

def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])

parser = argparse.ArgumentParser(description='Generate patches from Full Resolution images')
parser.add_argument('--src_dir', default='../datasets/our485', type=str, help='Directory for full resolution images')
parser.add_argument('--tar_dir', default='../datasets/our485_2patches',type=str, help='Directory for image patches')
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
    for j in range(NUM_PATCHES):
        rr = np.random.randint(0, H-PS)
        cc = np.random.randint(0, W-PS)
        low_patch = low_img[rr:rr+PS, cc:cc+PS, :]
        high_patch = high_img[rr:rr+PS, cc:cc+PS, :]

        save_name = os.path.split(low_file)[-1]    
        cv2.imwrite(os.path.join(low_patchDir, '{}_{}'.format(j+1, save_name)), low_patch)
        cv2.imwrite(os.path.join(high_patchDir, '{}_{}'.format(j+1, save_name)),  high_patch)
    

Parallel(n_jobs=NUM_CORES)(delayed(Generate_patches)(i) for i in tqdm(range(len(low_filenames))))
