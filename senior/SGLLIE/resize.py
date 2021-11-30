import argparse
from tqdm import tqdm
import os
from PIL import Image
import cv2 
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--gt', default='/home/mist/low-light/senior/test/input', help="path to the saved checkpoint of model")
parser.add_argument('--sg', default='/home/mist/low-light/senior/test/res/sgllie', help="path to the saved checkpoint of model")
args = parser.parse_args()


for file in tqdm(os.listdir(args.gt)):

  gt_path = os.path.join(args.gt, file)
  im_gt = Image.open(gt_path)
  (w,h) = im_gt.size
  print(w,h)
  st_path = os.path.join(args.sg, file)
  im_sg = Image.open(st_path)
  im_sg=im_sg.resize((w,h))

  im_sg.save(os.path.join(args.sg, file))



