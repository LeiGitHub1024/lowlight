import argparse
import numpy as np
import torch
import torch.nn as nn
import cv2
from model import UNet
from glob import glob
import os
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--path', default='./checkpoint.pth', help="path to the saved checkpoint of model")
parser.add_argument('--input_dir', default='./data/test', help="path to the saved checkpoint of model")
parser.add_argument('--output_dir', default='./data/results', help="path to the saved checkpoint of model")
args = parser.parse_args()

filenames = glob('./data/test/*')
filenames.sort()

model = UNet(n_channels=3, bilinear=True)
model.load_state_dict(torch.load(args.path))
model.to('cuda')

with torch.no_grad():
    for filename in tqdm(os.listdir(args.input_dir)):
        test = cv2.imread(os.path.join(args.input_dir, filename))/255.0        
        test = np.expand_dims(test.transpose([2,0,1]), axis=0)
        test = torch.from_numpy(test).to(device="cuda", dtype=torch.float32)

        out = model(test)

        out = out.to(device="cpu").numpy().squeeze()
        out = np.clip(out*255.0, 0, 255)

        path = os.path.join(args.output_dir, filename)
        cv2.imwrite(path, out.astype(np.uint8).transpose([1,2,0]))
