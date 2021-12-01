#from Python
import time
import csv
import os
import math
import numpy as np
import sys
from shutil import copyfile
import shutil

#from Pytorch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.utils as torch_utils
from torch.optim.lr_scheduler import StepLR

#from this project
from data_loader import get_loader
import data_loader as dl
import VisionOP
import model
import param as p
import utils 
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='./data/test', help="path to the saved checkpoint of model")
parser.add_argument('--output_dir', default='./data/results', help="path to the saved checkpoint of model")
args = parser.parse_args()

#local function
def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def norm(x):
    out = (x - 0.5) * 2
    return out.clamp(-1,1)

################ Hyper Parameters ################
# VERSION
version = '2019-12-19(LPGnet-with-LRblock)'
subversion = '1_1'

# data Set
dataSetName = p.dataSetName
dataSetMode = p.dataSetMode
dataPath = p.dataPath

maxDataNum = p.maxDataNum #in fact, 4500
batchSize = p.batchSize

MaxCropWidth = p.MaxCropWidth
MinCropWidth = p.MinCropWidth
MaxCropHeight = p.MaxCropHeight
MinCropHeight = p.MinCropHeight

# model
NOF = p.NOF

# train
MaxEpoch = p.MaxEpoch
learningRate = p.learningRate

# save
numberSaveImage = p.numberSaveImage

###########################################

torch.backends.cudnn.benchmark = True

# system setting


#init model
Retinex = model.LMSN()
Retinex = nn.DataParallel(Retinex).cuda()  
#model load
checkpoint_rt = torch.load('./data/model/Retinex' + '.pkl')
Retinex.load_state_dict(checkpoint_rt['model'])
dataSetMode = 'test'


for file in tqdm(os.listdir(args.input_dir)):
    file_path = os.path.join(args.input_dir, file)

    shutil.rmtree('./data/test/input')  
    os.mkdir('./data/test/input')  
    shutil.copy(file_path, os.path.join('./data/test/input', file))
    dataPath = './data/test/'
    data_loader = get_loader(dataPath,MaxCropWidth,MinCropWidth,MaxCropHeight,MinCropHeight,batchSize,dataSetName,dataSetMode)


    for epoch in range(0, 1):

        # ============= Train Retinex & Adjust module =============#

        torch.set_grad_enabled(False)

        j=0
        avg_in = 0
        avg_out = 0
        for i, (images) in enumerate(data_loader):
            b,c,h,w_ = images.size()
            w = int(w_/2)
            if i == 0:
                total_time = 0
    
            with torch.no_grad():
                torch.cuda.synchronize()
                Input = to_var(images).contiguous()
                if i >= 0:
                    a = time.perf_counter()

                    Scale1,Scale2,Scale3,res2,res3 = Retinex(Input)

                    olda = a
                    a = time.perf_counter()

                    total_time = total_time + a - olda


                    print('%d/500, time: %.5f sec ' % ((j+1),total_time / (j+1)), end="\n")
                    j=j+1
                else:
                    Scale1,Scale2,Scale3,res2,res3 = Retinex(Input)


                save_image(Scale3.data, os.path.join(args.output_dir,file))


