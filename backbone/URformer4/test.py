import numpy as np
import os,sys,math
import argparse
from tqdm import tqdm
from einops import rearrange, repeat

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from ptflops import get_model_complexity_info

sys.path.append('/home/wangzd/uformer/')

import scipy.io as sio
from utils.loader import get_validation_data
import utils

from model import UNet,Uformer,Uformer_Cross,Uformer_CatCross

from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from skimage.transform import resize


stage = 0


parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
parser.add_argument('--input_dir', default='/home/mist/lowlight/datasets/mit500/test',type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./log/Uformer32_1009_3_0/results_mit_mid/',type=str, help='Directory for results')
parser.add_argument('--weights', default='./log/Uformer32_1009_3_0/models/model_best.pth',type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--arch', default='Uformer', type=str, help='arch')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
parser.add_argument('--embed_dim', type=int, default=32, help='number of data loading workers')    
parser.add_argument('--win_size', type=int, default=8, help='number of data loading workers')
parser.add_argument('--token_projection', type=str,default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str,default='leff', help='ffn/leff token mlp')
# args for vit
parser.add_argument('--vit_dim', type=int, default=256, help='vit hidden_dim')
parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')
parser.add_argument('--global_skip', action='store_true', default=False, help='global skip connection')
parser.add_argument('--local_skip', action='store_true', default=False, help='local skip connection')
parser.add_argument('--vit_share', action='store_true', default=False, help='share vit module')

parser.add_argument('--train_ps', type=int, default=64, help='patch size of training sample')
args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

utils.mkdir(args.result_dir)

test_dataset = get_validation_data(args.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)

model_restoration= utils.get_arch(args)
model_restoration = torch.nn.DataParallel(model_restoration)

utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ", args.weights)

def tensorResize(tensor,X):
    #tensor->numpy
    # print(tensor.shape)
    np_arr = tensor.cpu().detach().numpy()
    np_arr = np_arr[0].transpose((1,2,0))

    #Image resize
    im_np_resize = resize(np_arr, (X, X))
    im_np_resize = im_np_resize.transpose((2,0,1))
    im_np_resize = im_np_resize[np.newaxis,:]
    # print(im_np_resize.shape)

    #numpy->tensor
    # print(im_np_resize.shape)
    return torch.from_numpy(im_np_resize)

model_restoration.cuda()
model_restoration.eval()
factor = 256.0
with torch.no_grad():
    psnr_val_rgb = []
    ssim_val_rgb = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        _, _, h, w = data_test[1].cuda().size()
        X = int(math.ceil(max(h,w)/float(factor))*factor)
        gt = tensorResize(data_test[0],X)
        input = tensorResize(data_test[1],X)
        
        rgb_gt = gt.numpy().squeeze().transpose((1,2,0))
        rgb_noisy = input.cuda()
        filenames = data_test[2]


        if stage == 1:
            tmp, rgb_restored = model_restoration(rgb_noisy, 1)
        else:
            rgb_restored = model_restoration(rgb_noisy, 0)
        #rgb_restored = tmp #*0.5 + rgb_restored*0.5
        rgb_restored = torch.clamp(rgb_restored,0,1).cpu().numpy().squeeze().transpose((1,2,0))
        psnr_val_rgb.append(psnr_loss(rgb_restored, rgb_gt))
        ssim_val_rgb.append(ssim_loss(rgb_restored, rgb_gt, multichannel=True))
        rgb_restored = resize(rgb_restored, (h, w))

        utils.save_img(os.path.join(args.result_dir,filenames[0]), img_as_ubyte(rgb_restored))

psnr_val_rgb = sum(psnr_val_rgb)/len(test_dataset)
ssim_val_rgb = sum(ssim_val_rgb)/len(test_dataset)
print("PSNR: %f, SSIM: %f " %(psnr_val_rgb,ssim_val_rgb))