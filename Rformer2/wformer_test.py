import numpy as np
import os,sys,math,shutil
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

from genR import copyGT,generateR
torch.cuda.empty_cache()
stage = 1
parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
parser.add_argument('--input_dir', default='/home/mist/low-light/imageInPaper/input',type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./test/res4',type=str, help='Directory for results')
parser.add_argument('--weights', default='./log/Wformer2.0_RD_250_1/models/model_best.pth',type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--arch', default='Wformer', type=str, help='arch')
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

parser.add_argument('--train_ps', type=int, default=256, help='patch size of training sample')
parser.add_argument('--unpaired', default=False, help='test set is unpaired')


args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

utils.mkdir(args.result_dir)

tmp_dir = './wformer_r'
if args.unpaired== False : # 对于paired data
    copyGT(os.path.join(args.input_dir, 'groundtruth'), tmp_dir+'/groundtruth')
    generateR(os.path.join(args.input_dir, 'input'), tmp_dir+"/input")
else :#unpaired
    copyGT(args.input_dir, tmp_dir+'/groundtruth')
    generateR(args.input_dir, tmp_dir+"/input")



test_dataset = get_validation_data(tmp_dir)
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
    return h, w, torch.from_numpy(im_np_resize)

def recover(img, h, w):
    np_arr = img.cpu().detach().numpy()
    np_arr = np_arr[0].transpose((1,2,0))
    im_np_resize = resize(np_arr, (h, w))
    im_np_resize = im_np_resize.transpose((2,0,1))
    im_np_resize = im_np_resize[np.newaxis,:]
    return torch.from_numpy(im_np_resize).cuda()

model_restoration.cuda()
model_restoration.eval()
factor = 128
with torch.no_grad():
    psnr_val_rgb = []
    ssim_val_rgb = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        _, _, h, w = data_test[1].cuda().size()
        X = int(math.ceil(max(h,w)/float(factor))*factor)
        gt = data_test[0]
        test_h, test_w, input = tensorResize(data_test[1],X)
        
        rgb_gt = gt.numpy().squeeze().transpose((1,2,0))
        rgb_noisy = input.cuda()
        filenames = data_test[2]
        # print(filenames)

        if stage == 0:
            rgb_restored = model_restoration(rgb_noisy, stage)
        else:
            _, rgb_restored = model_restoration(rgb_noisy, stage)
        
        rgb_restored = torch.clamp(rgb_restored,0,1)
        rgb_restored = recover(rgb_restored, test_h, test_w)
        rgb_restored = rgb_restored.cpu().numpy().squeeze().transpose((1,2,0))
        #rgb_restored = resize(rgb_restored, (h, w))
        # psnr_val_rgb.append(utils.batch_PSNR(rgb_restored, gt.cuda(), False).item())
        psnr_val_rgb.append(psnr_loss(rgb_restored, rgb_gt))
        ssim_val_rgb.append(ssim_loss(rgb_restored, rgb_gt, multichannel=True))
        

        utils.save_img(os.path.join(args.result_dir,filenames[0]), img_as_ubyte(rgb_restored))

psnr_val_rgb = sum(psnr_val_rgb)/len(test_dataset)
ssim_val_rgb = sum(ssim_val_rgb)/len(test_dataset)
if args.unpaired== False : # 对于paired data
    print("PSNR: %f, SSIM: %f " %(psnr_val_rgb,ssim_val_rgb))


else :
    print('done!')
shutil.rmtree(tmp_dir)