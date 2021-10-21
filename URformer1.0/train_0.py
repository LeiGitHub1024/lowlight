import os
import sys

# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'./auxiliary/'))
print(dir_name)

import argparse
import options
######### parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='image denoising')).parse_args()
print(opt)

import utils
######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
import torch
torch.backends.cudnn.benchmark = True

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from natsort import natsorted
import glob
import random
import time
import numpy as np
from einops import rearrange, repeat
import datetime
from pdb import set_trace as stx

from losses import SSIMLoss_Gray,SSIMLoss_RGB,CharbonnierLoss

from tqdm import tqdm 
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler

from utils.loader import  get_training_data,get_validation_data

######### Logs dir ###########
log_dir = os.path.join(dir_name,'log', opt.arch+opt.env)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logname = os.path.join(log_dir, datetime.datetime.now().isoformat()+'.txt') 
print("Now time is : ",datetime.datetime.now().isoformat())
result_dir = os.path.join(log_dir, 'results')
model_dir  = os.path.join(log_dir, 'models')
utils.mkdir(result_dir)
utils.mkdir(model_dir)

# ######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

######### Model ###########
model_restoration = utils.get_arch(opt)

with open(logname,'a') as f:
    f.write(str(opt)+'\n')
    f.write(str(model_restoration)+'\n')

######### Optimizer ###########
start_epoch = 1
if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
else:
    raise Exception("Error optimizer...")


######### DataParallel ###########
model_restoration = torch.nn.DataParallel (model_restoration)
model_restoration.cuda()

######### Resume ###########
if opt.resume:
    path_chk_rest = opt.pretrain_weights
    utils.load_checkpoint(model_restoration,path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    lr = utils.load_optim(optimizer, path_chk_rest)

    for p in optimizer.param_groups: p['lr'] = lr
    warmup = False
    new_lr = lr
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:",new_lr)
    print('------------------------------------------------------------------------------')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-start_epoch+1, eta_min=1e-6)

######### Scheduler ###########
if opt.warmup:
    print("Using warmup and cosine strategy!")
    warmup_epochs = opt.warmup_epochs
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
else:
    step = 50
    print("Using StepLR,step={}!".format(step))
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    scheduler.step()


######### Loss ###########

#ssim_1 = SSIMLoss_Gray().cuda()
#ssim_3 = SSIMLoss_RGB().cuda()
ssim_1 = CharbonnierLoss().cuda()
ssim_3 = CharbonnierLoss().cuda()
######### DataLoader ###########
print('===> Loading datasets')
img_options_train = {'patch_size':opt.train_ps}
train_dataset = get_training_data(opt.train_dir, img_options_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, 
        num_workers=opt.train_workers, pin_memory=True, drop_last=False)

val_dataset = get_validation_data(opt.val_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False, 
        num_workers=opt.eval_workers, pin_memory=False, drop_last=False)

len_trainset = train_dataset.__len__()
len_valset = val_dataset.__len__()
print("Sizeof training set: ", len_trainset,", sizeof validation set: ", len_valset)
######### validation ###########
with torch.no_grad():
    psnr_val_rgb = []
    for ii, data_val in enumerate((val_loader), 0):
        target = data_val[0].cuda()
        input_ = data_val[1].cuda()
        filenames = data_val[2]
        psnr_val_rgb.append(utils.batch_PSNR(input_, target, False).item())
    psnr_val_rgb = sum(psnr_val_rgb)/len_valset
    print('Input & GT (PSNR) -->%.4f dB'%(psnr_val_rgb))

######### train ###########
print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.nepoch))
best_loss = 1e9
best_epoch = 0
best_iter = 0
eval_now = len(train_loader)//4
print("\nEvaluation after every {} Iterations !!!\n".format(eval_now))

loss_scaler = NativeScaler()
torch.cuda.empty_cache()
for epoch in range(start_epoch, opt.nepoch + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    for i, data in enumerate(train_loader, 0): 
        # zero_grad
        optimizer.zero_grad()

        S_normal = data[0].cuda()
        S_low = data[1].cuda()
        
        Gray_normal = S_normal[:,0:1,:,:]*0.299 + S_normal[:,1:2,:,:]*0.587 + S_normal[:,2:3,:,:]*0.114
        Gray_low = S_low[:,0:1,:,:]*0.299 + S_low[:,1:2,:,:]*0.587 + S_low[:,2:3,:,:]*0.114
        Gray_low = Gray_low*0.7 + Gray_normal*0.3

        Gray_normal_3 = torch.cat((Gray_normal, Gray_normal, Gray_normal), 1)
        R_guide = torch.div(S_normal, Gray_normal_3)
        R_guide = torch.where(torch.isnan(R_guide), torch.full_like(R_guide, 1), R_guide)
        #if (torch.any(torch.isnan(R_guide))) :
        #    print("!!!Find NAN!!!")
        R_guide = torch.clamp(R_guide, 0,1)
        R_guide = R_guide*0.7 + S_normal*0.3

        #if epoch>5:
         #   target, input_ = utils.MixUp_AUG().aug(target, input_)
        with torch.cuda.amp.autocast():
            R_low, I_low = model_restoration(S_low, 0)
            R_normal, I_normal = model_restoration(S_normal, 0)


            R_low = torch.clamp(R_low, 0,1)  
            I_low = torch.clamp(I_low, 0,1)  
            R_normal = torch.clamp(R_normal, 0,1)  
            I_normal = torch.clamp(I_normal, 0,1)  
            
            #Our Loss
            #print(ssim_3(S_low, S_low))
            loss = ssim_3(S_low, R_low.mul(I_low)) + ssim_3(S_normal, R_normal.mul(I_normal)) \
                 + 0.2 * ssim_3(R_low, R_normal) + 0.2 * ssim_3(R_normal, R_guide) + 0.2 * ssim_3(R_low, R_guide)\
                 + 0.1 * ssim_1(I_low, Gray_low) + 0.1 * ssim_1(I_normal, Gray_normal)

            #Loss of KinD 
            #loss = decom_loss(S_low,S_normal,I_low,I_normal,R_low,R_normal)
            '''
            #Loss of Retinex Net 
            #loss_reconst_dec = reconst_loss(S_low, R_low.mul(I_low)) + reconst_loss(S_normal, R_normal.mul(I_normal)) \
            loss_reconst_dec = reconst_loss(S_low, R_low.mul(I_low)) + reconst_loss(S_normal, R_normal.mul(I_normal)) + 0.001*reconst_loss(S_normal, R_low.mul(I_normal)) + 0.001* reconst_loss(S_low, R_normal.mul(I_low))   
                        
            loss_ivref = 0.01 * reconst_loss(R_low, R_normal)
            

            smooth_loss_low = smooth_loss(I_low, R_low)
            smooth_loss_normal = smooth_loss(I_normal, R_normal)

            loss = loss_reconst_dec + loss_ivref + 0.1 * smooth_loss_low + 0.1 * smooth_loss_normal
            '''
        # print('loss:',loss.item())
        loss_scaler(
                loss, optimizer,parameters=model_restoration.parameters())
        epoch_loss +=loss.item()

        #### Evaluation ####
        if (i+1)%eval_now==0 and i>0:
            with torch.no_grad():
                model_restoration.eval()
                loss_val= []
                for ii, data_val in enumerate((val_loader), 0):
                    S_normal = data[0].cuda()
                    S_low = data[1].cuda()

                    Gray_normal = S_normal[:,0:1,:,:]*0.299 + S_normal[:,1:2,:,:]*0.587 + S_normal[:,2:3,:,:]*0.114
                    Gray_low = S_low[:,0:1,:,:]*0.299 + S_low[:,1:2,:,:]*0.587 + S_low[:,2:3,:,:]*0.114
                    Gray_low = Gray_low*0.7 + Gray_normal*0.3

                    Gray_normal_3 = torch.cat((Gray_normal, Gray_normal, Gray_normal), 1)
                    R_guide = torch.div(S_normal, Gray_normal_3)
                    R_guide = torch.where(torch.isnan(R_guide), torch.full_like(R_guide, 1), R_guide)
                    #if (torch.any(torch.isnan(R_guide))) :
                    #    print("!!!Find NAN!!!")
                    R_guide = torch.clamp(R_guide, 0,1)
                    R_guide = R_guide*0.7 + S_normal*0.3

                    filenames = data_val[2]
                    with torch.cuda.amp.autocast():
                        R_low, I_low = model_restoration(S_low, 0)
                        R_normal, I_normal = model_restoration(S_normal, 0)
                    R_low = torch.clamp(R_low,0,1)  
                    I_low = torch.clamp(I_low,0,1)  
                    R_normal = torch.clamp(R_normal,0,1)  
                    I_normal = torch.clamp(I_normal,0,1) 
                    
                    #Our Loss
                    vloss = ssim_3(S_low, R_low.mul(I_low)) + ssim_3(S_normal, R_normal.mul(I_normal)) \
                            + 0.2 * ssim_3(R_low, R_normal) + 0.2 * ssim_3(R_normal, R_guide) + 0.2 * ssim_3(R_low, R_guide) \
                            + 0.1 * ssim_1(I_low, Gray_low) + 0.1 * ssim_1(I_normal, Gray_normal)

                    #vloss = decom_loss(S_low,S_normal,I_low,I_normal,R_low,R_normal)
                    '''
                    #loss_reconst_dec = reconst_loss(S_low, R_low.mul(I_low)) + reconst_loss(S_normal, R_normal.mul(I_normal)) \
                    loss_reconst_dec = reconst_loss(S_low, R_low.mul(I_low)) + reconst_loss(S_normal, R_normal.mul(I_normal)) + 0.001*reconst_loss(S_normal, R_low.mul(I_normal)) + 0.001* reconst_loss(S_low, R_normal.mul(I_low))   
                    loss_ivref = 0.01 * reconst_loss(R_low, R_normal)
                    
                    #print(I_low.type(), R_low.type())
                    smooth_loss_low = smooth_loss(I_low, R_low)
                    smooth_loss_normal = smooth_loss(I_normal, R_normal)

                    vloss = loss_reconst_dec + loss_ivref + 0.1 * smooth_loss_low + 0.1 * smooth_loss_normal
                    '''

                    loss_val.append(vloss)

                valloss = sum(loss_val)/len_valset
                
                if best_loss > valloss:
                    best_loss = valloss
                    best_epoch = epoch
                    best_iter = i 
                    torch.save({'epoch': epoch, 
                                'state_dict': model_restoration.state_dict(),
                                'optimizer' : optimizer.state_dict()
                                }, os.path.join(model_dir,"model_best.pth"))

                print("[Ep %d it %d\t valloss: %.4f\t] --loss%.4f--  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " % (epoch, i, valloss,loss.item(),best_epoch,best_iter,best_loss))
                with open(logname,'a') as f:
                    f.write("[Ep %d it %d\t valloss: %.4f\t] --loss%.4f--  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " % (epoch, i, valloss,loss.item(),best_epoch,best_iter,best_loss))
                model_restoration.train()
                torch.cuda.empty_cache()
    scheduler.step()
    
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")
    with open(logname,'a') as f:
        f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, scheduler.get_lr()[0])+'\n')

    torch.save({'epoch': epoch, 
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_latest.pth"))   

    if epoch%opt.checkpoint == 0:
        torch.save({'epoch': epoch, 
                    'state_dict': model_restoration.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,"model_epoch_{}.pth".format(epoch))) 
print("Now time is : ",datetime.datetime.now().isoformat())
