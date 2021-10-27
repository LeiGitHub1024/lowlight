import os
import sys
import argparse
import options
import utils
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import time
import numpy as np
import datetime
from losses import MyLoss
from ssim import SSIM_Loss
from tqdm import tqdm 
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler
from utils.loader import get_training_data,get_validation_data



# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'./auxiliary/'))
print(dir_name)

######### parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='image denoising')).parse_args()
print(opt)


######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
torch.backends.cudnn.benchmark = True

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

######### DataParallel ###########
model_restoration = torch.nn.DataParallel (model_restoration)
model_restoration.cuda()

######### Loss ###########
criterion = SSIM_Loss(data_range=1.0, size_average=True, channel=3)

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
LR = 0.0002
# epoch_num = 250
# optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial)
optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
best_psnr = 0
best_epoch = 0
best_iter = 0
train_num, val_num = 0, 0
eval_now = len(train_loader)//3

loss_scaler = NativeScaler()
torch.cuda.empty_cache()
for epoch in range(opt.nepoch):
    epoch_start_time = time.time()
    epoch_loss = 0

    train_loss_epoch, val_loss_epoch = 0, 0
    
    # 训练
    for step, data in enumerate(train_loader):
        optimizer.zero_grad()

        target = data[0].cuda()
        input_ = data[1].cuda()
        model_restoration.train()

        with torch.cuda.amp.autocast():
            restored = model_restoration(input_)
            restored = torch.clamp(restored,0,1)  
            loss = criterion(restored, target)
        # restored = model_restoration(input_)
        # restored = torch.clamp(restored,0,1)  
        # loss = criterion(restored, target)


        # loss.backward()
        # optimizer.step()
        loss_scaler(loss, optimizer,parameters=model_restoration.parameters())

        
        epoch_loss +=loss.item()
        train_loss_epoch += loss.item() * input_.size(0)
        train_num += input_.size(0)

       
    #### Evaluation ####
    with torch.no_grad():
        model_restoration.eval()
        psnr_val_rgb = []
        for ii, data_val in enumerate((val_loader), 0):
            target = data_val[0].cuda()
            input_ = data_val[1].cuda()
            filenames = data_val[2]
            with torch.cuda.amp.autocast():
                restored = model_restoration(input_)
            restored = torch.clamp(restored,0,1)  
            psnr_val_rgb.append(utils.batch_PSNR(restored, target, False).item())

        psnr_val_rgb = sum(psnr_val_rgb)/len_valset
        
        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch = epoch
            best_iter = step
            torch.save({'epoch': epoch, 
                        'state_dict': model_restoration.state_dict(),
                        'optimizer' : optimizer.state_dict()
                        }, os.path.join(model_dir,"model_best.pth"))

        print("[Ep %d it %d\t PSNR SIDD: %.4f\t] --loss%.4f--  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " % (epoch, step, psnr_val_rgb,loss.item(),best_epoch,best_iter,best_psnr))
        with open(logname,'a') as f:
            f.write("[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " \
                % (epoch, step, psnr_val_rgb,best_epoch,best_iter,best_psnr)+'\n')
        model_restoration.train()
        torch.cuda.empty_cache()

    # 计算一个epoch的损失
    train_loss = train_loss_epoch / train_num
    # val_loss = val_loss_epoch / val_num
    # print('epoch',epoch+1,'loss',train_loss)
    train_num, val_num = 0, 0

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,train_loss,LR))
    print("------------------------------------------------------------------")
    with open(logname,'a') as f:
        f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}\n".format(epoch, time.time()-epoch_start_time,train_loss,LR))

    torch.save({'epoch': epoch, 
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_latest.pth"))   

    if epoch%opt.checkpoint == 0:
        torch.save({'epoch': epoch, 
                    'state_dict': model_restoration.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,"model_epoch_{}.pth".format(epoch))) 