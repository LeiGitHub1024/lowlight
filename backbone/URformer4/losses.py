import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM




def tv_loss(x, beta = 0.5, reg_coeff = 5):
    '''Calculates TV loss for an image `x`.
        
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta` 
    '''
    dh = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2)
    dw = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2)
    a,b,c,d=x.shape
    return reg_coeff*(torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta))/(a*b*c*d))

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]



class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

# def gaussian(window_size, sigma):
#     gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
#     return gauss/gauss.sum()

# def create_window(window_size, channel):
#     _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
#     _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
#     window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
#     return window

# def _ssim(img1, img2, window, window_size, channel, size_average = True):
#     mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
#     mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1*mu2

#     sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
#     sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
#     sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

#     C1 = 0.01**2
#     C2 = 0.03**2

#     ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

#     if size_average:
#         return 1-ssim_map.mean()
#     else:
#         return 1-ssim_map.mean(1).mean(1).mean(1)

# class SSIM(torch.nn.Module):
#     def __init__(self, window_size = 11, size_average = True):
#         super(SSIM, self).__init__()
#         self.window_size = window_size
#         self.size_average = size_average
#         self.channel = 1
#         self.window = create_window(window_size, self.channel)

#     def forward(self, img1, img2):
#         (_, channel, _, _) = img1.size()

#         if channel == self.channel and self.window.data.type() == img1.data.type():
#             window = self.window
#         else:
#             window = create_window(self.window_size, channel)
            
#             if img1.is_cuda:
#                 window = window.cuda(img1.get_device())
#             window = window.type_as(img1)
            
#             self.window = window
#             self.channel = channel


#         return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

# def ssim(img1, img2, window_size = 11, size_average = True):
#     (_, channel, _, _) = img1.size()
#     window = create_window(window_size, channel)
    
#     if img1.is_cuda:
#         window = window.cuda(img1.get_device())
#     window = window.type_as(img1)
    
#     return _ssim(img1, img2, window, window_size, channel, size_average)


def darkLoss(img1,img2):
    percent = 0.4
    index = int(256*256*percent-1)

    

class MyLoss(nn.Module):
    
    def __init__(self, eps=1e-3, window_size = 11, size_average = True):
        super(MyLoss, self).__init__()
        self.eps = eps

        # self.window_size = window_size
        # self.size_average = size_average
        # self.channel = 1
        # self.window = create_window(window_size, self.channel)

    def forward(self, x, y,epoch):
        """Charbonnier Loss (L1) （0-1）"""  
        diff = x - y
        l1_loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps))) #10->5

        """SSIM （0-1）"""
        ssim_module = SSIM(data_range=255, size_average=True, channel=3)
        # ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=3, win_size=7)

        ssim_loss =  100*(1 - ssim_module(x, y)) #100 ssim:50-7
        # ms_ssim_loss = 1000*(1 - ms_ssim_module(x,y)) #1000 ms-ssim:48 -> 3.4
        """Dark Loss"""
        # Dark_loss = darkLoss(x,y)



        loss =  l1_loss + ssim_loss  




        return loss

class SSIMLoss(nn.Module):
    def __init__(self, eps=1e-3, window_size = 11, size_average = True):
        super(SSIMLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y,epoch):
        """Charbonnier Loss (L1) （0-1）"""  
        diff = x - y
        l1_loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps))) #10->5

        """SSIM （0-1）"""
        ssim_module = SSIM(data_range=255, size_average=True, channel=3)
        # ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=3, win_size=7)

        ssim_loss =  (1 - ssim_module(x, y)) #100 ssim:50-7
        # ms_ssim_loss = 1000*(1 - ms_ssim_module(x,y)) #1000 ms-ssim:48 -> 3.4
        """Dark Loss"""
        # Dark_loss = darkLoss(x,y)
        loss =  ssim_loss  
        return loss

class L1Loss(nn.Module):
    
    def __init__(self, eps=1e-3, window_size = 11, size_average = True):
        super(L1Loss, self).__init__()
        self.eps = eps

        # self.window_size = window_size
        # self.size_average = size_average
        # self.channel = 1
        # self.window = create_window(window_size, self.channel)

    def forward(self, x, y,epoch):
        """Charbonnier Loss (L1) （0-1）"""  
        diff = x - y
        l1_loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps))) #10->5

        """SSIM （0-1）"""
        ssim_module = SSIM(data_range=255, size_average=True, channel=3)
        # ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=3, win_size=7)

        ssim_loss =  100*(1 - ssim_module(x, y)) #100 ssim:50-7
        # ms_ssim_loss = 1000*(1 - ms_ssim_module(x,y)) #1000 ms-ssim:48 -> 3.4
        """Dark Loss"""
        # Dark_loss = darkLoss(x,y)
        loss =  l1_loss
        return loss