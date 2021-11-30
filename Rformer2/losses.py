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

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class MixLoss(nn.Module):
    def __init__(self, alpha=0.8, eps=1e-6, window_size = 11, size_average = True):
        super(MixLoss, self).__init__()
        self.eps = eps
        self.alpha = alpha
        # self.window_size = window_size
        # self.size_average = size_average
        # self.channel = 1
        # self.window = create_window(window_size, self.channel)

    def forward(self, x, y, epoch=None):
        """Charbonnier Loss (L1) （0-1）"""  
        diff = x - y
        l1_loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps))) #10->5

        """SSIM （0-1）"""
        ssim_module = SSIM(data_range=1., size_average=True, channel=3)
        ssim_loss = 1 - ssim_module(x, y)
        loss =  (1-self.alpha)*l1_loss + self.alpha*ssim_loss  
        return loss

class SSIMLoss(nn.Module):
    def __init__(self, eps=1e-6, window_size = 11, size_average = True):
        super(SSIMLoss, self).__init__()
        self.eps = eps

        # self.window_size = window_size
        # self.size_average = size_average
        # self.channel = 1
        # self.window = create_window(window_size, self.channel)

    def forward(self, x, y, epoch=None):
        """SSIM （0-1）"""
        ssim_module = SSIM(data_range=1., size_average=True, channel=3)
        # ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=3, win_size=7)
        ssim_loss =  1 - ssim_module(x, y) #100 ssim:50-7
        # ms_ssim_loss = 1000*(1 - ms_ssim_module(x,y)) #1000 ms-ssim:48 -> 3.4
        return ssim_loss  