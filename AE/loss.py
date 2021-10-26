# mbllen 损失函数
# def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
#     window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
#     K1 = 0.01
#     K2 = 0.03
#     L = 1  # depth of image (255 in case the image has a differnt scale)
#     C1 = (K1*L)**2
#     C2 = (K2*L)**2
#     mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
#     mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
#     mu1_sq = mu1*mu1
#     mu2_sq = mu2*mu2
#     mu1_mu2 = mu1*mu2
#     sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
#     sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
#     sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
#     if cs_map:
#         value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
#                             (sigma1_sq + sigma2_sq + C2)),
#                         (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
#     else:
#         value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
#                             (sigma1_sq + sigma2_sq + C2))

#     if mean_metric:
#         value = tf.reduce_mean(value)
#     return value


# def my_loss(y_true, y_pred):
#     MAE_loss = K.mean(K.abs(y_pred[:,:,:,:3] - y_true))
#     SSIM_loss = tf_ssim(tf.expand_dims(y_pred[:, :, :, 0], -1),tf.expand_dims(y_true[:, :, :, 0], -1)) + tf_ssim(
#         tf.expand_dims(y_pred[:, :, :, 1], -1), tf.expand_dims(y_true[:, :, :, 1], -1)) + tf_ssim(
#         tf.expand_dims(y_pred[:, :, :, 2], -1), tf.expand_dims(y_true[:, :, :, 2], -1))
#     VGG_loss = K.mean(K.abs(y_pred[:, :, :, 3:19] - y_pred[:, :, :, 19:35]))

#     percent = 0.4
#     index = int(256 * 256 * percent - 1)
#     gray1 = 0.39 * y_pred[:, :, :, 0] + 0.5 * y_pred[:, :, :, 1] + 0.11 * y_pred[:, :, :, 2]
#     gray = tf.reshape(gray1, [-1, 256 * 256])
#     gray_sort = tf.nn.top_k(-gray, 256 * 256)[0]
#     yu = gray_sort[:, index]
#     yu = tf.expand_dims(tf.expand_dims(yu, -1), -1)
#     # mask = tf.to_float(gray1 <= yu)
#     mask = tf.cast(gray1 <= yu,tf.float32)
#     mask1 = tf.expand_dims(mask, -1)
#     mask = tf.concat([mask1, mask1, mask1], -1)

#     low_fake_clean = tf.multiply(mask, y_pred[:, :, :, :3])
#     high_fake_clean = tf.multiply(1 - mask, y_pred[:, :, :, :3])
#     low_clean = tf.multiply(mask, y_true[:, :, :, :])
#     high_clean = tf.multiply(1 - mask, y_true[:, :, :, :])
#     Region_loss = K.mean(K.abs(low_fake_clean - low_clean) * 4 + K.abs(high_fake_clean - high_clean))

#     loss = MAE_loss + VGG_loss/3. + 3 - SSIM_loss + Region_loss
#     return loss


#https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from torch import nn
from torchvision import models, transforms
from torch.autograd import Variable


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return 1-ssim_map.mean()
    else:
        return 1-ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)



class LXJ_LOSS(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        
        
    def forward(self, x, y,):

        MAE_loss = torch.mean(torch.abs(x - y))

        (_, channel, _, _) = x.size()
        if channel == self.channel and self.window.data.type() == x.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if x.is_cuda:
                window = window.cuda(x.get_device())
            window = window.type_as(x)
            
            self.window = window
            self.channel = channel
        SSIM_loss =  _ssim(x, y, window, self.window_size, channel, self.size_average)


        loss = MAE_loss 
        return loss

# def lxjloss(y_pred,y_true):
    
#     SSIM_loss = ssim(y_pred,y_true)

