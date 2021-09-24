import numpy as np
import matplotlib.image as mpimg # mpimg 用于读取图片
from skimage.metrics import peak_signal_noise_ratio
import torch
from tqdm import tqdm
import os
from network import DenoiseAutoEncoder

#在loldataset上计算平均psnr
def lolPSNR(path,device):
  DAEmodel = DenoiseAutoEncoder().to(device)
  DAEmodel.load_state_dict(torch.load(path))
  DAEmodel.eval()
  lolList = os.listdir('../data/lol485/high')
  PSNR_init = []
  PSNR_enhance = []
  PSNR_plus = []
  for imgName in tqdm(lolList):
      high = mpimg.imread('../data/lol485/high/'+imgName) # 
      low = mpimg.imread('../data/lol485/low/'+imgName) # 
      input = np.transpose(low, (2, 1, 0))
      input = torch.tensor(input, dtype=torch.float32).to(device)
      input = input.unsqueeze(0)
      _, output = DAEmodel(input)
      im_enhance= np.transpose(output.cpu().data.numpy(), (0, 3, 2, 1))
      im_enhance = im_enhance[0, ...]
      PSNR_init.append(peak_signal_noise_ratio(low, high))
      PSNR_enhance.append(peak_signal_noise_ratio(im_enhance, high))
      PSNR_plus.append(peak_signal_noise_ratio(im_enhance, high)-peak_signal_noise_ratio(low, high))
      # print("原始的PSNR:", peak_signal_noise_ratio(low, high))
      # print("暗光增强后的PSNR:", peak_signal_noise_ratio(im_enhance, high))
  print("原始平均PSNR",np.mean(PSNR_init))
  print("增强后平均PSNR",np.mean(PSNR_enhance))
  print("平均提升值",np.mean(PSNR_plus))