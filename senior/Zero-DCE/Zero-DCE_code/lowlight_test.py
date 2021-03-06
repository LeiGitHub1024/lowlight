import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
import cv2

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i',default='./data/test')
parser.add_argument('--output', '-r',default='./data/res')

args = parser.parse_args()

 
def lowlight(image_path):
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	data_lowlight = Image.open(image_path)
	data_lowlight = (np.asarray(data_lowlight)/255.0)

	data_lowlight = torch.from_numpy(data_lowlight).float()
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0)

	DCE_net = model.enhance_net_nopool().cuda()
	DCE_net.load_state_dict(torch.load('snapshots/Epoch99.pth'))
	start = time.time()
	_,enhanced_image,_ = DCE_net(data_lowlight)

	end_time = (time.time() - start)
	# print(end_time)
	# image_path = image_path.replace('input','res/zero')
	result_path = args.output
	if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
		os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))
	img = enhanced_image[0].cpu().numpy().transpose(1, 2, 0)
	RB = Image.fromarray(np.uint8(img*255))
	file_path = os.path.join(result_path, image_path.split("/")[-1])
	RB.save(file_path)
	# torchvision.utils.save_image(file_path, result_path)

if __name__ == '__main__':
# test_images
	with torch.no_grad():
		test_list = glob.glob(args.input +"/*") 
		for image in test_list:
			# image = image
			print(image)
			lowlight(image)

		

