import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import cv2
def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])
def copyGT(input_dir,output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    input_files = sorted(os.listdir(input_dir))
    input_filenames = [os.path.join(input_dir, x) for x in input_files if is_png_file(x)]
    for i in range(len(input_filenames)):
        im = Image.open(input_filenames[i])
        filename = os.path.split(input_filenames[i])[-1]
        im.save(os.path.join(output_dir, filename))

def generateR(input_dir='./fullpatches/input',output_dir = './fullpatches_RD/input'):
   
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    input_files = sorted(os.listdir(input_dir))
    input_filenames = [os.path.join(input_dir, x) for x in input_files if is_png_file(x)]

    for i in range(len(input_filenames)):
        im = Image.open(input_filenames[i])
        # im_array = np.array(im)/255

        # im_gray = im_array[:,:,0:1]*0.299 + im_array[:,:,1:2] *0.587 + im_array[:,:,2:3]*0.114
        # im_gray  = np.power(im_gray, 0.6)

        # im_I = np.concatenate((im_gray, im_gray, im_gray), axis=2)
        # im_R = im_array / im_I
        # im_R[np.isnan(im_R)] = 0

        # im_R_max = np.max(im_R, axis=2)
        # im_R_max = np.clip(im_R_max, 1, 100000)
        # im_R_max = np.expand_dims(im_R_max, 2)
        # im_R_norm = np.concatenate((im_R_max, im_R_max, im_R_max), axis=2)
        # im_R = im_R / im_R_norm
        # im_R = im_R*0.5 + im_array*0.5
        # im_R = cv2.GaussianBlur(im_R,(5,5),0)

        # output_img = Image.fromarray(np.uint8(im_R*255))
        filename = os.path.split(input_filenames[i])[-1]
        # output_img.save(os.path.join(output_dir, filename))
        im.save(os.path.join(output_dir, filename))