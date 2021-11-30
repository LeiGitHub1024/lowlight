import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils import is_png_file, load_img, Augment_RGB_torch
import torch.nn.functional as F
import random
import math
from skimage.transform import resize

augment   = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 

##################################################################################################
class DataLoaderTrain_stage0(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None):
        super(DataLoaderTrain_stage0, self).__init__()

        self.target_transform = target_transform
        
        gt_dir = 'groundtruth'
        input_dir = 'input'
        
        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        
        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if is_png_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x)       for x in noisy_files if is_png_file(x)]
        
        self.img_options=img_options

        self.tar_size = len(self.clean_filenames)  # get the size of target

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
        
        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        #Crop Input and Target
        ps = self.img_options['patch_size']
        H = clean.shape[1]
        W = clean.shape[2]
        # r = np.random.randint(0, H - ps) if not H-ps else 0
        # c = np.random.randint(0, W - ps) if not H-ps else 0
        if H-ps==0:
            r=0
            c=0
        else:
            r = np.random.randint(0, H - ps)
            c = np.random.randint(0, W - ps)
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps]

        apply_trans = transforms_aug[random.getrandbits(3)]

        clean = getattr(augment, apply_trans)(clean)
        noisy = getattr(augment, apply_trans)(noisy)        

        return clean, noisy, clean_filename, noisy_filename
        
class DataLoaderTrain_stage1(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None):
        super(DataLoaderTrain_stage1, self).__init__()

        self.target_transform = target_transform
        
        gt_dir = 'groundtruth'
        reflex_dir = 'reflex'
        input_dir = 'input'
        
        gt_files     = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        reflex_files = sorted(os.listdir(os.path.join(rgb_dir, reflex_dir)))
        input_files  = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        
        self.gt_filenames     = [os.path.join(rgb_dir, gt_dir, x)     for x in gt_files if is_png_file(x)]
        self.reflex_filenames = [os.path.join(rgb_dir, reflex_dir, x) for x in reflex_files if is_png_file(x)]
        self.input_filenames  = [os.path.join(rgb_dir, input_dir, x)  for x in input_files if is_png_file(x)]
        
        self.img_options = img_options

        self.tar_size = len(self.gt_filenames)  # get the size of target

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size
        
        gt_img     = torch.from_numpy(np.float32(load_img(self.gt_filenames[tar_index])))
        reflex_img = torch.from_numpy(np.float32(load_img(self.reflex_filenames[tar_index])))
        input_img  = torch.from_numpy(np.float32(load_img(self.input_filenames[tar_index])))
        
        gt_img     = gt_img.permute(2,0,1)
        reflex_img = reflex_img.permute(2,0,1)
        input_img  = input_img.permute(2,0,1)

        gt_filename     = os.path.split(self.gt_filenames[tar_index])[-1]
        reflex_filename = os.path.split(self.reflex_filenames[tar_index])[-1]
        input_filename  = os.path.split(self.input_filenames[tar_index])[-1]

        #Crop Input and Target
        ps = self.img_options['patch_size']
        H = gt_img.shape[1]
        W = gt_img.shape[2]
        # r = np.random.randint(0, H - ps) if not H-ps else 0
        # c = np.random.randint(0, W - ps) if not H-ps else 0
        if H-ps==0:
            r = 0
            c = 0
        else:
            r = np.random.randint(0, H - ps)
            c = np.random.randint(0, W - ps)
        gt_img     = gt_img[:, r:r + ps, c:c + ps]
        reflex_img = reflex_img[:, r:r + ps, c:c + ps]
        input_img  = input_img[:, r:r + ps, c:c + ps]

        apply_trans = transforms_aug[random.getrandbits(3)]

        gt_img     = getattr(augment, apply_trans)(gt_img)
        reflex_img = getattr(augment, apply_trans)(reflex_img)
        input_img  = getattr(augment, apply_trans)(input_img)        

        return gt_img, reflex_img, input_img, gt_filename, reflex_filename, input_filename
        
class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None):
        super(DataLoaderTrain, self).__init__()

        self.target_transform = target_transform
        
        gt_dir = 'groundtruth'
        input_dir = 'input'
        
        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        
        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if is_png_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x)       for x in noisy_files if is_png_file(x)]
        
        self.img_options=img_options

        self.tar_size = len(self.clean_filenames)  # get the size of target

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
        
        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        #Crop Input and Target
        ps = self.img_options['patch_size']
        H = clean.shape[1]
        W = clean.shape[2]
        # r = np.random.randint(0, H - ps) if not H-ps else 0
        # c = np.random.randint(0, W - ps) if not H-ps else 0
        if H-ps==0:
            r=0
            c=0
        else:
            r = np.random.randint(0, H - ps)
            c = np.random.randint(0, W - ps)
            clean = clean[:, r:r + ps, c:c + ps]
            noisy = noisy[:, r:r + ps, c:c + ps]

        apply_trans = transforms_aug[random.getrandbits(3)]

        clean = getattr(augment, apply_trans)(clean)
        noisy = getattr(augment, apply_trans)(noisy)        

        return clean, noisy, clean_filename, noisy_filename

class DataLoaderTrain_R(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None):
        super(DataLoaderTrain_R, self).__init__()

        self.target_transform = target_transform
        
        gt_dir = 'groundtruth'
        input_dir = 'input'
        
        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        
        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if is_png_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x)       for x in noisy_files if is_png_file(x)]
        
        self.img_options=img_options

        self.tar_size = len(self.clean_filenames)  # get the size of target

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        clean_arr = load_img(self.clean_filenames[tar_index])
        noisy_arr = load_img(self.noisy_filenames[tar_index])
        clean = torch.from_numpy(np.float32(clean_arr))
        noisy = torch.from_numpy(np.float32(noisy_arr))
        
        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        #Crop Input and Target
        ps = self.img_options['patch_size']
        H = clean.shape[1]
        W = clean.shape[2]
        # r = np.random.randint(0, H - ps) if not H-ps else 0
        # c = np.random.randint(0, W - ps) if not H-ps else 0
        if H-ps==0:
            r=0
            c=0
        else:
            r = np.random.randint(0, H - ps)
            c = np.random.randint(0, W - ps)
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps]

        apply_trans = transforms_aug[random.getrandbits(3)]

        clean = getattr(augment, apply_trans)(clean)
        noisy = getattr(augment, apply_trans)(noisy)        

        return clean, noisy, clean_filename, noisy_filename
##################################################################################################

class DataLoaderTrain_Gaussian(Dataset):
    def __init__(self, rgb_dir, noiselevel=5, img_options=None, target_transform=None):
        super(DataLoaderTrain_Gaussian, self).__init__()

        self.target_transform = target_transform
        #pdb.set_trace()
        clean_files = sorted(os.listdir(rgb_dir))
        #noisy_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        #clean_files = clean_files[0:83000]
        #noisy_files = noisy_files[0:83000]
        self.clean_filenames = [os.path.join(rgb_dir, x) for x in clean_files if is_png_file(x)]
        #self.noisy_filenames = [os.path.join(rgb_dir, 'input', x)       for x in noisy_files if is_png_file(x)]
        self.noiselevel = noiselevel
        self.img_options=img_options

        self.tar_size = len(self.clean_filenames)  # get the size of target
        print(self.tar_size)
    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        #print(self.clean_filenames[tar_index])
        clean = np.float32(load_img(self.clean_filenames[tar_index]))
        #noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
        # noiselevel = random.randint(5,20)
        noisy = clean + np.float32(np.random.normal(0, self.noiselevel, np.array(clean).shape)/255.)
        noisy = np.clip(noisy,0.,1.)
        
        clean = torch.from_numpy(clean)
        noisy = torch.from_numpy(noisy)

        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.clean_filenames[tar_index])[-1]

        #Crop Input and Target
        ps = self.img_options['patch_size']
        H = clean.shape[1]
        W = clean.shape[2]
        r = np.random.randint(0, H - ps)
        c = np.random.randint(0, W - ps)
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps]

        apply_trans = transforms_aug[random.getrandbits(3)]

        clean = getattr(augment, apply_trans)(clean)
        noisy = getattr(augment, apply_trans)(noisy)

        return clean, noisy, clean_filename, noisy_filename
##################################################################################################
def tensorResize(timg, factor=128):
    c, h, w = timg.size()
    X = int(math.ceil(max(h,w)/float(factor))*factor)

    np_arr = timg.cpu().detach().numpy()
    np_arr = np_arr.transpose((1,2,0))

    #Image resize
    im_np_resize = resize(np_arr, (X, X))
    im_np_resize = im_np_resize.transpose((2,0,1))

    return torch.from_numpy(im_np_resize)

class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform

        gt_dir = 'groundtruth'
        input_dir = 'input'
        
        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))


        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if is_png_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files if is_png_file(x)]
        

        self.tar_size = len(self.clean_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        
        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
                
        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)

        #print('before', clean.shape)
        #clean = tensorResize(clean, factor=128) 
        #print('after', clean.shape)
        #noisy = tensorResize(noisy, factor=128) 
        return clean, noisy, clean_filename, noisy_filename

##################################################################################################

class DataLoaderTest(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderTest, self).__init__()

        self.target_transform = target_transform

        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))


        self.noisy_filenames = [os.path.join(rgb_dir, 'input', x) for x in noisy_files if is_png_file(x)]
        

        self.tar_size = len(self.noisy_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        

        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
                
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        noisy = noisy.permute(2,0,1)

        return noisy, noisy_filename


##################################################################################################

class DataLoaderTestSR(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderTestSR, self).__init__()

        self.target_transform = target_transform

        LR_files = sorted(os.listdir(os.path.join(rgb_dir)))


        self.LR_filenames = [os.path.join(rgb_dir, x) for x in LR_files if is_png_file(x)]
        

        self.tar_size = len(self.LR_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        

        LR = torch.from_numpy(np.float32(load_img(self.LR_filenames[tar_index])))
                
        LR_filename = os.path.split(self.LR_filenames[tar_index])[-1]

        LR = LR.permute(2,0,1)

        return LR, LR_filename
