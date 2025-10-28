import os
from skimage import io, transform, color, img_as_ubyte
import numpy as np
from torch.utils.data import Dataset
import cv2
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from albumentations.pytorch.transforms import ToTensor 
import albumentations as A
from PIL import Image 
import random
import torchvision.transforms as pytorch_transforms

class BinaryLoader(Dataset):
        def __init__(self, data_name, jsfiles, out_transforms, pixel_mean=[123.675, 116.280, 103.530], pixel_std=[58.395, 57.12, 57.375]):
            self.path = f'/data/coding/datasets/{data_name}'
            self.jsfiles = jsfiles
            self.transforms = out_transforms
            
        
        def __len__(self):
            return len(self.jsfiles)
              
        
        def __getitem__(self,idx):
            image_id = list(self.jsfiles[idx].split('.'))[0]

            image_path = os.path.join(self.path,'images/',image_id)
            mask_path = os.path.join(self.path,'masks/',image_id)
 
   
            img = io.imread(image_path+'.png')[:,:,:3].astype('float32')
            mask = io.imread(mask_path+'.png', as_gray=True).astype(np.uint8)
            mask[mask>0]=255
            
            data_group = self.transforms(image=img, mask=mask)
            img_resized = data_group['image']
            mask = data_group['mask']

            return (img_resized, mask, image_id)
        

