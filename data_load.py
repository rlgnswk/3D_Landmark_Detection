import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import random
import natsort
import cv2 
from PIL import Image

class FaceLandMark_Loader(Dataset):
    def __init__(self, root, IsAug = True):
        super(FaceLandMark_Loader, self).__init__()
        print("#### MotionLoader ####")
        print("####### load data from {} ######".format(root))

        # read list of images
        # read list of landmarks
        self.img_path = os.path.join(root, "img")
        self.ldmks_path = os.path.join(root, "ldmks")
        img_list = natsort.natsorted(os.listdir(self.img_path))
        ldmks_list = natsort.natsorted(os.listdir(self.ldmks_path))
        
        self.std = 0.2
        self.mean = 0

        print(img_list[:10])
        print(ldmks_list[:10])

        #using augmentation functions which do not effect on landmarks position 
    def __getitem__(self, idx):
        
        # get an image
        # get a landmark info
        img = Image.open(self.img_path + '/' + img_list[idx])
        text_file = open(self.ldmks_path + '/' + ldmks_list[idx], "r")
        ldmks = text_file.read().split(' ').replace('\n', ' ').split(' ')
        ldmks.pop() # remove blank of last row

        '''
        Paper:
        During training, we perform data augmentation including
        rotations, perspective warps, < -- landmark also should be changed 
        blurs, modulations to brightness and contrast, addition of noise, and conversion to grayscale. < -- does not be related with landmark
        '''
        #blurs, modulations to brightness and contrast, addition of noise, and conversion to grayscale. < -- does not be related with landmark
        img = F.adjust_brightness(img, brightness_factor = random.uniform(0.5,1.5)) # brightness_factor 0(black) ~ 2(white)
        img = F.adjust_contrast(img, contrast_factor = random.uniform(0.5,1.5)) # contrast_factor 0(solid gray) ~ 2
        img = F.gaussian_blur(img, kernel_size = random.randint(3, 7))
        img = F.rgb_to_grayscale(img, num_output_channels =3)
        img = img + torch.randn(img.size()) * self.std + self.mean 
        
        #crop image 256x256 including face info --> how?
        
        #conduct augmentation --> how handle the annotation simultaneously??

        return landmark_GT, input_Image

def get_dataloader(dataroot, batch_size, IsSuffle = True):
    dataset = MotionLoader(dataroot)
    print("# of dataset:", len(dataset))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=IsSuffle, drop_last=True)
   
    return dataloader, dataset



if __name__ == '__main__':
    root = "/data2/MS-FaceSynthetic"
    #temp = FaceLandMark_Loader(root = "/data2/MS-FaceSynthetic")


    ldmks_list = natsort.natsorted(os.listdir(os.path.join(root, "ldmks")))
    text_file = open(os.path.join(root, "ldmks") +'/'+ldmks_list[1], "r")
    ldmks = text_file.read().replace('\n', ' ').split(' ')
    ldmks.pop()
    print(ldmks)
    print(len(ldmks))
    #print(np.array(ldmks).shape)