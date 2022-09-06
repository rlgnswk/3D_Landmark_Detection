'''

During training, we perform data augmentation including
rotations, perspective warps, blurs, modulations to brightness
and contrast, addition of noise, and conversion to grayscale.

'''

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import random

class FaceLandMark_Loader(Dataset):
    def __init__(self, root):
        super(FaceLandMark_Loader, self).__init__()
        print("#### MotionLoader ####")
        print("####### load data from {} ######".format(root))



    def __getitem__(self, idx):

        
        return landmark_GT, input_Image


def get_dataloader(dataroot, batch_size, IsSuffle = True):
    dataset = MotionLoader(dataroot)
    print("# of dataset:", len(dataset))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=IsSuffle, drop_last=True)
   
    return dataloader, dataset
