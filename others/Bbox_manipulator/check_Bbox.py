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
import pandas
import matplotlib.pyplot as plt

idx = 0
root = "/data2/MS-FaceSynthetic"
#temp = FaceLandMark_Loader(root = "/data2/MS-FaceSynthetic")

img_path = os.path.join(root, "img")
img_list = natsort.natsorted(os.listdir(img_path))
img = Image.open(img_path + '/' + img_list[idx])
#print(img.shape)
print(np.array(img).shape)

bbox_leftcorner_coord_path = os.path.join(root, "bbox_leftcorner_coord")
bbox_list = natsort.natsorted(os.listdir(bbox_leftcorner_coord_path))

bbox_leftcorner = pandas.read_csv(bbox_leftcorner_coord_path +'/'+bbox_list[idx],  header=None, sep=' ')
bbox_leftcorner = np.asarray(bbox_leftcorner) # shape : (2, 1) # x, y
print(bbox_leftcorner)
print(bbox_leftcorner.shape)
print(bbox_leftcorner[0][0]) # x
print(bbox_leftcorner[0][1]) # y
print(type(bbox_leftcorner[0][1]))

plt.imshow(img)
plt.savefig('image_GT_%d.png'%(idx))

plt.clf()
plt.imshow(img.crop((bbox_leftcorner[0][0], bbox_leftcorner[0][1], bbox_leftcorner[0][0]+ 256, bbox_leftcorner[0][1] + 256)))
plt.savefig('bounded_image_%d.png'%(idx))