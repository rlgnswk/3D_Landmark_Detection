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

idx = 0 # target image index
root = "/data2/MS-FaceSynthetic"
#temp = FaceLandMark_Loader(root = "/data2/MS-FaceSynthetic")

ldmks_list = natsort.natsorted(os.listdir(os.path.join(root, "ldmks")))
print("len(ldmks_list): ", len(ldmks_list))

ldmks = pandas.read_csv(os.path.join(root, "ldmks") +'/'+ldmks_list[idx],  header=None, sep=' ')
ldmks = np.asarray(ldmks)
print("ldmks.shape: ",ldmks.shape)

img_path = os.path.join(root, "img")
img_list = natsort.natsorted(os.listdir(img_path))
img = Image.open(img_path + '/' + img_list[idx])
#print(img.shape)
print(np.array(img).shape)

# with eye points: 70
print("ldmks[:, 0].shape: ",ldmks[:, 0].shape)
plt.imshow(img)
plt.scatter(ldmks[:, 0], ldmks[:, 1], s=10, marker='.', c='g')
plt.savefig('visual_test_70points.png')

# without eye points: 68
plt.clf()
print("ldmks[:-2, 0].shape: ", ldmks[:-2, 0].shape)
plt.imshow(img)
plt.scatter(ldmks[:-2, 0], ldmks[:-2, 1], s=10, marker='.', c='g')
plt.savefig('visual_test_68points.png')
#print(np.array(ldmks).shape)