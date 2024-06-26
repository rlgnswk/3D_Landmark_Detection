import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
import torchvision
import torch
import torch.nn as nn
#import torch.nn.functional as F
import torchvision.transforms.functional as F

from torchvision.utils import save_image
import math
import torch.optim as optim
import os
import random
import natsort
import cv2 
from PIL import Image, ImageFilter, ImageOps
import pandas
import matplotlib.pyplot as plt

# References
# https://github.com/FunkyKoki/Laplace_Landmark_Localization/blob/e8067bcd90c66227676cc1482e9defc3c9baa659/datasets/datasetsTools.py#L100
# https://tutorials.pytorch.kr/beginner/data_loading_tutorial.html
# https://towardsdatascience.com/face-landmarks-detection-with-pytorch-4b4852f5e9c4

class FaceLandMark_Loader(Dataset):
    def __init__(self, root, args, IsAug = True):
        super(FaceLandMark_Loader, self).__init__()
        print("#### MotionLoader ####")
        print("####### load data from {} ######".format(root))

        # read list of images
        # read list of landmarks
        # read list of bbox
        self.img_path = os.path.join(root, "img")
        self.ldmks_path = os.path.join(root, "ldmks")
        self.bbox_leftcorner_coord_path = os.path.join(root, "bbox_leftcorner_coord")

        self.img_list = natsort.natsorted(os.listdir(self.img_path))
        self.ldmks_list = natsort.natsorted(os.listdir(self.ldmks_path))
        self.bbox_list = natsort.natsorted(os.listdir(self.bbox_leftcorner_coord_path))

        assert len(self.img_list) == len(self.ldmks_list)
        assert len(self.img_list) == len(self.bbox_list)

        self.IsAug = IsAug
        self.totensor = torchvision.transforms.ToTensor()
        self.greyscale = torchvision.transforms.Grayscale(num_output_channels=3)
        self.blurrer = torchvision.transforms.GaussianBlur\
                (kernel_size=(args.GaussianBlur_kernel_w, args.GaussianBlur_kernel_h), sigma=(args.GaussianBlur_sigma_min, args.GaussianBlur_sigma_max))
        self.perspective_transformer = torchvision.transforms.RandomPerspective(distortion_scale=args.perspective_distortion_scale, p=args.perspective_distortion_prob)
        self.rotation_max_angle = args.rotation_max_angle
        self.noise_std_scale = args.noise_std_scale
        self.grayscale_prob = args.grayscale_prob

        self.brightness_factor_min = args.brightness_factor_min
        self.brightness_factor_max = args.brightness_factor_max
        self.contrast_factor_min = args.contrast_factor_min
        self.contrast_factor_max = args.contrast_factor_max
        #print(self.img_list[:10])
        #print(self.ldmks_list[:10])
         
    def __getitem__(self, idx):
        
        img = Image.open(self.img_path + '/' + self.img_list[idx])
        
        ldmks = pandas.read_csv(self.ldmks_path +'/'+self.ldmks_list[idx],  header=None, sep=' ')
        ldmks = np.asarray(ldmks) # shape : (70, 2), last two row for centers of eyes

        bbox_leftcorner = pandas.read_csv(self.bbox_leftcorner_coord_path +'/'+self.bbox_list[idx],  header=None, sep=' ')
        bbox_leftcorner = np.asarray(bbox_leftcorner) # shape : (2, 1) # x, y
        
        #crop image 256x256 including face info --> how? # crop이 가장 위에 가야지 계산 효율성이 좋을 듯!
        crop_img = img.crop((bbox_leftcorner[0][0], bbox_leftcorner[0][1], bbox_leftcorner[0][0]+ 256, bbox_leftcorner[0][1] + 256))
        crop_ladmks = self._landmark_processing4crop(ldmks, bbox_leftcorner)

        crop_img = self.totensor(crop_img)
        if self.IsAug == True:
            
            crop_img = F.adjust_brightness(crop_img, brightness_factor = random.uniform(self.brightness_factor_min,self.brightness_factor_max)) # brightness_factor 0(black) ~ 2(white)
            crop_img = F.adjust_contrast(crop_img, contrast_factor = random.uniform(self.contrast_factor_min,self.contrast_factor_max)) # contrast_factor 0(solid gray) ~ 2
            
            is_gray = random.randint(0,self.grayscale_prob) # 25% conduct gray scale
            if is_gray == 1:
                crop_img = self.greyscale(crop_img)
            else:
                crop_img = crop_img

            #Add Noise
            crop_img = crop_img + torch.randn_like(crop_img) * self.noise_std_scale
            #Blur
            crop_img = self.blurrer(crop_img)
            #perspective_warp
            crop_img, crop_ladmks = self._perspective_warp(crop_img , crop_ladmks)

            angle = random.randint(0, self.rotation_max_angle)
            crop_img = F.rotate(crop_img, angle)
            crop_ladmks = self._rotate(crop_ladmks, angle = angle)
        
        return np.array(img), ldmks, crop_img,  torch.Tensor(crop_ladmks), bbox_leftcorner
        #return np.array(img), ldmks, crop_img,  crop_ladmks, bbox_leftcorner # for test module(here)
    
    def __len__(self):
        return len(self.img_list)

    def _landmark_processing4crop(self, ldmks, bbox_leftcorner):
        #make ldmks point corresponding to cropped image
        #ldmks = ldmks - [bbox_leftcorner[0][1] ,bbox_leftcorner[0][0]] # landmarks = landmarks - [left, top] 
        ldmks = ldmks - [bbox_leftcorner[0][0] ,bbox_leftcorner[0][1]]
        return ldmks

    def _gray_scaling(self, crop_img):
        '''
        input : PIL_img
        return 3-channel gray img
        '''
        crop_img_rgb_to_grayscale = ImageOps.grayscale(crop_img)
        crop_img_rgb_to_grayscale = np.expand_dims(np.array(crop_img_rgb_to_grayscale), axis = -1)
        crop_img_rgb_to_grayscale = np.concatenate((crop_img_rgb_to_grayscale, crop_img_rgb_to_grayscale, crop_img_rgb_to_grayscale), axis = 2)
        
        return crop_img_rgb_to_grayscale

    def _rotate(self, crop_ladmks, angle = 0, imgWidth =256, imgHeight =256):
        '''
        input : array
        ouput : array
        conduct rotate
        '''
        #rotated_img = crop_img.rotate(angle)
        #crop_img = np.array(crop_img)
        rad = math.radians(angle)
        c, s = np.cos(rad), np.sin(rad)
        rot_mat = np.array(((c,-s), (s, c)))
        center = np.array([imgWidth / 2.0, imgHeight / 2.0], dtype=np.float32)
        #center -> 0 , rotate , zero center -> original center
        
        #rotated_img = cv2.warpAffine(crop_img, cv2.getRotationMatrix2D(center, angle, 1.0), (imgWidth, imgHeight))
        rotated_ladmks = np.matmul(crop_ladmks - center, rot_mat) + center

        return rotated_ladmks
    
    def _perspective_warp(self, crop_img, crop_ladmks, imgWidth =256, imgHeight =256):
        '''
        input : array
        ouput : array
    
        conduct perspective_warp
        '''

        #print(crop_img.shape)
        _, height, width = crop_img.shape
        #[x, y]
        topLeft = [0,0]
        topRight = [width -1, 0]
        bottomRight = [width -1, height - 1]
        bottomLeft = [0, height - 1]
        origin_pts = [topLeft, topRight, bottomRight, bottomLeft]

        change_range = 30
        trans_topLeft = [0 + random.randint(0, change_range), 0 + random.randint(0, change_range)]
        trans_topRight = [width - 1 - random.randint(0, change_range), 0 + random.randint(0, change_range)]
        trans_bottomRight = [width - 1 - random.randint(0, change_range), height - 1 - random.randint(0, change_range)]
        trans_bottomLeft = [0 + random.randint(0, change_range), height - 1 - random.randint(0, change_range)]
        transform_pts = [trans_topLeft, trans_topRight, trans_bottomRight, trans_bottomLeft]
        
        #print("origin_pts: ", origin_pts)
        #print("transform_pts: ", transform_pts)
        
        crop_img = F.perspective(crop_img, origin_pts, transform_pts)
        mtrx = cv2.getPerspectiveTransform(np.float32(origin_pts), np.float32(transform_pts))
        #print("mtrx.transpose():", mtrx[:2, :].shape)
        crop_ladmks = np.concatenate((crop_ladmks, np.ones((70,1))), axis = 1)
        #print("crop_ladmks.shape: ", crop_ladmks.shape)
        
        crop_ladmks = np.matmul(crop_ladmks, mtrx[:, :].transpose()) # =--> x, y, w(projection vector)
        crop_ladmks[:, 0] = crop_ladmks[:, 0] / crop_ladmks[:, 2]
        crop_ladmks[:, 1] = crop_ladmks[:, 1] / crop_ladmks[:, 2]
        #print("new crop_ladmks.shape: ", crop_ladmks.shape)
        return crop_img, crop_ladmks[:,:2]

def get_dataloader(args, IsSuffle = True, num_workers = 16, IsAug =True, train_val_ratio = 0.80):
    dataset = FaceLandMark_Loader(args.datasetPath, args, IsAug = IsAug)
    print("# of dataset:", len(dataset))

    train_dataset, valid_dataset = random_split(dataset, [int(len(dataset) * train_val_ratio), len(dataset)-int(len(dataset) * train_val_ratio)])

    print("# of train dataset:", len(train_dataset))
    print("# of valid dataset:", len(valid_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batchSize, shuffle=IsSuffle, drop_last=True, num_workers = num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batchSize, shuffle=IsSuffle, drop_last=True, num_workers = num_workers)
    return train_dataloader, valid_dataloader

def get_test_dataloader(dataroot, IsSuffle = True, num_workers = 0):
    dataset = FaceLandMark_Loader(dataroot)
    print("# of dataset:", len(dataset))

    test_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True, num_workers = num_workers)
    
    return test_dataloader

if __name__ == '__main__':
    
    root = "/data2/MS-FaceSynthetic"

    #print_all_augmented_images(root)
    dataset = FaceLandMark_Loader(root = "/data2/MS-FaceSynthetic")

    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)

    for idx, item in enumerate(data_loader):
        
        img_GT, landmark_GT, crop_img, crop_ladmks, bbox_leftcorner = item

        crop_img = crop_img.cuda()
        crop_ladmks = crop_ladmks.cuda()
        #crop_ladmks = crop_ladmks.to(device, dtype=torch.float)
        print("img_GT.shape: ", img_GT.shape)
        print("landmark_GT.shape: ", landmark_GT.shape)

        print("crop_img.shape: ", crop_img.shape)
        print("crop_ladmks.shape: ", crop_ladmks.shape)

        print("bbox_leftcorner.shape: ", bbox_leftcorner.shape)
         
        #crop_img_result = crop_img[0]cpu().numpy()
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        import utils
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--name', type=str)
        parser.add_argument('--datasetPath', type=str, default="/data2/MS-FaceSynthetic")
        parser.add_argument('--saveDir', type=str, default='/personal/GiHoonKim/face_ldmk_detection')
        args = parser.parse_args()
        saveUtils = utils.saveData(args)

        crop_ladmks = crop_ladmks.cpu()

        saveUtils.save_visualization(crop_img, crop_ladmks, crop_ladmks, 0)
        
        crop_img_result = crop_img[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        
        print("crop_img_result.shape: ", crop_img_result.shape)
        plt.imshow(crop_img_result)
        plt.scatter(crop_ladmks[0, :, 0], crop_ladmks[0, :, 1], s=10, marker='.', c='g')
        plt.savefig('data_load_sample_test.png')  

        break

    print("############ Done ############")

