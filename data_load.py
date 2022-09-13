import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch
import torch.nn as nn
#import torch.nn.functional as F
import torchvision.transforms.functional as F
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
    def __init__(self, root, IsAug = True):
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

        #print(self.img_list[:10])
        #print(self.ldmks_list[:10])
         
        #using augmentation functions which do not effect on landmarks position 
    def __getitem__(self, idx):
        
        # get an image
        # get a landmark info
        # get a bbox info
        img = Image.open(self.img_path + '/' + self.img_list[idx])
        
        ldmks = pandas.read_csv(self.ldmks_path +'/'+self.ldmks_list[idx],  header=None, sep=' ')
        ldmks = np.asarray(ldmks) # shape : (70, 2), last two row for centers of eyes

        bbox_leftcorner = pandas.read_csv(self.bbox_leftcorner_coord_path +'/'+self.bbox_list[idx],  header=None, sep=' ')
        bbox_leftcorner = np.asarray(bbox_leftcorner) # shape : (2, 1) # x, y
        
        #crop image 256x256 including face info --> how? # crop이 가장 위에 가야지 계산 효율성이 좋을 듯!
        crop_img = img.crop((bbox_leftcorner[0][0], bbox_leftcorner[0][1], bbox_leftcorner[0][0]+ 256, bbox_leftcorner[0][1] + 256))
        crop_ladmks = self._landmark_processing4crop(ldmks, bbox_leftcorner)

        '''
        Paper:
        During training, we perform data augmentation including
        rotations, perspective warps, < -- landmark also should be changed 
        blurs, modulations to brightness and contrast, addition of noise, and conversion to grayscale. < -- does not be related with landmark
        '''

        #blurs, modulations to brightness and contrast, addition of noise, and conversion to grayscale. < -- does not be related with landmark
        crop_img = F.adjust_brightness(crop_img, brightness_factor = random.uniform(0.5,1.5)) # brightness_factor 0(black) ~ 2(white)
        crop_img = F.adjust_contrast(crop_img, contrast_factor = random.uniform(0.5,1.5)) # contrast_factor 0(solid gray) ~ 2
        
        #crop_img = F.rgb_to_grayscale(crop_img, num_output_channels =3)
        is_gray = random.randint(0,1)
        if is_gray == True:
            self._gray_scaling(crop_img)
        else:
            crop_img = np.array(crop_img)

        #Add Noise
        #crop_img = F.gaussian_blur(crop_img, kernel_size = random.randint(3, 7))
        crop_img = np.array(crop_img) + (np.random.randn(256, 256, 3) * random.randint(0, 5))
        crop_img = np.clip(crop_img, 0, 255).astype(np.uint8)

        #Blur
        #crop_img = crop_img + torch.randn(crop_img.size()) * self.std + self.mean  
        crop_img = cv2.GaussianBlur(crop_img, (0, 0), 0.1) # array input

        #rotations, perspective warps, < -- landmark also should be changed 
        crop_img, crop_ladmks = self._rotate(crop_img , crop_ladmks, angle = random.randint(0, 90))
        crop_img, crop_ladmks = self._perspective_warp(crop_img , crop_ladmks, beta = 0.5)
        #conduct augmentation --> how handle the annotation simultaneously??

        return np.array(img), ldmks, crop_img, crop_ladmks, bbox_leftcorner

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

    def _rotate(self, crop_img, crop_ladmks, angle = 0, imgWidth =256, imgHeight =256):
        '''
        input : array
        ouput : array

        conduct rotate
        '''
        #rotated_img = crop_img.rotate(angle)
        crop_img = np.array(crop_img)
        rad = math.radians(angle)
        c, s = np.cos(rad), np.sin(rad)
        rot_mat = np.array(((c,-s), (s, c)))
        center = np.array([imgWidth / 2.0, imgHeight / 2.0], dtype=np.float32)
        #center -> 0 , rotate , zero center -> original center
        
        rotated_img = cv2.warpAffine(crop_img, cv2.getRotationMatrix2D(center, angle, 1.0), (imgWidth, imgHeight))
        rotated_ladmks = np.matmul(crop_ladmks - center, rot_mat) + center

        return rotated_img, rotated_ladmks
    
    def _perspective_warp(self, crop_img, crop_ladmks, beta = 0.0, imgWidth =256, imgHeight =256):
        '''
        input : array
        ouput : array
    
        conduct perspective_warp
        '''
        crop_img = np.array(crop_img)
        cX = random.uniform(-beta, beta)
        cY = random.uniform(-beta, beta)
        
        shearMat = np.array([[1., cX, 0.], [cY, 1., 0.]], dtype=np.float32)
        crop_img = cv2.warpAffine(crop_img, shearMat, (imgWidth, imgHeight))
        crop_ladmks = np.matmul(crop_ladmks, (shearMat[:, :2]).transpose())
        
        return crop_img, crop_ladmks

def get_dataloader(dataroot, batch_size, IsSuffle = True):
    dataset = MotionLoader(dataroot)
    print("# of dataset:", len(dataset))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=IsSuffle, drop_last=True)
   
    return dataloader, dataset


if __name__ == '__main__':
    
    root = "/data2/MS-FaceSynthetic"

    #print_all_augmented_images(root)
    dataset = FaceLandMark_Loader(root = "/data2/MS-FaceSynthetic")

    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)

    for idx, item in enumerate(data_loader):
        
        img_GT, landmark_GT, crop_img, crop_ladmks, bbox_leftcorner = item

        print("img_GT.shape: ", img_GT.shape)
        print("landmark_GT.shape: ", landmark_GT.shape)

        print("crop_img.shape: ", crop_img.shape)
        print("crop_ladmks.shape: ", crop_ladmks.shape)

        print("bbox_leftcorner.shape: ", bbox_leftcorner.shape)

        plt.imshow(crop_img[0])
        plt.scatter(crop_ladmks[0, :, 0], crop_ladmks[0, :, 1], s=10, marker='.', c='g')
        plt.savefig('data_load_sample.png')  

        break
    print("############ Done ############")


    




def print_all_augmented_images(root = "/data2/MS-FaceSynthetic"):
    '''
    Test function for printing resutls of all augment functions
    '''
    idx = 0
    root = "/data2/MS-FaceSynthetic"
    temp = FaceLandMark_Loader(root = "/data2/MS-FaceSynthetic")
    
    img_path = os.path.join(root, "img")
    img_list = natsort.natsorted(os.listdir(img_path))
    img = Image.open(img_path + '/' + img_list[idx])

    ldmks_list = natsort.natsorted(os.listdir(os.path.join(root, "ldmks")))
    ldmks = pandas.read_csv(os.path.join(root, "ldmks") +'/'+ldmks_list[idx],  header=None, sep=' ')
    ldmks = np.asarray(ldmks)
    
    bbox_leftcorner_coord_path = os.path.join(root, "bbox_leftcorner_coord")
    bbox_list = natsort.natsorted(os.listdir(bbox_leftcorner_coord_path))
    bbox_leftcorner = pandas.read_csv(bbox_leftcorner_coord_path +'/'+ bbox_list[idx],  header=None, sep=' ')
    bbox_leftcorner = np.asarray(bbox_leftcorner) # shape : (2, 1) # x, y

    crop_img = img.crop((bbox_leftcorner[0][0], bbox_leftcorner[0][1], bbox_leftcorner[0][0]+ 256, bbox_leftcorner[0][1] + 256))
    crop_ladmks = temp._landmark_processing4crop(ldmks, bbox_leftcorner)

    # with eye points: 70
    print("ldmks[:, 0].shape: ",ldmks[:, 0].shape)
    print("crop_img.shape: ", np.array(crop_img).shape)
    print(crop_img)
    plt.imshow(crop_img)
    plt.scatter(crop_ladmks[:, 0], crop_ladmks[:, 1], s=10, marker='.', c='g')
    plt.savefig('crop_img.png')  
        
    plt.clf()
    crop_img_adjust_brightness = F.adjust_brightness(crop_img, brightness_factor = random.uniform(0.5,1.5)) # brightness_factor 0(black) ~ 2(white)
    plt.imshow(crop_img_adjust_brightness)
    plt.scatter(crop_ladmks[:, 0], crop_ladmks[:, 1], s=10, marker='.', c='g')
    plt.savefig('crop_img_adjust_brightness.png')

    plt.clf()
    crop_img_adjust_contrast = F.adjust_contrast(crop_img, contrast_factor = random.uniform(0.5,1.5)) # contrast_factor 0(solid gray) ~ 2
    plt.imshow(crop_img_adjust_contrast)
    plt.scatter(crop_ladmks[:, 0], crop_ladmks[:, 1], s=10, marker='.', c='g')
    plt.savefig('crop_img_adjust_contrast.png')

    plt.clf()
    crop_img_gaussian_blur = crop_img.filter(ImageFilter.GaussianBlur(radius = random.randint(3, 7)))
    #crop_img_gaussian_blur = F.GaussianBlur(crop_img, kernel_size = random.randint(3, 7))
    plt.imshow(crop_img_gaussian_blur)
    plt.scatter(crop_ladmks[:, 0], crop_ladmks[:, 1], s=10, marker='.', c='g')
    plt.savefig('crop_img_gaussian_blur.png')
    
    plt.clf()
    #crop_img_rgb_to_grayscale = F.rgb_to_grayscale(crop_img, num_output_channels =3)
    crop_img_rgb_to_grayscale = ImageOps.grayscale(crop_img)
    print("crop_img_rgb_to_grayscale.shape: ", np.array(crop_img_rgb_to_grayscale).shape)
    crop_img_rgb_to_grayscale = np.expand_dims(np.array(crop_img_rgb_to_grayscale), axis = -1)
    print("crop_img_rgb_to_grayscale.shape: ", np.array(crop_img_rgb_to_grayscale).shape)
    crop_img_rgb_to_grayscale = np.concatenate((crop_img_rgb_to_grayscale, crop_img_rgb_to_grayscale, crop_img_rgb_to_grayscale), axis = 2)
    print("crop_img_rgb_to_grayscale.shape: ", np.array(crop_img_rgb_to_grayscale).shape)
    #plt.imshow(crop_img_rgb_to_grayscale, cmap='gray')
    plt.imshow(crop_img_rgb_to_grayscale)
    plt.scatter(crop_ladmks[:, 0], crop_ladmks[:, 1], s=10, marker='.', c='g')
    plt.savefig('crop_img_rgb_to_grayscale.png')
    
    plt.clf()
    crop_img_noise = np.array(crop_img) + (np.random.randn(256, 256, 3) * 10  + 0.0)
    #print(crop_img_noise[:10])
    crop_img_noise = np.clip(crop_img_noise, 0, 255).astype(np.uint8)
    plt.imshow(Image.fromarray(crop_img_noise), vmin=0, vmax=255)
    plt.scatter(crop_ladmks[:, 0], crop_ladmks[:, 1], s=10, marker='.', c='g')
    plt.savefig('crop_img_noise.png')

    plt.clf()
    rot_crop_img, rot_crop_ladmks = temp._rotate(crop_img , crop_ladmks, angle = random.randint(0,359))
    plt.imshow(rot_crop_img)
    plt.scatter(rot_crop_ladmks[:, 0], rot_crop_ladmks[:, 1], s=10, marker='.', c='g')
    plt.savefig('crop_img_rotate.png')

    plt.clf()
    warp_crop_img, warp_crop_ladmks = temp._perspective_warp(crop_img , crop_ladmks, beta = 1.0)
    plt.imshow(warp_crop_img)
    plt.scatter(warp_crop_ladmks[:, 0], warp_crop_ladmks[:, 1], s=10, marker='.', c='g')
    plt.savefig('crop_img_warp.png')

    plt.clf()
    augment_image = F.adjust_brightness(crop_img, brightness_factor = random.uniform(0.5,1.5))
    augment_image = F.adjust_contrast(augment_image, contrast_factor = random.uniform(0.5,1.5))
    
    #augment_image = augment_image.filter(ImageFilter.GaussianBlur(radius = random.randint(3, 5)))
    augment_image = cv2.GaussianBlur(np.array(augment_image), (0, 0), 0.1)
    
    augment_image = np.array(augment_image) + (np.random.randn(256, 256, 3) * random.randint(0, 5)  + 0.0)
    #print(crop_img_noise[:10])
    augment_image = np.clip(augment_image, 0, 255).astype(np.uint8)
    augment_image = Image.fromarray(augment_image)

    augment_image, augment_ladmks = temp._rotate(augment_image , crop_ladmks, angle = random.randint(0, 90))
    augment_image, augment_ladmks = temp._perspective_warp(augment_image , augment_ladmks, beta = 0.5)

    plt.imshow(augment_image)
    plt.scatter(augment_ladmks[:, 0], augment_ladmks[:, 1], s=10, marker='.', c='g')
    plt.savefig('multi-augmented_image.png')