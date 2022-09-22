import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import random
import os
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import natsort
from retinaface import RetinaFace
import matplotlib.pyplot as plt

import moblieNetV2
import resNet34
import utils
import data_load as data_load
import math
import numpy as np

class test_module():
    def __init__(self, datasetPath = None, pertrained = './pretrained/resNet_GNLL_120epoch.pt', saveDir = './test_result', IsGNLL = False, modelType = 'ResNet34'):
        
        self.device  = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if datasetPath is not None:
            self.img_dir = datasetPath
            self.img_list = natsort.natsorted(os.listdir(self.img_dir))
        self.pertrained = pertrained
        self.saveDir = saveDir
        self.modelType = modelType
        self.IsGNLL = IsGNLL
        
        #model
        if self.modelType == "ResNet34":
            if self.IsGNLL == True:
                self.model = resNet34.ResNet34(output_param = 3).to(self.device).eval() # x, y, sigma
            else:
                self.model = resNet34.ResNet34(output_param = 2).to(self.device).eval() # x, y
        elif self.modelType == "MoblieNetv2":
            if self.IsGNLL == True:
                self.model = moblieNetV2.moblieNetV2(output_param = 3).to(self.device).eval() # x, y, sigma
            else:
                self.model = moblieNetV2.moblieNetV2(output_param = 2).to(self.device).eval() # x, y
        else:
            print("There is no proper model type.")
            raise ValueError

        self.model.load_state_dict(torch.load(args.pertrained))

    def _save_result(self, crop_img, pred_ladmks, image_name, face_num):
        plt.clf()
        plt.imshow(crop_img)
        #print(pred_ladmks.shape)
        plt.scatter(pred_ladmks[0, :, 0], pred_ladmks[0, : , 1], s=10, marker='.', c='g')
        plt.savefig(self.saveDir + '/'+ image_name + '_face_%d.png'%(face_num))

    def _save_result_std(self, crop_img, pred_ladmks, image_name, face_num):
        cm = plt.cm.get_cmap('RdYlBu')
        plt.clf()
        plt.imshow(crop_img)
        #print(pred_ladmks.shape)
        plt.colorbar(plt.scatter(pred_ladmks[0, :, 0], pred_ladmks[0, : , 1], s=10, marker='.', c=np.exp(pred_ladmks[0, : , 2]), cmap=cm))
        plt.savefig(self.saveDir + '/'+ image_name + '_face_%d.png'%(face_num))

    def _merge4final_image(self, img, results_lst, image_name):
        
        plt.clf()
        plt.imshow(img)
        for i in range(len(results_lst)):
            left_corner_X, left_corner_Y, pred_ladmks, ratio = results_lst[i][0], results_lst[i][1], results_lst[i][2], results_lst[i][3]
            # transfer to coordinate of original image
            pred_ladmks[:, :, :2] = pred_ladmks[:, :, :2] * ratio + np.array([left_corner_X ,left_corner_Y]) 
            print(pred_ladmks.shape)
            plt.scatter(pred_ladmks[0, :, 0], pred_ladmks[0, : , 1], s=10, marker='.', c='g')
        
        plt.savefig(self.saveDir + '/'+ image_name + '_inference.png')
    
    def inference_imgFolder(self, img_dir):
        '''
        img_dir: image directory(multi-images)
        inference from image directory
        '''
        img_list = natsort.natsorted(os.listdir(img_dir))
        result_total_lst = []

        for iter_num, img_name in enumerate(img_list):

            img_path = img_dir + '/' + img_list[iter_num]
            results_lst = self.inference_imgPath(img_path)
            result_total_lst.append(results_lst)

        return result_total_lst
     
    def inference_imgPath(self, img_path):
        '''
        img_path: image path
        inference from image path
        '''
        results_lst = []
        img = Image.open(img_path)
        
        #size check
        width, height = img.size
        
        # face detection
        resp = RetinaFace.detect_faces(img_path = img_path)
        # detection check
        assert len(resp) != 0, "Can not detect face"
        
        # Conduct inference up to the number of detected faces
        for i in range(1, len(resp) + 1):
            try:
                detect_w = resp["face_1"]['facial_area'][2] - resp["face_1"]['facial_area'][0]
                detect_h = resp["face_1"]['facial_area'][3] - resp["face_1"]['facial_area'][1]
                max_side = max(detect_w, detect_h)
                # make the crop ROI as square followed by longer side
                if detect_w <= detect_h:
                    left_corner_X = resp["face_"+str(i)]['facial_area'][0] -  ((detect_h - detect_w) //2)
                    left_corner_Y = resp["face_"+str(i)]['facial_area'][1]
                    ratio = detect_h/256
                else:
                    left_corner_X = resp["face_"+str(i)]['facial_area'][0] 
                    left_corner_Y = resp["face_"+str(i)]['facial_area'][1]-  ((detect_w - detect_h) //2)
                    ratio = detect_w/256
            except TypeError:
                #When there is any error, just assume that bounding box is placed in center of the image
                print("detection error occur: There is certain case which can not proper alginmnet")
                raise ValueError

            #crop image
            crop_img = img.crop((left_corner_X, left_corner_Y, left_corner_X + max_side, left_corner_Y + max_side))
            crop_img = crop_img.resize((256, 256))

            #inference
            pred_ladmks = self.inference_img(crop_img)
            pred_ladmks[ :, :2] = pred_ladmks[ :, :2] * ratio + np.array([left_corner_X ,left_corner_Y]) 
            results_lst.append([pred_ladmks])
            
        return results_lst
    
    def inference_img(self, crop_img):
        '''
        crop_img: PIL image

        inference input img
        '''
        width, height = crop_img.size
        assert  width == 256 and height == 256, "The size of input image must be 256x256 but it is not"
        crop_img = torchvision.transforms.ToTensor()(crop_img).to(self.device)
        with torch.no_grad():
            pred_ladmks = self.model(crop_img.unsqueeze(0)).squeeze(0)
            #print(pred_ladmks.shape)
        if self.IsGNLL == True:
            return pred_ladmks.reshape(-1 ,3).cpu().numpy()
        else:
            return pred_ladmks.reshape(-1 ,2).cpu().numpy()

    def inference(self):
        for iter_num, img_name in enumerate(self.img_list):

            results_lst = []
            img_path = self.img_dir + '/' + self.img_list[iter_num]
            img = Image.open(img_path)

            # face detection
            resp = RetinaFace.detect_faces(img_path = img_path)
            # detection check
            assert len(resp) != 0, "Can not detect face"
            
            # Conduct inference up to the number of detected faces & align
            for i in range(1, len(resp) + 1):
                try:
                    detect_w = resp["face_1"]['facial_area'][2] - resp["face_1"]['facial_area'][0]
                    detect_h = resp["face_1"]['facial_area'][3] - resp["face_1"]['facial_area'][1]
                    max_side = max(detect_w, detect_h)
                    # make the crop ROI as square followed by longer side
                    if detect_w <= detect_h:
                        left_corner_X = resp["face_"+str(i)]['facial_area'][0] - ((detect_h - detect_w) //2)
                        left_corner_Y = resp["face_"+str(i)]['facial_area'][1]
                        ratio = detect_h/256
                    else:
                        left_corner_X = resp["face_"+str(i)]['facial_area'][0] 
                        left_corner_Y = resp["face_"+str(i)]['facial_area'][1] - ((detect_w - detect_h) //2)
                        ratio = detect_w/256
                except TypeError:
                    #When there is any error, just assume that bounding box is placed in center of the image
                    print("detection error occur: There is certain case which can not proper alginmnet")
                    raise ValueError

                #crop image
                crop_img = img.crop((left_corner_X, left_corner_Y, left_corner_X + max_side, left_corner_Y + max_side))
                crop_img = crop_img.resize((256, 256))
                crop_img = torchvision.transforms.ToTensor()(crop_img).to(self.device)
                
                #inference
                with torch.no_grad():
                    pred_ladmks = self.model(crop_img.unsqueeze(0))

                #save individual image
                crop_img = crop_img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

                if self.IsGNLL == True:
                    pred_ladmks = pred_ladmks.reshape(1, -1 ,3).cpu().numpy()
                    self._save_result_std(crop_img, pred_ladmks, self.img_list[iter_num], i)
                else:
                    pred_ladmks = pred_ladmks.reshape(1, -1 ,2).cpu().numpy()
                    self._save_result(crop_img, pred_ladmks, self.img_list[iter_num], i)

                results_lst.append([left_corner_X, left_corner_Y, pred_ladmks, ratio])
            
            #merge with overall image
            self._merge4final_image(img, results_lst, self.img_list[iter_num])
            print("######### One inference is completed #########")

        return results_lst

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


import argparse
parser = argparse.ArgumentParser()
#parser.add_argument('--name', type=str)
parser.add_argument('--datasetPath', type=str, default="./test_image")
parser.add_argument('--pertrained', type=str, default='./pretrained/resNet_GNLL_120epoch.pt')

parser.add_argument('--saveDir', type=str, default='./test_result')
parser.add_argument('--gpu', type=str, default='0', help='gpu')

parser.add_argument('--IsGNLL', type=str2bool, default=False, help='using GNLL or MSE loss for training')
parser.add_argument('--modelType', type=str, default='ResNet34')
args = parser.parse_args()

def test(args):

    #gpu
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    
    torch.multiprocessing.set_start_method('spawn') # for using mutli num_workers

    test_class = test_module(datasetPath = args.datasetPath , pertrained = args.pertrained, saveDir = args.saveDir, IsGNLL = args.IsGNLL, modelType = args.modelType)
    
    #_ = test_class.inference()

    #test code level 

    '''info_1 = test_class.inference_imgFolder(args.datasetPath)
    print("info_1: ", info_1)
    info_2 = test_class.inference_imgPath("/root/landmark_detection/test_image/000000.png")
    print("info_2: ", info_2)
    path = "/root/landmark_detection/test_image/000000.png"
    img = Image.open(path)
    img = img.resize((256, 256))
    info_3 = test_class.inference_img(img)
    print("info_3: ", info_3)
    
    print("######### Check your result #########")
    print("######### Test Done #########")'''

    #visual test

    path = "/root/landmark_detection/test_image/FFHQ00002.png"
    img = Image.open(path)
    img = img.resize((256, 256))

    info_3 = test_class.inference_img(img)
    #MSE model
    test_class._save_result(img, np.expand_dims(info_3, axis = 0), "_save_result_std", 0)
    #GNLL model
    test_class._save_result_std(img, np.expand_dims(info_3, axis = 0), "_save_result_std", 0)
    
    
if __name__ == "__main__":
    test(args)