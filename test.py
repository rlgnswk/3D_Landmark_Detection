import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import random
import os
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

import natsort
from retinaface import RetinaFace
import matplotlib.pyplot as plt

import moblieNetV2
import AdaptationNet
import ResNet34
import utils
import data_load as data_load
import math

class test_module():
    def __init__(self, datasetPath = None, pertrained = './pretrained/model_26.pt', saveDir = './test_result', IsGNLL = False, modelType = 'ResNet34'):
        
        self.device  = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if datasetPath is not None:
            self.datasetPath = datasetPath
            self.root = self.datasetPath
            self.img_dir = os.path.join(self.root, "test_image")
            self.img_list = natsort.natsorted(os.listdir(self.img_dir))
        self.pertrained = pertrained
        self.saveDir = saveDir
        self.modelType = modelType
        self.IsGNLL = IsGNLL
        
        if self.IsGNLL == True:
            self.model =  ResNet34.ResNet34(output_param = 3).to(self.device).eval() # x, y, sigma
        else:
            self.model = ResNet34.ResNet34(output_param = 2).to(self.device).eval() # x, y
        self.model.load_state_dict(torch.load(args.pertrained))

        #model
        if self.modelType == "ResNet34":
            if self.IsGNLL == True:
                model4Landmark = ResNet34.ResNet34(output_param = 3).to(self.device) # x, y, sigma
            else:
                model4Landmark = ResNet34.ResNet34(output_param = 2).to(self.device) # x, y
        elif self.modelType == "MoblieNetv2":
            if self.IsGNLL == True:
                model4Landmark = moblieNetV2.moblieNetV2(output_param = 3).to(self.device) # x, y, sigma
            else:
                model4Landmark = moblieNetV2.moblieNetV2(output_param = 2).to(self.device) # x, y
        else:
            print("There is no proper model type.")
            raise ValueError

    def _save_result(self, crop_img, pred_ladmks, image_name, face_num):
        
        plt.clf()
        plt.imshow(crop_img)
        print(pred_ladmks.shape)
        plt.scatter(pred_ladmks[0, :, 0], pred_ladmks[0, : , 1], s=10, marker='.', c='g')
        #plt.savefig('valid_GT_%d.png'%(num_epoch))
        plt.savefig(self.saveDir + '/'+ image_name + '_face_%d.png'%(face_num))

    def _merge4final_image(self, img, results_lst, image_name):
        plt.clf()
        plt.imshow(img)

        for i in range(len(results_lst)):
            left_corner_X, left_corner_Y, pred_ladmks = results_lst[i][0], results_lst[i][1], results_lst[i][2]
            # transfer to coordinate of original image
            pred_ladmks = pred_ladmks + [left_corner_X ,left_corner_Y]
            plt.scatter(pred_ladmks[0, :, 0], pred_ladmks[0, : , 1], s=10, marker='.', c='g')
        
        plt.savefig(self.saveDir + '/'+ image_name + '_inference.png')
    

    def inference_imgFolder(self, img_dir):
        img_list = natsort.natsorted(os.listdir(img_dir))
        result_total_lst = []

        for iter_num, img_name in enumerate(img_list):

            img_path = img_dir + '/' + img_list[iter_num]
            results_lst = self.inference_imgPath(img_path)
            result_total_lst.append(results_lst)

        return result_total_lst
     
    def inference_imgPath(self, img_path):
        results_lst = []
        img = Image.open(img_path)
        img = img.resize((512, 512)) ##  확인 필요 img pre processing에서 얼굴이 다 담겨야하는데 그렇지않음.
        #size check
        width, height = img.size
        print("test image width : ", width)
        print("test image height : ", height)
        assert  width >= 512 or height >= 512, "Test image is too small for this module. It should be bigger thant 512 x 512"
        img.save(img_path)
        # face detection
        resp = RetinaFace.detect_faces(img_path = img_path)
        # detection check
        assert len(resp) != 0, "Can not detect face"
        
        # Conduct inference up to the number of detected faces
        for i in range(1, len(resp) + 1):
            try:
                if resp["face_"+str(i)]['facial_area'][0] + 256 >= 512:
                    resp["face_"+str(i)]['facial_area'][0] = 255
                if resp["face_"+str(i)]['facial_area'][1] + 256 >= 512:
                    resp["face_"+str(i)]['facial_area'][1] = 255
                #Usually, the width of the bounding box is smaller than 256. 
                #Thus, shift X coordinate to proper position for 256x256 bounding box
                X_extra = (256 - (resp["face_"+str(i)]['facial_area'][2] - resp["face_"+str(i)]['facial_area'][0]))//2
    
                left_corner_X = resp["face_"+str(i)]['facial_area'][0]- X_extra
                left_corner_Y = resp["face_"+str(i)]['facial_area'][1]

            except TypeError:
                #When there is any error, just assume that bounding box is placed in center of the image
                print("detection error occur")
                left_corner_X = 128
                left_corner_Y = 128

            #crop image
            crop_img = img.crop((left_corner_X, left_corner_Y, left_corner_X + 256, left_corner_Y + 256))
            
            #inference
            pred_ladmks = self.inference_img(crop_img)

            #save individual image
            crop_img = crop_img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            pred_ladmks = pred_ladmks.reshape(1, -1 ,2).cpu().numpy()
            self._save_result(crop_img, pred_ladmks, self.img_list[iter_num], i)
            results_lst.append([left_corner_X, left_corner_Y, pred_ladmks])
        
        #merge with overall image
        self._merge4final_image(img, results_lst, self.img_list[iter_num])
        print("######### One inference is completed #########")
        return results_lst
    
    def inference_img(self, crop_img)
        '''
        crop_img => PIL image
        '''
        width, height = crop_img.size
        assert  width == 256 and height == 256, "The size of input image must be 256x256 but it is not"
        crop_img = torchvision.transforms.ToTensor()(crop_img).to(self.device)
        with torch.no_grad():
            pred_ladmks = self.model(crop_img.unsqueeze(0))
        
        return pred_ladmks

    def inference(self):
        for iter_num, img_name in enumerate(self.img_list):

            results_lst = []

            img_path = self.img_dir + '/' + self.img_list[iter_num]
            img = Image.open(img_path)
            img = img.resize((512, 512))
            
            #size check
            width, height = img.size
            print("test image width : ", width)
            print("test image height : ", height)
            assert  width >= 512 or height >= 512, "Test image is too small for this module. It should be bigger thant 512 x 512"
            img.save(img_path)
            # face detection
            resp = RetinaFace.detect_faces(img_path = img_path)
            # detection check
            assert len(resp) != 0, "Can not detect face"
            
            # Conduct inference up to the number of detected faces
            for i in range(1, len(resp) + 1):
                try:
                    if resp["face_"+str(i)]['facial_area'][0] + 256 >= 512:
                        resp["face_"+str(i)]['facial_area'][0] = 255
                    if resp["face_"+str(i)]['facial_area'][1] + 256 >= 512:
                        resp["face_"+str(i)]['facial_area'][1] = 255
                    
                    #Usually, the width of the bounding box is smaller than 256. 
                    #Thus, shift X coordinate to proper position for 256x256 bounding box
                    X_extra = (256 - (resp["face_"+str(i)]['facial_area'][2] - resp["face_"+str(i)]['facial_area'][0]))//2
                    
                    left_corner_X = resp["face_"+str(i)]['facial_area'][0]- X_extra
                    left_corner_Y = resp["face_"+str(i)]['facial_area'][1]

                except TypeError:
                    #When there is any error, just assume that bounding box is placed in center of the image
                    print("detection error occur")
                    left_corner_X = 128
                    left_corner_Y = 128

                #crop image
                crop_img = img.crop((left_corner_X, left_corner_Y, left_corner_X + 256, left_corner_Y + 256))
                crop_img = torchvision.transforms.ToTensor()(crop_img).to(self.device)
                
                #inference
                with torch.no_grad():
                    pred_ladmks = self.model(crop_img.unsqueeze(0))

                #save individual image
                
                crop_img = crop_img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                pred_ladmks = pred_ladmks.reshape(1, -1 ,2).cpu().numpy()
                self._save_result(crop_img, pred_ladmks, self.img_list[iter_num], i)
                results_lst.append([left_corner_X, left_corner_Y, pred_ladmks])
            
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
parser.add_argument('--datasetPath', type=str, default=".")
parser.add_argument('--pertrained', type=str, default='./pretrained/model_99.pt')

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
    
    _ = test_class.inference()

    print("######### Check your result #########")
    print("######### Test Done #########")
    
    # inference 함수로 옮겨야함.
    
if __name__ == "__main__":
    test(args)