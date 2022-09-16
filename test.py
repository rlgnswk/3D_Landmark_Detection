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

import AdaptationNet
import ResNet34
import utils
import data_load as data_load
import math

import argparse
parser = argparse.ArgumentParser()
#parser.add_argument('--name', type=str)
parser.add_argument('--datasetPath', type=str, default=".")
parser.add_argument('--pertrained', type=str, default='./pretrained/model_26.pt')

parser.add_argument('--saveDir', type=str, default='./test_result')
parser.add_argument('--gpu', type=str, default='0', help='gpu')

parser.add_argument('--IsGNLL', type=bool, default=False, help='using GNLL or MSE loss for training')

args = parser.parse_args()

class test_module():
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.IsGNLL = args.IsGNLL
        if self.IsGNLL == True:
            self.model =  ResNet34.ResNet34(output_param = 3).to(self.device).eval() # x, y, sigma
        else:
            self.model = ResNet34.ResNet34(output_param = 2).to(self.device).eval() # x, y
        self.model.load_state_dict(torch.load(args.pertrained))

        self.root = args.datasetPath
        self.img_dir = os.path.join(self.root, "test_image")
        self.img_list = natsort.natsorted(os.listdir(self.img_dir))

    def _save_result(self, crop_img, pred_ladmks, image_name, face_num):
        
        plt.clf()
        plt.imshow(crop_img)
        print(pred_ladmks.shape)
        plt.scatter(pred_ladmks[0, :, 0], pred_ladmks[0, : , 1], s=10, marker='.', c='g')
        #plt.savefig('valid_GT_%d.png'%(num_epoch))
        plt.savefig(self.args.saveDir + '/'+ image_name + '_face_%d.png'%(face_num))

    def _merge4final_image(self, img, results_lst, image_name):
        plt.clf()
        plt.imshow(img)

        for i in range(len(results_lst)):
            left_corner_X, left_corner_Y, pred_ladmks = results_lst[i][0], results_lst[i][1], results_lst[i][2]
            # transfer to coordinate of original image
            pred_ladmks = pred_ladmks + [left_corner_X ,left_corner_Y]
            plt.scatter(pred_ladmks[0, :, 0], pred_ladmks[0, : , 1], s=10, marker='.', c='g')
        
        plt.savefig(self.args.saveDir + '/'+ image_name + '_inference.png')

    def inference(self):
        for iter_num, img_name in enumerate(self.img_list):

            results_lst = []

            img_path = self.img_dir + '/' + self.img_list[iter_num]
            img = Image.open(img_path)
            
            #size check
            width, height = img.size
            assert  width <= 512 or height <= 512, "Test image is too small for this module. It should be bigger thant 512 x 512"
            
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


def test(args):

    #gpu
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.multiprocessing.set_start_method('spawn') # for using mutli num_workers

    test_class = test_module(args, device)
    
    _ = test_class.inference()

    print("######### Check your result #########")
    print("######### Test Done #########")
    
    # inference 함수로 옮겨야함.
    
if __name__ == "__main__":
    test(args)