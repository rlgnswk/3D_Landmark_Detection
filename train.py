import torch
import torch.nn as nn
import torch.optim as optim   
import random
import os
from torch.utils.tensorboard import SummaryWriter

import AdaptationNet
import ResNet34
import utils
import data_load as data_load
import math

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str)
parser.add_argument('--datasetPath', type=str, default="/data2/MS-FaceSynthetic")
parser.add_argument('--saveDir', type=str, default='/personal/GiHoonKim/face_ldmk_detection')
parser.add_argument('--gpu', type=str, default='0', help='gpu')


parser.add_argument('--numEpoch', type=int, default=120, help='# of epoch')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size for training')
parser.add_argument('--lr_landmark', type=float, default=0.001, help='learning rate')
#parser.add_argument('--lr_adaptation', type=float, default=0.001, help='learning rate')
parser.add_argument('--print_interval', type=int, default=100, help='print interval')

parser.add_argument('--IsGNLL', type=bool, default=False, help='using GNLL or MSE loss for training')
parser.add_argument('--IsAug', type=bool, default=True, help='conduct augmentation of not')

args = parser.parse_args()

def main(args):
    #gpu
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.multiprocessing.set_start_method('spawn') # for using mutli num_workers

    #util
    saveUtils = utils.saveData(args)
    saveUtils.save_log(str(args))
    writer = SummaryWriter(saveUtils.save_dir_tensorBoard)

    #model
    if args.IsGNLL == True:
        model4Landmark = ResNet34.ResNet34(output_param = 3).to(device) # x, y, sigma
        # https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html # output = loss(input, target, var)
        lossFunction = nn.GaussianNLLLoss()
    else:
        model4Landmark = ResNet34.ResNet34(output_param = 2).to(device) # x, y
        lossFunction = nn.MSELoss()
    #model4adaptation = AdaptationNet.AdaptationNet()
    
    # optimizer
    optimizer4landmark = torch.optim.Adam(model4Landmark.parameters(), lr=args.lr_landmark)
    #optimizer4adaptation = torch.optim.Adam(model4adaptation.parameters(), lr=args.lr_adaptation)
    
    # data loader
    train_dataloader, valid_dataloader = data_load.get_dataloader(args.datasetPath , args.batchSize, IsAug = args.IsAug)
    
    print_train_loss = 0
    print_val_loss = 0 
    print_interval = 10
    for num_epoch in range(args.numEpoch):
        for iter_num, item in enumerate(train_dataloader):
            #print(iter_num)
            img_GT, landmark_GT, crop_img, crop_ladmks, bbox_leftcorner = item
            
            crop_img = crop_img.to(device, dtype=torch.float)
            crop_ladmks = crop_ladmks.to(device, dtype=torch.float)
        
            pred_ladmks = model4Landmark(crop_img)
            
            #print("pred_ladmks[args.batchSize, :2].shape: ", pred_ladmks[:, :, :2].reshape(args.batchSize, -1 ,2).shape)
            #print("crop_ladmks.shape: ", crop_ladmks.shape)
            if args.IsGNLL == True:
                pred_ladmks = pred_ladmks.reshape(args.batchSize, -1 ,3)# x, y, sigma
                #Paper: Rather than directly outputting σ, we predict log σ, and take its exponential to ensure σ is positive
                #torch.pow(torch.log(torch.nn.functional.relu(pred_ladmks[:,:,2]) + 1e-10)) # add 1e-10 for non-zero log input
                train_loss = lossFunction(crop_ladmks, pred_ladmks[:, :, :2], torch.nn.functional.relu(pred_ladmks[:,:,2]).add_(1e-10))
            else:
                pred_ladmks = pred_ladmks.reshape(args.batchSize, -1 ,2)# x, y
                train_loss = lossFunction(crop_ladmks, pred_ladmks)
            
            print_train_loss += train_loss.item()

            optimizer4landmark.zero_grad()
            train_loss.backward()
            optimizer4landmark.step()

            #print and logging
            if iter_num % print_interval == 0:
                print_train_loss = print_train_loss/print_interval
                log = "Train: [Epoch %d][Iter %d] [Train Loss: %.4f]" % (num_epoch, iter_num, print_train_loss)
                print(log)
                saveUtils.save_log(log)
                writer.add_scalar("Train Loss/ iter", print_train_loss, iter_num)
                print_train_loss = 0
            
        #validation
        model4Landmark.eval()
        for iter, item in enumerate(valid_dataloader):
            
            img_GT, landmark_GT, crop_img, crop_ladmks, bbox_leftcorner = item
        
            crop_img = crop_img.to(device, dtype=torch.float)
            crop_ladmks = crop_ladmks.to(device, dtype=torch.float)

            with torch.no_grad():
                pred_ladmks = model4Landmark(crop_img)
            if args.IsGNLL == True:
                pred_ladmks = pred_ladmks.reshape(args.batchSize, -1 ,3)# x, y, sigma
                print_val_loss += lossFunction(crop_ladmks, pred_ladmks[:, :, :2], torch.nn.functional.relu(pred_ladmks[:,:,2]).add_(1e-10)).item()
            else:
                pred_ladmks = pred_ladmks.reshape(args.batchSize, -1 ,2)# x, y
                print_val_loss += lossFunction(crop_ladmks, pred_ladmks).item()
        
        model4Landmark.train()
        #print, logging, save model per epoch 
        print_val_loss = print_val_loss/len(valid_dataloader)
        log = "Valid: [Epoch %d] [Valid Loss: %.4f]" % (num_epoch, print_val_loss)
        print(log)
        saveUtils.save_log(log)
        writer.add_scalar("Valid Loss/ Epoch", print_val_loss, num_epoch)
        saveUtils.save_model(model4Landmark, num_epoch)
        if args.IsGNLL == True:
            saveUtils.save_visualization(crop_img, crop_ladmks, pred_ladmks[:, :, :2], num_epoch)
        else:
            saveUtils.save_visualization(crop_img, crop_ladmks, pred_ladmks, num_epoch)
        print_val_loss = 0

if __name__ == "__main__":
    main(args)