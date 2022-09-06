import torch.optim as optim   
import random
import os
from torch.utils.tensorboard import SummaryWriter

import AdaptationNet
import ResNet34
import utils
import data_load
import visualization

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str)
parser.add_argument('--datasetPath', type=str, default='/input/MotionInfillingData/train_data')
parser.add_argument('--ValdatasetPath', type=str, default='/input/MotionInfillingData/valid_data')
parser.add_argument('--saveDir', type=str, default='/personal/GiHoonKim/reproduce_infilling')
parser.add_argument('--gpu', type=str, default='0', help='gpu')

parser.add_argument('--numEpoch', type=int, default=120, help='# of epoch')
parser.add_argument('--batchSize', type=int, default=80, help='input batch size for training')
parser.add_argument('--lr_landmark', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_adaptation', type=float, default=0.001, help='learning rate')
parser.add_argument('--print_interval', type=int, default=100, help='print interval')
args = parser.parse_args()




def main(args):
    #gpu
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    #util
    saveUtils = utils.saveData(args)
    writer = SummaryWriter(saveUtils.save_dir_tensorBoard)

    #model
    model4Landmark = ResNet34.ResNet34()
    model4adaptation = AdaptationNet.AdaptationNet()
    
    # optimizer
    optimizer4landmark = torch.optim.Adam(model4Landmark.parameters(), lr=args.lr_landmark)
    optimizer4adaptation = torch.optim.Adam(model4adaptation.parameters(), lr=args.lr_adaptation)
    
    # data loader
    train_dataloader, train_dataset = data_load.get_dataloader(args.datasetPath , args.batchSize)
    valid_dataloader, valid_dataset = data_load.get_dataloader(args.ValdatasetPath , args.batchSize)

    MSEloss = nn.MSELoss()

    print_train_loss = 0
    print_val_loss = 0 
    print_interval = 100
    for num_epoch in range(args.numEpoch):

        for iter_num, item in enumerate(train_dataloader):
            
            landmark_GT, input_Image = item
            landmark_GT = landmark_GT.to(device, dtype=torch.float)
            input_Image = input_Image.to(device, dtype=torch.float)
        
            pred = model4Landmark(input_Image)

            train_loss = MSEloss(landmark_GT, pred)
            print_train_loss += train_loss.item()

            optimizer4landmark.zero_grad()
            total_train_loss.backward()
            optimizer4landmark.step()

            #print and logging
            if iter_num % print_interval == 0 and iter_num != 0:
                print_train_loss = print_train_loss/print_interval
                log = "Train: [Epoch %d][Iter %d] [Train Loss: %.4f]" % (num_epoch, iter_num, print_train_loss)
                print(log)
                saveUtils.save_log(log)
                writer.add_scalar("Train Loss/ iter", train_iter_loss, iter_num)
                print_train_loss = 0
            
            #validation
            model.eval()
            for iter, item in enumerate(valid_dataloader):
                
                landmark_GT, input_Image = item
                landmark_GT = landmark_GT.to(device, dtype=torch.float)
                input_Image = input_Image.to(device, dtype=torch.float)

                with torch.no_grad():
                    pred = model4Landmark(masked_input)

                print_val_loss += MSEloss(landmark_GT, pred).item()
            model.train()
            #print, logging, save model per epoch 
            print_val_loss = print_val_loss/len(valid_dataloader)
            log = "Valid: [Epoch %d] [Valid Loss: %.4f]" % (num_epoch, print_val_loss)
            print(log)
            saveUtils.save_log(log)
            writer.add_scalar("Valid Loss/ Epoch", print_val_loss, num_epoch)
            saveUtils.save_model(model, num_epoch)

if __name__ == "__main__":
    main(args)