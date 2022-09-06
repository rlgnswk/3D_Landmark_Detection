
import os
import os.path
import torch
import sys
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as 

import visualization

class saveData():
    def __init__(self, args):
        self.args = args
        #Generate Savedir folder
        self.save_dir = os.path.join(args.saveDir, args.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        #Generate Savedir/model
        self.save_dir_model = os.path.join(self.save_dir, 'model')
        if not os.path.exists(self.save_dir_model):
            os.makedirs(self.save_dir_model)

        #Generate Savedir/validation
        self.save_dir_validation = os.path.join(self.save_dir, 'validation')
        if not os.path.exists(self.save_dir_validation):
            os.makedirs(self.save_dir_validation)

        #Generate Savedir/tensorBoard
        self.save_dir_tensorBoard = os.path.join(self.save_dir, 'tensorBoard')
        if not os.path.exists(self.save_dir_tensorBoard):
            os.makedirs(self.save_dir_tensorBoard)

        #Generate Savedir/log.txt
        if os.path.exists(self.save_dir + '/log.txt'):
            self.logFile = open(self.save_dir + '/log.txt', 'a')
        else:
            self.logFile = open(self.save_dir + '/log.txt', 'w')

    def save_log(self, log):
        sys.stdout.flush()
        self.logFile.write(log + '\n')
        self.logFile.flush()
        
    def save_model(self, model, epoch):
        torch.save(
            model.state_dict(),
            self.save_dir_model + '/model_' + str(epoch) + '.pt')

    def save_visualization(self, image, landmark, landmark_GT, saveUtils, num_for_visual = 5)

        for i in range(num_for_visual):
            pred_visual = make_image(image, landmark)
            GT_visual = make_image(image, landmark_GT)

        #save(pred_visual,  )
        #save(GT_visual,  )