import torch
import torch.nn as nn
import torch.nn.functional as F


class moblieNet(nn.Module):
    def __init__(self, input_channel = 3, output_class = 70, output_param = 2):
        super(moblieNet, self).__init__()

        self.conv_first = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        


    def forward(self, x):

        return out
