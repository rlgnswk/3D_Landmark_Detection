import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Paper:
Method We train a ResNet34 [21] with mean squared error loss to directly predict 68 2D landmark coordinates per-image. 
We use the provided bounding boxes to extract a 256×256 pixel region-of-interest from each image.  ### gihoon : I think it is because of the input size of ResNet34
The private set has no bounding boxes, so we use a tight crop around landmarks.
'''

#https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

class ResidualBlock(nn.Module):
    def __init__(self, input_channel, output_channel, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channel)

        self.relu = nn.ReLU(inplace=True)
        
        if self.downsample is not None or input_channel != output_channel: 
            self.downsample = nn.Conv2d(input_channel, output_channel, kernel_size=1, stride = stride, bias=False)
            self.downsample_norm = nn.BatchNorm2d(output_channel) 
        
        else:
            self.downsample = downsample
        

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            identity = self.downsample_norm(identity)
            
        out += identity
        out = self.relu(out)

        return out

class ResidualBlockModule(nn.Module):
    def __init__(self, input_channel, output_channel, block_nums, stride=1, downsample=None):
        super(ResidualBlockModule, self).__init__()
        self.block_nums = block_nums
        self.blocks = []
        if 
        self.blocks.append(ResidualBlock(input_channel, output_channel , stride=2)) # first layer for downsampling and changing the channel depth
        for _ in range(1, block_nums):
            self.blocks.append(ResidualBlock(output_channel, output_channel))
            
    def forward(self, x):

        for i in range(self.block_nums):
            print(i)
            x = self.blocks[i](x)
        
        return x

class ResNet34(nn.Module):
      def __init__(self, input_channel = 3, output_class = 68):
        super(ResNet34, self).__init__()
        
        self.conv_first = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn_first = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool_first = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.RBM1 = ResidualBlockModule(64, 64 , 3)
        self.RBM2 = ResidualBlockModule(64, 128 , 4)
        self.RBM3 = ResidualBlockModule(128, 256 , 6)
        self.RBM4 = ResidualBlockModule(256, 512, 3)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_final = nn.Linear(512, output_class)

        #initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

      def forward(self, x):

        x = self.conv_first(x)
        x = self.bn_first(x)
        x = self.relu (x)
        x = self.pool_first(x) 

        x = self.RBM1(x)
        x = self.RBM2(x)
        x = self.RBM3(x)
        x = self.RBM4(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1) # remove 1 X 1 grid and make vector of tensor shape 
        x = self.fc_final(x)
        
        return x

if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]= "0"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    print("########### Size Check ###########")
    
    model = ResNet34()
    
    input_x = torch.randn(1, 3, 256, 256)
    print("input shape : ", input_x.shape)

    output = model(input_x)
    print("output shape : ", output.shape)
    

    print("########### Done ###########")