import torch
import torch.nn as nn
import torch.nn.functional as F


class moblieNetV2(nn.Module):
    def __init__(self, input_channel = 3, output_class = 70, output_param = 2):
        super(moblieNetV2, self).__init__()

        self.conv_first = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        

    def forward(self, x):

        return out



if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]= "0"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    print("########### Size Check ###########")
    
    model = moblieNetV2()
    print(model)
    model.cuda()
    input_x = torch.randn(1, 3, 256, 256).cuda()
    print("input shape : ", input_x.shape)

    output = model(input_x)
    print("output shape : ", output.shape)
    
    print("########### Done ###########")