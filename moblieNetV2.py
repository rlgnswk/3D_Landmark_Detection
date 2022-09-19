import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

'''
https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py
class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
    )
'''

class moblieNetV2(nn.Module):
    def __init__(self, input_channel = 3, output_class = 70, output_param = 2):
        super(moblieNetV2, self).__init__()

        self.model = torchvision.models.mobilenet_v2(num_classes = output_class * output_param)
        
    def forward(self, x):

        out = self.model(x)

        return out



if __name__ == '__main__':

    model = torchvision.models.mobilenet_v2(num_classes = 70 * 3)
    print(model)
    torch.save(model.state_dict(), 'sample.pt')
    
    '''
    print(model) output ==> 

    MobileNetV2(
  (features): Sequential(
    (0): ConvBNActivation(
      (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
    )
    (1): InvertedResidual(
      (conv): Sequential ...

    '''
    '''import os
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
    
    print("########### Done ###########")'''