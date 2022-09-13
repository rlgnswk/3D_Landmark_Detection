import torch
import torch.nn as nn
import torch.nn.functional as F
'''
Paper:
Label adaptation is performed using a' two-layer perceptron' to address systematic differences between synthetic
and real landmark labels (Figure 15). This network is never exposed to any real images during training.
'''

class AdaptationNet(nn.Module):
      def __init__(self, input_num, hidden_num, output_num):
        super(AdaptationNet, self).__init__()
        
        self.fc1 = nn.Linear(input_num, hidden_num)
        self.fc2 = nn.Linear(hidden_num, output_num)

        self.ReLU = nn.ReLU(True)
        
      def forward(self, x):
        
        x = self.ReLU(self.fc1(x))
        out = self.fc2(x)

        return out