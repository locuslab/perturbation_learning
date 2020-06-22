# Adapted from the encoder architecture at 
# https://github.com/pytorch/examples/blob/master/vae/main.py
import torch
from torch import nn
from torch.nn import functional as F

class BaseEncoder(nn.Module): 
    def __init__(self, input_sz):
        super(BaseEncoder, self).__init__()

        self.fc1 = nn.Linear(input_sz, 784)
        self.fc21 = nn.Linear(784, 784)
        self.fc22 = nn.Linear(784, 784)

    def forward(self, x): 
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc21(x), self.fc22(x)

class Prior(BaseEncoder): 
    def __init__(self):
        super(Prior, self).__init__(784)

class Recognition(BaseEncoder):
    def __init__(self):
        super(Recognition, self).__init__(784*2)

    def forward(self, x, hx): 
        x_hx = torch.cat([x,hx],dim=1).view(-1,784*2)
        return super(Recognition, self).forward(x_hx)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc3 = nn.Linear(784*2, 784)
        self.fc4 = nn.Linear(784, 784)

    def forward(self, x, z): 
        x_z = torch.cat([x.view(-1,784),z],dim=1)
        x_z = F.relu(self.fc3(x_z))
        return torch.sigmoid(self.fc4(x_z)).view(-1,1,28,28)