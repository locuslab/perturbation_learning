import torch
from torch import nn
from torch.nn import functional as F

class BaseEncoder(nn.Module): 
    def __init__(self, input_ch, input_width, input_height):
        super(BaseEncoder, self).__init__()
        h,w = input_height//4, input_width//4

        self.conv1 = nn.Conv2d(input_ch, 32, 3, padding=1)
        self.pooling = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv31 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv32 = nn.Conv2d(64, 128, 3, padding=1)

    def forward(self, x): 
        x = self.pooling(F.relu(self.conv1(x)))
        x = self.pooling(F.relu(self.conv2(x)))
        return self.conv31(x), self.conv32(x)

class Prior(BaseEncoder): 
    def __init__(self):
        super(Prior, self).__init__(1,28,28)

class Recognition(BaseEncoder):
    def __init__(self):
        super(Recognition, self).__init__(2,28,28)

    def forward(self, x, hx): 
        x_hx = torch.cat([x,hx],dim=1)
        return super(Recognition, self).forward(x_hx)

class Generator(nn.Module):
    def __init__(self, input_height=28, input_width=28):
        super(Generator, self).__init__()
        self.h, self.w = h,w = input_height//4, input_width//4

        self.downsample3 = nn.Conv2d(1, 1, 1, stride=4)
        self.downsample4 = nn.Conv2d(1, 1, 1, stride=2)

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv3 = nn.Conv2d(128+1, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64+1, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(32+1, 1, 3, padding=1)

    def forward(self, x, z): 
        x_z = torch.cat([self.downsample3(x),z],dim=1)
        x_z = F.relu(self.conv3(x_z))

        x_z = torch.cat([self.downsample4(x),self.upsample(x_z)],dim=1)
        x_z = F.relu(self.conv4(x_z))

        x_z = torch.cat([x,self.upsample(x_z)],dim=1)

        return torch.sigmoid(self.conv5(x_z))