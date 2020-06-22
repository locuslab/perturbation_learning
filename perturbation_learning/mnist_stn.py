import torch
from torch import nn
from torch.nn import functional as F
from .STNModule import SpatialTransformer

class Block(nn.Module): 
    def __init__(self, nch, k=5, p=2):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(nch,nch,k)
        self.conv2 = nn.Conv2d(nch,nch,k)
        self.pad = nn.ReflectionPad2d(p)
        self.bn1 = nn.BatchNorm2d(nch)
        self.bn2 = nn.BatchNorm2d(nch)
    def forward(self,x): 
        h1 = self.conv1(self.pad(F.relu(self.bn1(x))))
        h2 = self.conv2(self.pad(F.relu(self.bn2(h1))))
        return h2

class BaseEncoder(nn.Module): 
    def __init__(self, input_ch, input_width, input_height):
        super(BaseEncoder, self).__init__()
        h,w = input_height, input_width

        self.block = Block(input_ch)
        self.fc31 = nn.Linear(h*w*input_ch, 128)
        self.fc32 = nn.Linear(h*w*input_ch, 128)


    def forward(self, x): 
        x = self.block(x)
        x = x.view(x.size(0), -1)
        return self.fc31(x), self.fc32(x)

class Prior(BaseEncoder): 
    def __init__(self):
        super(Prior, self).__init__(1,42,42)


class Recognition(BaseEncoder):
    def __init__(self):
        super(Recognition, self).__init__(2,42,42)

    def forward(self, x, hx): 
        x_hx = torch.cat([x,hx],dim=1)
        return super(Recognition, self).forward(x_hx)

class Generator(nn.Module):
    def __init__(self, input_height=28, input_width=28):
        super(Generator, self).__init__()
        self.fc = nn.Linear(128,42*42)
        self.stns = nn.ModuleList([SpatialTransformer(1,(42,42)) for _ in range(5)])
        self.block = Block(5)
        self.pad = nn.ReflectionPad2d(2)
        self.conv = nn.Conv2d(5,1,5)

    def forward(self, x, z): 
        z = self.fc(z).view(z.size(0),1,42,42)
        x_z = torch.cat([stn(x,z) for stn in self.stns],dim=1)
        h = self.conv(self.pad(self.block(x_z)))
        return torch.sigmoid(h)