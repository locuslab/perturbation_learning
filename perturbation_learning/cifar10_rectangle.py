import torch
from torch import nn
from torch.nn import functional as F
from .scaled_tanh import ScaledTanh

class Block(nn.Module): 
    def __init__(self, in_planes, mid_planes, out_planes, k, p): 
        super(Block, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, mid_planes, 
                               kernel_size=k, padding=p, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv2 = nn.Conv2d(mid_planes, out_planes, 
                               kernel_size=k, padding=p, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, 
                                kernel_size=1, padding=0, bias=False) or None

    def forward(self, x): 
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class BaseEncoder(nn.Module): 
    def __init__(self, input_ch, config):
        super(BaseEncoder, self).__init__()

        latent_dim = config.model.latent_dim
        nblocks = config.model.nblocks
        big_ch = config.model.big_ch
        small_ch = config.model.small_ch
        k = config.model.kernel_size
        p = config.model.padding

        layers = [nn.Conv2d(input_ch, big_ch, 
                    kernel_size=k, padding=p, bias=False)]
        for i in range(nblocks): 
            layers.append(Block(big_ch, small_ch, big_ch, k=k, p=p))

        self.encoder = nn.Sequential(*layers, 
                nn.BatchNorm2d(big_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(big_ch, small_ch, 
                    kernel_size=k, padding=p, bias=False),
                nn.BatchNorm2d(small_ch),
                nn.ReLU(inplace=True)
            )
        self.fc1 = nn.Linear(small_ch*32*32, latent_dim)
        self.fc2 = nn.Linear(small_ch*32*32, latent_dim)
        self.tanh = ScaledTanh()

    def forward(self, x): 
        x = self.encoder(x).view(x.size(0), -1)
        return self.fc1(x), self.tanh(self.fc2(x))


class Prior(BaseEncoder): 
    def __init__(self, config):
        super(Prior, self).__init__(3, config)


class Recognition(BaseEncoder):
    def __init__(self, config):
        super(Recognition, self).__init__(6, config)

    def forward(self, x, hx): 
        x_hx = torch.cat([x,hx],dim=1)
        return super(Recognition, self).forward(x_hx)

class Generator(nn.Module): 
    def __init__(self, config):
        super(Generator, self).__init__()

        latent_dim = config.model.latent_dim
        nblocks = config.model.nblocks
        big_ch = config.model.big_ch
        small_ch = config.model.small_ch
        k = config.model.kernel_size
        p = config.model.padding

        self.fc = nn.Linear(latent_dim,32*32)

        layers = [nn.Conv2d(4, big_ch, 
                    kernel_size=k, padding=p, bias=False)]
        for i in range(nblocks): 
            layers.append(Block(big_ch, small_ch, big_ch, k=k, p=p))

        self.decoder = nn.Sequential(*layers, 
                nn.BatchNorm2d(big_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(big_ch, 3, 
                    kernel_size=k, padding=p, bias=False)
            )


    def forward(self, x, z): 
        z = self.fc(z).view(z.size(0),1,32,32)
        x_z = torch.cat([x,z],dim=1)
        h = self.decoder(x_z)
        return torch.sigmoid(h)