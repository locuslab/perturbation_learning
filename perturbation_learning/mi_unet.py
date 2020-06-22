import torch
from torch import nn
from torch.nn import functional as F

from .unet_parts import *
from .scaled_tanh import ScaledTanh

def parse_unet_params(config): 
    in_h,in_w = config.model.dim

    bilinear = config.model.bilinear
    factor = 2 if bilinear else 1

    chs = config.model.chs
    up_chs = [ch for ch in chs]
    down_chs = [ch for ch in chs]
    down_chs[4] = chs[4] // factor
    chs = down_chs

    hs = [in_h//(2**i) for i in range(len(chs))]
    ws = [in_w//(2**i) for i in range(len(chs))]

    shapes = list(zip (chs,hs,ws))
    hws = [h*w for h,w in zip(hs,ws)]
    dims = [ch*hw for ch,hw in zip(chs,hws)]

    ldims = config.model.latent_dim

    return {
        "bilinear": bilinear, 
        "factor": factor, 
        "chs": chs, 
        "up_chs": up_chs,
        "hws": hws, 
        "shapes": shapes, 
        "dims": dims, 
        "ldims": ldims, 
        "pool": config.model.pool
    }

class BaseEncoder(nn.Module): 
    def __init__(self, input_ch, config):
        super(BaseEncoder, self).__init__()
        n_channels = input_ch
        params = parse_unet_params(config)
        chs = params["chs"]
        hws = params["hws"]
        ldims = params["ldims"]
        pool = params["pool"]

        reduce_ch = 16

        self.inc = DoubleConv(n_channels, chs[0])
        self.down1 = Down(chs[0], chs[1])
        self.down2 = Down(chs[1], chs[2])
        self.down3 = Down(chs[2], chs[3])
        self.down4 = Down(chs[3], chs[4])

        self.pool = nn.ModuleList([nn.AdaptiveAvgPool2d(p) for p in pool])
        self.conv = nn.ModuleList([nn.Conv2d(ch, reduce_ch, kernel_size=1) for ch in chs])

        self.fc1 = nn.ModuleList([
                nn.Linear(reduce_ch*h*w,ld) for (h,w),ld in zip(pool,ldims)
        ])
        self.fc2 = nn.ModuleList([
                nn.Linear(reduce_ch*h*w,ld) for (h,w),ld in zip(pool,ldims)
        ])
        self.tanh = ScaledTanh()

    def forward(self, x): 
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        xs = [x1, x2, x3, x4, x5]
        xs = [F.relu(conv(x)) for conv,x in zip(self.conv,xs)]
        xs = [pool(x) for pool,x in zip(self.pool,xs)]
        n = x.size(0)
        zs = [(fc1(z.view(n,-1)),fc2(z.view(n,-1))) for z,fc1,fc2 in zip(xs,self.fc1,self.fc2)]
        mus, logvars = zip(*zs)

        return torch.cat(mus,dim=1), self.tanh(torch.cat(logvars,dim=1))


class Prior(BaseEncoder): 
    def __init__(self, config):
        super(Prior, self).__init__(3, config)


class Recognition(BaseEncoder):
    def __init__(self, config):
        super(Recognition, self).__init__(6, config)

    def forward(self, x, hx): 
        x_hx = torch.cat([x,hx],dim=1)
        return super(Recognition, self).forward(x_hx)

def split(z,ldims): 
    start = 0
    zs = []
    for i in ldims: 
        zs.append(z[:,start:start+i])
        start += i
    return zs

class Generator(nn.Module): 
    def __init__(self, config):
        super(Generator, self).__init__()
        n_channels = 3
        params = parse_unet_params(config)
        chs = params["chs"]
        up_chs = params["up_chs"]
        hws = params["hws"]
        ldims = params["ldims"]
        factor = params["factor"]
        bilinear = params["bilinear"]
        pool = params["pool"]

        self.inc = DoubleConv(n_channels, chs[0])
        self.down1 = Down(chs[0], chs[1])
        self.down2 = Down(chs[1], chs[2])
        self.down3 = Down(chs[2], chs[3])
        self.down4 = Down(chs[3], chs[4])

        self.up1 = Up(up_chs[4]+2, up_chs[3] // factor, bilinear)
        self.up2 = Up(up_chs[3]+1, up_chs[2] // factor, bilinear)
        self.up3 = Up(up_chs[2]+1, up_chs[1] // factor, bilinear)
        self.up4 = Up(up_chs[1]+1, up_chs[0], bilinear)
        self.outc = OutConv(up_chs[0], 3)

        self.fc = nn.ModuleList([
                nn.Linear(ld,h*w) for ld,(h,w) in zip(ldims,pool)
        ])
        self.shapes = params["shapes"]
        self.ldims = ldims
        self.pool = pool

    def forward(self, x, z): 
        batch_size = x.size(0)
        zs = split(z, self.ldims)
        zs = [fc(z) for z,fc in zip(zs,self.fc)]
        zs = [z.view(batch_size,1,h,w) for z,fc,(h,w) in zip(zs,self.fc,self.pool)]


        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        xs = [x1,x2,x3,x4,x5]
        zs = [F.interpolate(z,x.size()[2:], mode='bilinear') for x,z in zip(xs,zs)]
        z1,z2,z3,z4,z5 = zs        

        xz1 = torch.cat([x1,z1],dim=1)
        xz2 = torch.cat([x2,z2],dim=1)
        xz3 = torch.cat([x3,z3],dim=1)
        xz4 = torch.cat([x4,z4],dim=1)
        xz5 = torch.cat([x5,z5],dim=1)

        x = self.up1(xz5, xz4)
        x = self.up2(x, xz3)
        x = self.up3(x, xz2)
        x = self.up4(x, xz1)
        logits = self.outc(x)
        return torch.sigmoid(logits)