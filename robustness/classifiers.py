from . import preactresnet as res
from . import wideresnet
from . import unet_model

import torch
import torch.nn as nn

from torchvision.models import resnet 

import warnings

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CIFAR10Normalize(nn.Module): 
    def __init__(self):
        super(CIFAR10Normalize, self).__init__()
        self.register_buffer('mu', torch.Tensor((0.4914, 0.4822, 0.4465)).view(3,1,1))
        self.register_buffer('std', torch.Tensor((0.2023, 0.1994, 0.2010)).view(3,1,1)) 
    def forward(self, x): 
        return (x-self.mu)/self.std

def MNISTConv(config):
    model = nn.Sequential(
        nn.Conv2d(1, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64*7*7,1024),
        nn.ReLU(),
        nn.Linear(1024, 10)
    )
    return model

def WideResNet(config): 
    layers = config.model.layers
    width_factor = config.model.width_factor
    n_classes = config.model.n_classes
    model = wideresnet.WideResNet(layers,n_classes,widen_factor=width_factor)

    if config.model.normalize: 
        model = nn.Sequential(
            CIFAR10Normalize(), 
            model
        )
    else: 
        warnings.warn("not normalizing the WideResNet with CIFAR10 channel stats")
    return model

def UNet(config): 
    n_channels = config.model.n_channels
    n_classes = config.model.n_classes
    bilinear = config.model.bilinear
    
    return unet_model.UNet(n_channels, n_classes, bilinear)    

def ResNet50(config): 
    return resnet.resnet50(num_classes=config.model.n_classes)

models = {
    "mnist_conv": MNISTConv, 
    "wideresnet": WideResNet, 
    "unet": UNet
}
