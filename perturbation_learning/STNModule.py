""" A plug and play Spatial Transformer Module in Pytorch 
Reference: https://github.com/aicaffeinelife/Pytorch-STN/blob/master/models/STNModule.py
""" 
import os 
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class Flatten(nn.Module): 
    def forward(self, x): 
        return x.view(x.size(0),-1)

class SpatialTransformer(nn.Module):
    """
    Implements a spatial transformer 
    as proposed in the Jaderberg paper. 
    Comprises of 3 parts:
    1. Localization Net
    2. A grid generator 
    3. A roi pooled module.

    The current implementation uses a very small convolutional net with 
    2 convolutional layers and 2 fully connected layers. Backends 
    can be swapped in favor of VGG, ResNets etc. TTMV
    Returns:
    A roi feature map with the same input spatial dimension as the input feature map. 
    """
    def __init__(self, in_channels, spatial_dims):
        super(SpatialTransformer, self).__init__()
        self._h, self._w = spatial_dims 
        self._in_ch = in_channels 

        d = self._h * self._w * self._in_ch
        self.localization = nn.Sequential(
            Flatten(), 
            nn.Linear(d,512), 
            nn.ReLU(True),
            nn.Linear(512, 128), 
            nn.ReLU(True)
        )
 
        self.fc = nn.Linear(128, 6)
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))


    def forward(self, batch_images, x): 
        """
        Forward pass of the STN module. 
        batch_images -> input to be transformed
        x -> localization input 
        """
        bs = x.size(0)
        x = self.localization(x)
        # print("Pre view size:{}".format(x.size()))
        x = self.fc(x) # params [Nx6]
        x = x.view(-1, 2,3) # change it to the 2x3 matrix 
        #print(x.size())
        affine_grid_points = F.affine_grid(x, torch.Size((x.size(0), self._in_ch, self._h, self._w)))
        assert(affine_grid_points.size(0) == batch_images.size(0)), "The batch sizes of the input images must be same as the generated grid."
        rois = F.grid_sample(batch_images, affine_grid_points,
            padding_mode="border")
        #print("rois found to be of size:{}".format(rois.size()))
        return rois
        

        

        
