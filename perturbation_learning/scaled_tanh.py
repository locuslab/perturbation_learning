import torch.nn as nn
import math

TANHUPPER = math.log(10)
TANHLOWER = math.log(1e-3)

class ScaledTanh(nn.Tanh): 
    def __init__(self, *args, lower=TANHLOWER, upper=TANHUPPER, **kwargs): 
        super(ScaledTanh, self).__init__(*args, **kwargs)
        self.shift = (lower + upper)/2
        self.scale = (lower - upper)/2 
    def forward(self, x): 
        x = super(ScaledTanh, self).forward(x)
        return x*self.scale + self.shift