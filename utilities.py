from collections import namedtuple
import json

import numpy as np
from scipy.stats import truncnorm
import torch

def torch_truncnorm(r, *size): 
    numel = np.prod(size)
    x = truncnorm.rvs(-r, r, size=numel)
    return torch.from_numpy(x).view(*size).float()

def get_config(config_path):
    with open(config_path) as config_file:
        config = json.load(config_file)

    return config


def config_to_namedtuple(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = config_to_namedtuple(value) 
        return namedtuple('GenericDict', obj.keys())(**obj)
    elif isinstance(obj, list):
        return [config_to_namedtuple(item) for item in obj]
    else:
        return obj

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':.4f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class MultiAverageMeter(object): 
    def __init__(self, names): 
        self.AMs = { name: AverageMeter(name) for name in names }

    def __getitem__(self, k): 
        return self.AMs[k].avg

    def reset(self): 
        for am in self.AMs.values(): 
            am.reset()

    def update(self, vals, n=1): 
        for k,v in vals.items(): 
            self.AMs[k].update(v, n=n)

    def __str__(self): 
        return ' '.join([str(am) for am in self.AMs.values()])


