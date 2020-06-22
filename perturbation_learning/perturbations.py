import torch
from torchvision import transforms

def linfinity(x, config): 
    d = torch.zeros_like(x).uniform_(-config.epsilon, config.epsilon)
    return torch.clamp(x+d, min=config.min, max=config.max)
def _linfinity(config): 
    return lambda x: linfinity(x[0],config)

def rotation(config): 
    t = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.RandomRotation(config.degree, fill=(0,)), 
        transforms.ToTensor()
        ])
    return torch.cat([t(x[i]) for i in range(x.size(0))], dim=0).unsqueeze(1)
def _rotation(config): 
    return lambda x: rotation(x[0],config)

def rts(x, config): 
    t = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.RandomAffine(config.angle, scale=config.scale, fillcolor=0), 
        transforms.RandomCrop(config.crop_sz,padding=config.padding),
        transforms.ToTensor()
        ])
    return torch.cat([t(x[i]) for i in range(x.size(0))], dim=0).unsqueeze(1)
def _rts(config): 
    return lambda x: rts(x[0],config)

hs = {
    "linfinity": _linfinity, 
    "rotation": _rotation, 
    "rts": _rts, 
    "dataloader": lambda config: (lambda x: x[2]),
    "none": lambda config: (lambda x: x[0])
}