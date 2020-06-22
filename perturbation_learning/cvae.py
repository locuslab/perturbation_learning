import torch
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image

from . import mnist_fc, mnist_conv
from . import mnist_stn
from . import cifar10_rectangle
from . import mi_unet

class BaseAutoEncoder(nn.Module):
    def forward(self, x, hx):
        raise ValueError("forward not implemented")
    def sample(self, x):
        raise ValueError("sample not implemented")
    def dataparallel(self):
        pass
    def undataparallel(self):
        pass

class CVAE(BaseAutoEncoder):
    def __init__(self, prior, recognition, generator):
        super(CVAE, self).__init__()
        self.prior = prior
        self.recognition = recognition
        self.generator = generator

    def encode(self, x, hx):
        prior_params = self.prior(x)
        assert len(prior_params) == 2, "Prior network must output two outputs (mu, log_var)"

        recog_params = self.recognition(x, hx)
        assert len(prior_params) == 2, "Recognition network must output two outputs (mu, log_var)"
        return prior_params, recog_params

    def reparameterize(self, recog_params, eps=None):
        """ If eps is given, reparameterize it. Otherwise, draw from N(0,1) """
        mu, logvar = recog_params
        std = torch.exp(0.5*logvar)
        if eps is None:
            eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, x, z):
        return self.generator(x,z)

    def forward(self, x, hx):
        prior_params, recog_params = self.encode(x,hx)
        z = self.reparameterize(recog_params)
        return self.decode(x,z), prior_params, recog_params

    def sample(self, x, eps=None):
        prior_params = self.prior(x)
        assert len(prior_params) == 2, "Prior network must output two outputs (mu, log_var)"
        z = self.reparameterize(prior_params, eps=eps)
        return self.decode(x,z)

    def dataparallel(self):
        self.prior = nn.DataParallel(self.prior)
        self.recognition = nn.DataParallel(self.recognition)
        self.generator = nn.DataParallel(self.generator)

    def undataparallel(self):
        self.prior = self.prior.module
        self.recognition = self.recognition.module
        self.generator = self.generator.module

# Feedforward VAEs

def _CVAE(module, *args, **kwargs):
    return CVAE(module.Prior(*args, **kwargs),
                module.Recognition(*args, **kwargs),
                module.Generator(*args, **kwargs))

def MNIST_FCCVAE(config):
    return _CVAE(mnist_fc)

def MNIST_ConvCVAE(config):
    return _CVAE(mnist_conv)

def MNIST_STNCVAE(config):
    return _CVAE(mnist_stn)

def CIFAR10_Rectangle(config):
    return _CVAE(cifar10_rectangle, config)

def MI_UNet(config): 
    return _CVAE(mi_unet, config)

models = {
    "mnist_fc" : MNIST_FCCVAE,
    "mnist_conv" : MNIST_ConvCVAE,
    "mnist_stn" : MNIST_STNCVAE,
    "cifar10_rectangle" : CIFAR10_Rectangle, 
    "mi_unet": MI_UNet
}


def KL(mu0, logvar0, mu1, logvar1):
    bs = mu0.size(0)
    var0 = torch.exp(logvar0)
    var1 = torch.exp(logvar1)
    return 0.5*((var0/var1) + ((mu1-mu0)**2)/var1 - 1 + logvar1 - logvar0).view(bs,-1).sum(-1)

def reconstruction_loss(recon_hx, hx, distribution):
    bs = recon_hx.size(0)
    if distribution == 'bernoulli':
        neg_ll = F.binary_cross_entropy(recon_hx.view(bs,-1), hx.view(bs, -1), reduction='none').sum(-1)
    elif distribution == 'gaussian':
        neg_ll = F.mse_loss(recon_hx.view(bs,-1), hx.view(bs, -1), reduction='none').sum(-1)
    else:
        raise ValueError("Unknown output distribution")
    return neg_ll

def vae_loss(hx, recon_hx, prior_params, recog_params, beta=1, reduction='sum', distribution='bernoulli'):
    bs = recon_hx.size(0)
    neg_ll = reconstruction_loss(recon_hx, hx, distribution)

    mu_prior, logvar_prior = prior_params
    mu, logvar = recog_params
    kl_args = (a.view(bs,-1) for a  in [mu, logvar, mu_prior, logvar_prior])
    KLD = KL(*kl_args)

    # transformation log likelihood
    if reduction =='sum':
        return neg_ll.sum(), (KLD*beta).sum()
    elif reduction=='mean':
        return neg_ll.mean(), (KLD*beta).mean()
    elif reduction=='none':
        return neg_ll, (KLD*beta)
    else:
        raise ValueError