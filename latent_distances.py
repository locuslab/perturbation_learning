import matplotlib
import matplotlib.pyplot as plt
import os

import utilities
from perturbation_learning import cvae, perturbations, datasets
import torch 
import torch.nn.functional as F
import torchvision.utils
from torchvision.utils import save_image

import argparse
parser = argparse.ArgumentParser(
                    description='Latent space calculator',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('name', type=str)
args = parser.parse_args()
name = args.name
print(f"loading model and dataloader for {name}")
config_dict = utilities.get_config(f'configs/{name}.json')
config = utilities.config_to_namedtuple(config_dict)

train_loader, test_loader, val_loader = datasets.loaders[config.dataset.type](config)

model = cvae.models[config.model.type](config)
model.cuda()
h = perturbations.hs[config.perturbation.test_type](config.perturbation)
model.load_state_dict(torch.load(f'experiments/{name}/checkpoints/checkpoint_latest.pth')['model_state_dict'])
model.eval();

if False: 
    print("plotting pictures")
    with torch.no_grad(): 
        for batch_idx, batch in enumerate(test_loader):
            data = batch[0]

            hdata = h(batch)
            data = data.to(config.device)
            hdata = hdata.to(config.device)
            output = model(data, hdata)
            hsample = model.sample(data)

            n = min(data.size(0), 8)
            recon_hbatch = output[0]
            if 'cifar10' in name: 
                pass
            else:
                data, hdata, recon_hbatch = [F.interpolate(t, size=(125,187), mode="bilinear") for t in [data,hdata,recon_hbatch]]
            hcomparison = torch.cat([
                                    data[:n],
                                    hdata[:n],
                                    recon_hbatch.view(*hdata.size())[:n]])
            save_image(hcomparison.cpu(),
                     os.path.join('figures', f'hreconstruction.png'), nrow=n)

            save_image(hsample[:min(64,config.eval.batch_size)],
                       os.path.join('figures', f'hsample.png'))

            repeat_hsample = torch.cat([model.sample(data)[:8].unsqueeze(1) for i in range(8)],dim=1)
            repeat_hsample = repeat_hsample.view(-1,*hdata.size()[1:])
            save_image(repeat_hsample[:min(64,config.eval.batch_size)],
                       os.path.join('figures', f'repeat_hsample.png'))
            break

if False: 
    print("making interpolation")
    for batch in test_loader: 
        data = batch[0]
        break
    hdata = h(batch)
    data = data.to(config.device)
    hdata = hdata.to(config.device)
    output = model(data, hdata)
    recon_hbatch = output[0]

    n = 4
    nperturb = 12
    l = [(data[i*nperturb:(i+1)*nperturb],
                            hdata[i*nperturb:(i+1)*nperturb],
                            recon_hbatch.view(*hdata.size())[i*nperturb:(i+1)*nperturb]) for i in range(n)]
    hcomparison = torch.cat([inner for outer in l for inner in outer])

    z = model.recognition(data, hdata)[0][[0,4,6,10]]

    nrow = 10
    latent_dim = config.model.latent_dim
    if isinstance(latent_dim, list): 
        latent_dim = sum(latent_dim)
    def interpolate(z, n=6): 
        z1, z2, z3, z4 = z
        z12 = torch.cat([(t*z1 + (1-t)*z2).unsqueeze(0) for t in torch.linspace(0,1,n)])
        z34 = torch.cat([(t*z3 + (1-t)*z4).unsqueeze(0) for t in torch.linspace(0,1,n)])
        z1234 = torch.cat([(t*z12 + (1-t)*z34).unsqueeze(0) for t in torch.linspace(0,1,n)])
        return z1234
    zbatch = interpolate(z, n=nrow).view(-1,latent_dim)
    Xbatch = data[:1].repeat(nrow*nrow,1,1,1)

    img_interpolation = model.generator(Xbatch, zbatch)
    fig, ax = plt.subplots(figsize=(12,12))
    ax.imshow(torchvision.utils.make_grid(img_interpolation, nrow=nrow).cpu().detach().permute(1,2,0).numpy())
    fig.savefig(f'figures/latent_interpolation_{name}.png')

print("loading all dataset loader")
config_dict = utilities.get_config(f'configs/{name}.json')
if 'cifar10' in name: 
    config_dict['dataset']['type'] = 'cifar10c_all'
elif 'mi_unet' in name:
    config_dict['dataset']['mode'] = 'all'
    config_dict['eval']['batch_size']=32
config = utilities.config_to_namedtuple(config_dict)
train_loader, test_loader, val_loader = datasets.loaders[config.dataset.type](config)

loops = 5 if 'mnist' in name else 0


print("making distribution")
norms = []
with torch.no_grad(): 
    for _ in range(loops): 
        for batch in val_loader: 
            data = batch[0]
            hdata = h(batch)

            data = data.to(config.device)
            hdata = hdata.to(config.device)

            mu_prior, logvar_prior = model.prior(data)
            mu_recog, logvar_recog = model.recognition(data,hdata)

            std = torch.exp(0.5*logvar_prior)
            norms.append(((mu_recog - mu_prior)/std).view(data.size(0),-1).norm(p=2,dim=1))


import numpy as np
import scipy.stats as stats

norms_np = torch.cat(norms).cpu().numpy()
print(f'Max {norms_np.max()} Mean {norms_np.mean()} Std {norms_np.std()}')
p = np.percentile(norms_np, [25,50,75,99.9])
print(f'25% {p[0]} 50% {p[1]} 75% {p[2]} 99% {p[3]}')
x = np.linspace(0,25,100)
density = stats.gaussian_kde(norms_np)
fig, ax = plt.subplots(figsize=(4,4))
ax.plot(x, density(x))
fig.savefig(f'figures/latent_distribution_{name}.png')
np.save(f"data/latent_norms_{name}.npy", norms_np)