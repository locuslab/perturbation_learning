import argparse
import json
import os
import numpy as np 
import logging

import utilities
from perturbation_learning import cvae, perturbations, datasets

import torch
import torch.nn.functional as F
from torch import optim
from torchvision.utils import save_image 

def approximation(X, hX, cvae, max_dist=1, 
            alpha=0.2, niters=10, over=True, distribution='gaussian'): 
    bs = X.size(0)
    with torch.no_grad(): 
        if over: 
            # maximizing error: start randomly
            prior_params = cvae.prior(X)
            delta = torch.zeros_like(prior_params[0])
            delta.normal_()
            norm = delta.norm(p=2,dim=1)
            magnitude = max_dist*torch.rand(*norm.size()).to(norm.device)
            delta = (delta * (magnitude / norm).unsqueeze(1))
        else: 
            # minimizing error: start at encoding
            prior_params, recog_params = cvae.encode(X, hX)
            delta = (recog_params[0] - prior_params[0])/prior_params[1]
            delta = delta.renorm(2,1,max_dist)

    for i in range(niters):
        # print(f"iteration {i}")
        with torch.enable_grad():
            delta.requires_grad = True

            # generate perturbed image
            z = cvae.reparameterize(prior_params, eps=delta)
            X_cvae = cvae.decode(X, z)

            # compute loss and backward
            if distribution == 'gaussian': 
                loss = F.mse_loss(X_cvae, hX)
            elif distribution == 'bernoulli': 
                loss = F.binary_cross_entropy(X_cvae, hX)
            else: 
                raise ValueError
            if not over: 
                loss = -loss
            loss.backward()

        with torch.no_grad(): 
            # take L2 gradient step
            g = delta.grad
            g = g / g.norm(p=2,dim=1).unsqueeze(1)
            delta = delta + alpha * g
            # project onto ball of radius max_dist
            delta = delta.renorm(2,1,max_dist)
        delta = delta.clone().detach()
    
    with torch.no_grad(): 
        z = cvae.reparameterize(prior_params, eps=delta)
        X_cvae = cvae.decode(X, z).detach()

        if distribution == 'gaussian': 
            loss = F.mse_loss(X_cvae, hX)
        elif distribution == 'bernoulli': 
            loss = F.binary_cross_entropy(X_cvae, hX)
        else: 
            raise ValueError
        if not over: 
            loss = -loss
        return loss

def expected_approximation(X, hX, cvae, max_dist, distribution='gaussian',
                           n=5):
    with torch.no_grad(): 
        prior_params = cvae.prior(X)

        losses = []
        for i in range(n): 
            eps = torch.randn_like(prior_params[0])
            if (eps.view(eps.size(0),-1).norm(dim=1) > max_dist).any(): 
                # rather than doing rejection sampling, just use the scipy
                # truncnorm implementation, can replace with pytorch truncated 
                # normal when officially released
                # https://github.com/pytorch/pytorch/pull/32397
                eps = utilities.torch_truncnorm(max_dist, *eps.size()).to(eps.device)
                # print(eps.view(eps.size(0),-1).norm(dim=1), max_dist)
                # raise NotImplementedError("Sample has large norm but truncated normal not implemented")
            z = cvae.reparameterize(prior_params, eps=eps)
            X_cvae = cvae.decode(X,z)

            if distribution == 'gaussian': 
                loss = F.mse_loss(X_cvae, hX)
            elif distribution == 'bernoulli': 
                loss = F.binary_cross_entropy(X_cvae, hX)
            else: 
                raise ValueError
            losses.append(loss.item())
        return torch.Tensor(losses).mean()



def fast_approximation(X, hX, cvae, max_dist, distribution='gaussian'): 
    with torch.no_grad(): 
        prior_params, recog_params = cvae.encode(X, hX)
        eps = (recog_params[0] - prior_params[0])/prior_params[1]
        eps = eps.renorm(2,1,max_dist)
        z = cvae.reparameterize(prior_params, eps=eps)
        X_cvae = cvae.decode(X, z)
        if distribution == 'gaussian': 
            return F.mse_loss(X_cvae, hX)
        elif distribution == 'bernoulli': 
            return F.binary_cross_entropy(X_cvae, hX)
        else: 
            raise ValueError
        # return F.mse_loss(X_cvae,X)

def loop(config, model, logger, loader, h): 
    meters = utilities.MultiAverageMeter([
        "enc_ae", "ae", "eae", "oae", "kl", "recon", "loss"
    ])
    model.eval()
    for batch_idx, batch in enumerate(loader):
        data = batch[0]
        hdata = h(batch)
        data = data.to(config.device)
        hdata = hdata.to(config.device)

        output = model(data, hdata)
        recon_loss, kl_loss = cvae.vae_loss(hdata, *output, beta=1,
                             distribution=config.model.output_distribution)

        enc_ae = fast_approximation(data, hdata, model, config.ae.max_dist, 
                                    distribution=config.model.output_distribution)

        ae = approximation(data, hdata, model, 
            max_dist=config.ae.max_dist, 
            alpha=config.ae.alpha, niters=config.ae.niters, 
            over=False, 
            distribution=config.model.output_distribution)

        eae = expected_approximation(data, hdata, model, config.ae.max_dist, 
                                    distribution=config.model.output_distribution)

        oae = torch.zeros(1)
        # oae = approximation(data, hdata, model, 
        #     max_dist=config.oae.max_dist, 
        #     alpha=config.oae.alpha, 
        #     niters=config.oae.niters, over=True, 
        #     distribution=config.model.output_distribution)

        meters.update({
            "enc_ae" : enc_ae.item(), 
            "ae": ae.item(), 
            "eae": eae.item(),
            "oae": oae.item(), 
            "kl" : kl_loss.item()/len(data), 
            "recon": recon_loss.item()/(data.numel()),
            "loss" : (recon_loss.item() + kl_loss.item())/(data.numel())
        }, n=data.size(0))
        if batch_idx % 20 == 0: 
            logger.info('Eval Epoch: [{}/{} ({:.0f}%)]\t{}'.format(
                batch_idx, len(loader),
                100. * batch_idx / len(loader),
                str(meters)))

    logger.info('====> {} set loss: {}'.format(
          "Test".capitalize().ljust(6), str(meters)))
    return meters

    
def eval(config, eval_config, output_dir, passes):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(output_dir,'eval.log')),
            logging.StreamHandler()
        ])

    model = cvae.models[config.model.type](config)
    model.to(config.device)

    h_train = perturbations.hs[config.perturbation.train_type](config.perturbation)
    h_test = perturbations.hs[config.perturbation.test_type](config.perturbation)
    train_loader, test_loader, val_loader = datasets.loaders[eval_config.dataset.type](eval_config)
    
    resume = config.resume
    if resume == 'best': 
        logger.info("using best checkpoint")
        resume = os.path.join(output_dir, "checkpoints", "checkpoint_best.pth")
    elif resume == 'latest': 
        logger.info("using latest checkpoint")
        resume = os.path.join(output_dir, "checkpoints", "checkpoint_latest.pth")
    elif resume is None: 
        logger.info(f"no checkpoint specificied, using latest checkpoint")
        resume = os.path.join(output_dir, "checkpoints", "checkpoint_latest.pth")

    d = torch.load(resume)
    logger.info(f"Resume model checkpoint {d['epoch']}...")
    model.load_state_dict(d["model_state_dict"])
    
    if config.dataparallel: 
        model.dataparallel()

    args = (eval_config, model, logger)
    with torch.no_grad(): 
        meters = []
        for i in range(passes): 
            meters.append(loop(*args, test_loader, h_test))
        for k in meters[0].AMs.keys(): 
            stats = np.array([m[k] for m in meters])
            logger.info(f"{k}: {stats.mean():.4f} +- {stats.std():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        description='Train script options',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', type=str,
                        help='path to config file',
                        default='config.json', required=False)
    parser.add_argument('-ce', '--config-eval', type=str,
                        help='path to config file',
                        default='eval_config.json', required=False)
    parser.add_argument('-dp', '--dataparallel', 
                        help='data paralllel flag', action='store_true')
    parser.add_argument('--passes', type=int, default=1)
    parser.add_argument('--resume', default=None, help='path to checkpoint')
    args = parser.parse_args()
    config_dict = utilities.get_config(args.config)
    config_dict['dataparallel'] = args.dataparallel

    assert os.path.splitext(os.path.basename(args.config))[0] == config_dict['model']['model_dir']

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    output_dir = os.path.join(config_dict['output_dir'], 
                              config_dict['model']['model_dir'])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for s in ['images', 'checkpoints']: 
        extra_dir = os.path.join(output_dir,s)
        if not os.path.exists(extra_dir):
            os.makedirs(extra_dir)

    # keep the configuration file with the model for reproducibility
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, sort_keys=True, indent=4)

    config_dict['resume'] = args.resume

    # make the load argument optional
    if 'load' not in config_dict['model']: 
        config_dict['model']['load'] = False

    config = utilities.config_to_namedtuple(config_dict)
    eval_config_dict = utilities.get_config(args.config_eval)
    eval_config = utilities.config_to_namedtuple(eval_config_dict)
    eval(config, eval_config, output_dir, args.passes)
