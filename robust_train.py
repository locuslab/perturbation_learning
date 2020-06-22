import waitGPU
waitGPU.wait(gpu_ids=[7,8,9], nproc=0, interval=120)

import argparse
import json
import os
import numpy as np
import logging

import utilities

from robustness import classifiers
from attacks import attacks
from perturbation_learning import datasets

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.transforms import ToTensor
from matplotlib import cm

TRAIN_MODE = 'train'
VAL_MODE = 'val'
TEST_MODE = 'test'

def optimizers(config, parameters): 
    if config.training.optimizer == "adam": 
        return optim.Adam(parameters, 
            lr=1, 
            weight_decay=config.training.weight_decay
        )
    elif config.training.optimizer == "sgd": 
        return optim.SGD(parameters, 
            lr=1, 
            weight_decay=config.training.weight_decay, 
            momentum=config.training.momentum
        )
    else:
        raise ValueError


def save_chkpt(model, optimizer, epoch, test_loss, name, dp): 
    if dp: 
        model = model.module
    torch.save({
        "model_state_dict": model.state_dict(), 
        "optimizer_state_dict": optimizer.state_dict(), 
        "epoch": epoch,
        "test_loss": test_loss
    }, name)

def cross_entropy(out, tar): 
    return F.cross_entropy(out, tar, reduction='none')

def loop(config, model, optimizer, attack, lr_schedule, logger, output_dir, epoch, loader, mode=TRAIN_MODE, 
         topk = 1, criterion=cross_entropy): 
    meters = utilities.MultiAverageMeter([
        "nat loss", "rob loss", "all loss", "nat err", "rob err"
    ])

    for batch_idx, batch in enumerate(loader):
        data = batch[0]
        target = batch[1].long()
        epoch_idx = epoch + (batch_idx + 1) / len(loader)
        lr = lr_schedule(epoch_idx)
        optimizer.param_groups[0].update(lr=lr)

        data = data.to(config.device)
        target = target.to(config.device)
        hdata = attack(data, target, model)

        if mode == TRAIN_MODE: 
            optimizer.zero_grad()

        robust_output = model(hdata)
        #robust_err = (robust_output.max(1)[1] != target).float().mean()
        # print((robust_output.topk(topk,1).indices != target.unsqueeze(1)).all(1).float().mean())
        # print((robust_output.topk(topk,1).indices != target.unsqueeze(1)).any(1).float().mean())
        robust_err = (robust_output.topk(topk,1).indices != target.unsqueeze(1)).all(1).float().mean()
        robust_loss = criterion(robust_output, target)

        if config.attack.type == 'cvae_attack':
        # or config.attack.type == 'cvae_aug': 
            output = model(data)
            err = (output.max(1)[1] != target).float().mean()
            loss = criterion(output, target)

            overall_loss = torch.max(loss, robust_loss).mean()
            robust_loss = robust_loss.mean()
            loss = loss.mean()
        else: 
            output = robust_output
            overall_err = err = robust_err
            overall_loss = loss = robust_loss = robust_loss.mean()

        if mode == TRAIN_MODE: 
            overall_loss.backward()
            optimizer.step()

        meters.update({
            "nat loss": loss.item(), 
            "rob loss": robust_loss.item(), 
            "nat err": err.item(), 
            "rob err": robust_err.item(), 
            "all loss": overall_loss.item()
        }, n=data.size(0))

        if mode == TRAIN_MODE and batch_idx % config.training.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\t{}'.format(
                epoch, batch_idx, len(loader),
                100. * batch_idx / len(loader),
                str(meters)))

        if mode == TEST_MODE and batch_idx == 0 and (epoch+1) % config.eval.sample_interval == 0:
            n = min(data.size(0)//2, 8)
            hcomparison = torch.cat([data[:n],
                                     hdata[:n],
                                     data[n:2*n],
                                     hdata[n:2*n]])
            save_image(hcomparison.cpu(), os.path.join(output_dir, 'images', f'hadversarial_{epoch}.png'), nrow=n)

            if config.eval.plot_segmentation: 
                # https://discuss.pytorch.org/t/how-to-visualize-segmentation-output-multiclass-feature-map-to-rgb-image/26986/2
                seg = torch.cat([target[:n], 
                                 output.max(1)[1][:n], 
                                 robust_output.max(1)[1][:n]]).cpu()

                cmap = cm.tab20(range(config.model.n_classes))[:,:3]
                seg = torch.cat([ToTensor()(cmap[s]).unsqueeze(0) for s in seg], dim=0)
                save_image(seg, os.path.join(output_dir, 'images', f'segmentation_{epoch}.png'), nrow=n)



    logger.info('====> {} set: {} {} lr {:.4f}'.format(
          mode.capitalize().ljust(6), epoch, str(meters), lr))
    return meters


def train(config, output_dir):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(output_dir,'output.log')),
            logging.StreamHandler()
        ])

    model = classifiers.models[config.model.type](config)
    model.to(config.device)

    attack = attacks[config.attack.type](config)
    train_loader, test_loader, val_loader = datasets.loaders[config.dataset.type](config)
    
    optimizer = optimizers(config, model.parameters())
    # optimizer = optimizers[config.training.optimizer](model.parameters(), 
    #                 lr=1, weight_decay=config.training.weight_decay)
                    #momentum=config.training.momentum)

    lr_schedule = lambda t: np.interp([t], *config.training.step_size_schedule)[0]
    best_val_err = 1

    start_epoch = 0
    if config.resume is not None: 
        d = torch.load(config.resume)
        logger.info(f"Resume model checkpoint {d['epoch']}...")
        optimizer.load_state_dict(d["optimizer_state_dict"])
        model.load_state_dict(d["model_state_dict"])
        start_epoch = d["epoch"] + 1

        try: 
            d = torch.load(os.path.join(output_dir, 'checkpoints', 'checkpoint_best.pth'))
            best_test_loss = d["test_loss"]
        except: 
            logger.info("No best checkpoint to resume test loss from")
    
    if config.dataparallel: 
        model = nn.DataParallel(model)

    # these remain the same throughout train/validation/test
    args = (config, model, optimizer, attack, lr_schedule, logger, output_dir)
    for epoch in range(start_epoch, config.training.epochs): 
        # Training
        model.train()
        loop(*args, epoch, train_loader, mode=TRAIN_MODE)

        # Testing
        model.eval()
        with torch.no_grad(): 
            val_meters = loop(*args, epoch, val_loader, mode=VAL_MODE)
            test_meters = loop(*args, epoch, test_loader, mode=TEST_MODE)
            val_err = val_meters.AMs['rob err'].avg

            if config.training.checkpoint_interval != "skip": 
                if (epoch+1) % config.training.checkpoint_interval == 0: 
                    save_chkpt(model, optimizer, epoch, val_err, 
                               os.path.join(output_dir, 'checkpoints', f'checkpoint_{epoch}.pth'), 
                               config.dataparallel)

                if val_err < best_val_err: 
                    save_chkpt(model, optimizer, epoch, val_err, 
                               os.path.join(output_dir, 'checkpoints', 'checkpoint_best.pth'), 
                               config.dataparallel)
                    best_val_err = val_err

                save_chkpt(model, optimizer, epoch, val_err, 
                           os.path.join(output_dir, 'checkpoints', 'checkpoint_latest.pth'), 
                           config.dataparallel)   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        description='Train script options',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-cr', '--config-robust', type=str,
                        help='path to config file',
                        default='config.json', required=False)
    parser.add_argument('-dp', '--dataparallel', 
                        help='data parallel flag', action='store_true')
    parser.add_argument('--resume', default=None, help='path to checkpoint')
    args = parser.parse_args()
    config_dict = utilities.get_config(args.config_robust)
    config_dict['dataparallel'] = args.dataparallel

    assert os.path.splitext(os.path.basename(args.config_robust))[0] == config_dict['model']['model_dir']

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

    config = utilities.config_to_namedtuple(config_dict)
    train(config, output_dir)
