# import waitGPU
# waitGPU.wait(gpu_ids=[3], available_memory=40000) 

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

from robust_train import loop as normal_loop

from time import time
import datetime

# from https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy/comments
SMOOTH = 1e-6
def neg_iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.max(1)[1]  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    return -iou

    # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    # return thresholded  # Or thresholded.mean() if you are interested in average across the batch

def certify_loop(config, model, optimizer, attack, lr_schedule, logger, output_dir, epoch, loader, mode="test", 
                 topk = 1, criterion=None): 
    meters = utilities.MultiAverageMeter([
        "radius", "robust_acc"
    ])

    if mode != "test": 
        logging.info("skipping certify loop for computational reasons because not a test set")
        return meters

    if config.resume_certification: 
        certification_dir = config.resume_certification
        # checkpoint = torch.load(config.resume_certification)
        # all_predictions = checkpoint["predictions"]
        # all_radii = checkpoint["radii"]
    else: 
        # all_predictions = []
        # all_radii = []
        start_stamp = datetime.datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
        certification_dir = os.path.join(output_dir, f"{start_stamp}_certification")

    if not os.path.exists(certification_dir):
        os.makedirs(certification_dir)

    checkpoint_idx = 0
    for batch_idx, batch in enumerate(loader):
        data = batch[0]
        target = batch[1].long()

        radius_fname = os.path.join(certification_dir, f"batch{batch_idx}_radius.npy")
        predictions_fname = os.path.join(certification_dir, f"batch{batch_idx}_prediction.npy")
        if os.path.exists(radius_fname): 
            radius_np = np.load(radius_fname)
            predictions_np = np.load(predictions_fname)
            meters.update({
                "radius": radius_np.mean(), 
                "robust_acc" : (predictions_np == target.numpy()).mean()
                }, n=radius_np.size)
            logger.info(f"Batch {batch_idx} already computed")
            logger.info('[{}/{} ({:.0f}%)]\t{}'.format(
                batch_idx, len(loader),
                100. * batch_idx / len(loader),
                str(meters)))
            continue

        data = data.to(config.device)
        target = target.to(config.device)

        radius_np = []
        predictions_np = []
        for i,(x,label) in enumerate(zip(data, target)): 
            # if checkpoint_idx < len(all_predictions): 
            #     meters.update({
            #         "radius": all_radii[checkpoint_idx].mean().item(), 
            #         "robust_acc": (all_predictions[checkpoint_idx] == label.cpu()).float().mean().item()
            #         }, n=1)
            #     checkpoint_idx += 1
            #     continue

            # checkpoint_idx += 1

            before_time = time()
            prediction, radius = attack(x, label, model)
            after_time = time()
            correct = (prediction == label).float().mean()
            avg_radius = radius.mean()
            time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
            logger.info("{}\t{:.3}\t{:.3}\t{}".format(
                i, avg_radius, correct, time_elapsed))

            predictions_np.append(prediction.cpu().numpy())
            radius_np.append(radius.cpu().numpy())

            # torch.save({
            #     "predictions": all_predictions, 
            #     "radii": all_radii
            #     }, os.path.join(output_dir,f"{start_stamp}_certification.pth"))

            meters.update({
                "radius": avg_radius.item(), 
                "robust_acc": correct.item()
                }, n=prediction.numel())

        # radius_np = np.array([t.cpu().numpy() for t in all_radii[start_idx:checkpoint_idx]])
        # predictions_np = np.array([t.cpu().numpy() for t in all_predictions[start_idx:checkpoint_idx]])
        np.save(radius_fname, np.array(radius_np))
        np.save(predictions_fname, np.array(predictions_np))
        logger.info('[{}/{} ({:.0f}%)]\t{}'.format(
            batch_idx, len(loader),
            100. * batch_idx / len(loader),
            str(meters)))

    logger.info('====> {} set: {} '.format(
          mode.capitalize().ljust(6), str(meters)))
    return meters

def eval(config, attack_config, output_dir):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(output_dir, attack_config.attack.type + '_eval.log')),
            logging.StreamHandler()
        ])
    logger.info(f"config file: {config.model.model_dir}, attack type: {attack_config.attack.type}")
    logger.info(attack_config)

    model = classifiers.models[config.model.type](config)
    model.to(config.device)
    model.eval()
    dummy_opt = optim.SGD(model.parameters(), lr=1)

    attack = attacks[attack_config.attack.type](attack_config)
    if attack_config.attack.type == 'cvae_certify' or attack_config.attack.type == 'cvae_predict': 
        loop = certify_loop
    else: 
        loop = normal_loop

    if attack_config.criterion == "miou": 
        criterion = neg_iou_pytorch
    elif attack_config.criterion == "cross_entropy": 
        criterion = lambda a,b: F.cross_entropy(a,b,reduction="none")
    else: 
        raise ValueError

    train_loader, test_loader, val_loader = datasets.loaders[attack_config.dataset.type](attack_config)
    # attack = attacks[config.attack.type](config)
    # train_loader, test_loader, val_loader = datasets.loaders[config.dataset.type](config)

    best_val_err = 1

    resume = config.resume
    if resume == 'best': 
        logger.info("using best checkpoint")
        resume = os.path.join(output_dir, "../checkpoints", "checkpoint_best.pth")
    elif resume == 'latest': 
        logger.info("using latest checkpoint")
        resume = os.path.join(output_dir, "../checkpoints", "checkpoint_latest.pth")
    elif resume is None: 
        logger.info(f"no checkpoint specificied, using latest checkpoint")
        resume = os.path.join(output_dir, "../checkpoints", "checkpoint_latest.pth")

        #raise ValueError("Need to provide checkpoint via --resume argument to evaluate")

    d = torch.load(resume)
    logger.info(f"loading model checkpoint {d['epoch']}...")
    model.load_state_dict(d["model_state_dict"])
    
    if config.dataparallel: 
        model = nn.DataParallel(model)

    # these remain the same throughout train/validation/test
    args = (config, model, dummy_opt, attack, lambda x: 0, logger, output_dir)

    with torch.no_grad():  
        #val_meters = loop(*args, -1, val_loader, mode="val", topk=attack_config.topk, criterion=criterion)
        print(f"# of minibatches ==> {len(test_loader)}")
        test_meters = loop(*args, -1, test_loader, mode="test", topk=attack_config.topk, criterion=criterion)
        #val_err = val_meters.AMs['rob err'].avg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        description='Eval script options',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config-robust', type=str,
                        help='path to config file',
                        default='config.json', required=False)
    parser.add_argument('-ac', '--config-attack', type=str,
                        help='path to attack config file',
                        default='attack_config.json', required=False)
    parser.add_argument('-dp', '--dataparallel', 
                        help='data parallel flag', action='store_true')
    parser.add_argument('--resume', default=None, help='path to checkpoint')
    parser.add_argument('--resume-certification', default=None, help='path to certification file')
    args = parser.parse_args()
    config_dict = utilities.get_config(args.config_robust)
    config_dict['dataparallel'] = args.dataparallel

    assert os.path.splitext(os.path.basename(args.config_robust))[0] == config_dict['model']['model_dir']

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    output_dir = os.path.join(config_dict['output_dir'], 
                              config_dict['model']['model_dir'], 
                              "eval")

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
    config_dict['resume_certification'] = args.resume_certification
    config = utilities.config_to_namedtuple(config_dict)

    attack_config_dict = utilities.get_config(args.config_attack)
    attack_config = utilities.config_to_namedtuple(attack_config_dict)
    eval(config, attack_config, output_dir)
