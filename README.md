# Learning perturbation sets for robust machine learning
*A repository that implements  perturbation learning code, capable of learning perturbation sets from data for MNIST, CIFAR10, and Multi-Illumination datasets. Created by [Eric Wong](https://riceric22.github.io) with [Zico Kolter](http://zicokolter.com), with the code structure loosely based off of the [robustness repostory here](https://github.com/MadryLab/robustness). See our paper on arXiv [here][paper] and our corresponding [blog post][blog].*

[paper]: https://arxiv.org/abs/2007.08450
[blog]: https://locuslab.github.io/2020-07-20-perturbation/

## News
+ 7/16/2020 - Paper and blog post released 

# Overview
One of the core tenents of making machine learning models that are robust to adversarial attacks is to define the threat model that contains all possible perturbations, which is critical for performing a proper robustness evaluation. However, well-defined threat models are have been largely limited to mathematically nice sets that can be described a priori, such as the Lp ball or Wasserstein metric, whereas many real-world transformations may be impossible to define mathematically. This work aims to bridge this gap, by learning perturbation sets as being generated from an Lp ball in an underlying latent space. This simple characterization of a perturbation set allows us to leverage state of the art approaches in adversarial training directly in the latent space, while at the same time capturing complex real-world perturbations. 

# Repository information

## Configuration files
+ `configs/` contains configuration files to train perturbation sets 
+ `configs_eval/` contains configuration files to evaluate perturbation sets
+ `configs_robust/` contains configuration files to train robust models (e.g. with perturbation sets and data augmentation baselines)
+ `configs_attack/` contains configuration files to evaluate robust models (e.g. with standard and robust metrics)

## Usage
+ To train a perturbation set with configuration `<perturbation>`, use `python train.py --config configs/<perturbation>.json`
+ To evaluate a perturbation set with configuration `<perturbation>` on a metric `<metric>, use `python eval.py --config configs/<perturbation>.json --eval-config configs_eval/<metric>.json`
+ To gather perturbation statistics on the validation set (e.g. to get the maximum radius for the latent space) with configuration `<perturbation>`, use `python latent_distances.py <perturbation>`
+ To train a robust model with `<method>`, use `python robust_train.py --config configs_robust/<method>.json`
+ To evaluate a robust model trained with `<method>` on an attack `<attack>`, use `python robust_eval.py --config configs_robust/<method>.json --config-attack configs_attack/<attack>.json`

A full list of all commands and configurations to run all experiments in the paper is in `all.sh`. 

### Generating perturbation datasets
We include some convenience scripts for generating the CIFAR10 common corruptions and Multi-Illumination in `datasets/`, which are based on the corresponding official repositories 
+ CIFAR10 common corruptions: https://github.com/hendrycks/robustness
+ Multi illumination: https://github.com/lmurmann/multi_illumination

## Model weights
+ Pretrained model weights for all learned perturbation sets and downstream robustly trained classifiers can be found here: https://drive.google.com/drive/folders/1JEY2wgtERcj7TjCGDzPfZ_QAsDYIAJgI?usp=sharing

Within this folder, `perturbation_sets/` contains the model weights for the following learned perturbation sets: 
+ CIFAR10 common corruptions learned perturbation sets using the three strategies from Table 5+6
+ MI learned perturbation sets at varying resolutions from Table 8+9

Additional, `robust_models/` contains the model weights for the robust models trained using our CVAE perturbation sets: 
+ CIFAR10 classifier trained with CVAE adversarial training from Table 1
+ CIFAR10 classifier trained with CVAE data augmentation from Table 1
+ Certifiably robust CIFAR10 classifier using randomized smoothing at the three noise levels in Table 7. 
+ MI segmentation model trained with CVAE adversarial training from Table 10
+ MI segmentation model trained with CVAE data augmentation from Table 10
+ Certifiably robust MI segmentation model using randomized smoothing as reported in Figure 16. 

## Contents of implementation 
+ `perturbation_learning/` contains the main components of the CVAE perturbation sets. 
    + `cvae.py` contains the general CVAE implementation for a generic encoder, decoder, and prior network. The CVAE is further implemented by defining a module with these corresponding components as follows: 
        + `mnist_conv.py`, `mnist_fc.py`, and `mnist_stn.py` implement the networks for MNIST perturbation sets, with the last one using `STNModule.py` to implement the spatial transformer. 
        + `cifar10_rectangle.py` implements residual networks for the CIFAR10 common corruptions perturbation sets
        + `mi_unet.py` implements UNets for multi-illumination perturbation sets, using the UNet components defined in `unet_parts.py`. 
        + `scaled_tanh.py` implements the scaled tanh activation function used to stabilize the log variance prediction
    + `datasets.py` contains the dataloaders for the MNIST/CIFAR10/MI datasets and their corresponding perturbed datasets
    + `perturbations.py` contains the MNIST perturbations defined by PyTorch transform 
+ `robustness/` contains the main components of the robustness experiments using these CVAE perturbation sets
    + `classifiers.py` aggregates the models and interfaces to a config file
        + `wideresnet.py` implements a standard WideResnet for CIFAR10 classification
        + `unet_model.py` implements a standard UNet for MI material segmentation using `unet_parts.py`
    + `smoothing_core.py` implements a generic randomized smoothing prediction and certification wrapper for CVAE models
+ `datasets/` contains come convenience scripts for fetching and generating the CIFAR10 common corruptions dataset and the MI dataset
