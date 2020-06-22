import torch
from torchvision import datasets, transforms
from torchvision.datasets.folder import DatasetFolder
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import os
import numpy as np
import glob
import random
from PIL import Image

#### Joint transformations ####

#defined for numpy arrays of dimension (h,w) and (h,w,ch)
def crop(image, i, j, h, w): 
    # print(i,j,h,w)
    return image[i:i+h,j:j+w] 
def center_crop(image, output_sz): 
    crop_height, crop_width = output_sz
    image_height, image_width = image.shape[:2]
    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))
    return crop(image, crop_top, crop_left, crop_height, crop_width)
def hflip(image): 
    return np.flip(image,1)
def pad(image, p): 
    if image.ndim == 3: 
        padding = ((p,p),(p,p),(0,0))
    elif image.ndim == 2: 
        padding = ((p,p),(p,p))
    else: 
        raise ValueError
    return np.pad(image, padding, mode='constant')

class JointPadFlipAndCrop():
    def __init__(self, output_sz, pad, random_flip=True):
        self.output_sz = output_sz
        self.random_flip = random_flip
        self.p = pad

    def __call__(self, *images):
        output_sz = self.output_sz

        # Random crop
        images = [pad(image,self.p) for image in images]
        i, j, h, w = transforms.RandomCrop.get_params(TF.to_pil_image(images[0].astype(np.uint8)), output_size=output_sz)
        images = [crop(image, i, j, h, w) for image in images]

        # Random horizontal flipping
        if self.random_flip and random.random() > 0.5:
            images = [hflip(image).copy() for image in images]

        return images

class TensorJointPadFlipAndCrop(JointPadFlipAndCrop):
    def __call__(self, *images):
        images = [image.permute(1,2,0).numpy() for image in images]
        super(TensorJointPadFlipAndCrop, self).__call__(*images)
        return [torch.from_numpy(image).permute(2,0,1) for image in images]

#### MNIST dataloaders ####

def mnist_loaders(config):
    kwargs = {'num_workers': 1, 'pin_memory': True} if config.device == 'cuda' else {}
    t = transforms.ToTensor()
    if config.dataset.padding > 0:
        t = transforms.Compose([
                transforms.Pad(config.dataset.padding),
                transforms.ToTensor()
            ])
    all_data = datasets.MNIST(config.dataset.data_path, train=True, download=True, transform=t)

    seed = torch.get_rng_state()
    torch.manual_seed(0)
    train_split, val_split = torch.utils.data.random_split(all_data, [59000, 1000])
    torch.set_rng_state(seed)

    train_loader = torch.utils.data.DataLoader(
        train_split,
        batch_size=config.eval.batch_size, shuffle=False, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        val_split,
        batch_size=config.training.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(config.dataset.data_path, train=False, transform=t),
        batch_size=config.eval.batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader, val_loader

#### Extend TensorDataset for MNIST and CIFAR10 ####

class DatasetSampler(TensorDataset):
    def __init__(self, *tensors, k=20, joint_transform=None):
        super(DatasetSampler, self).__init__(*tensors)
        if tensors[-1].size(-1) < k:
            raise ValueError
        self.k = k
        self.joint_transform = joint_transform

    def __getitem__(self, index):
        t1,t2,t3 = super(DatasetSampler, self).__getitem__(index)
        i = torch.randint(0, self.k, (1,))
        t3 = t3[:,:,:,i].squeeze(-1)
        if self.joint_transform: 
            t1, t3 = self.joint_transform(t1,t3)
        return t1,t2,t3 

class DatasetSamplerNoOriginal(DatasetSampler): 
    def __getitem__(self, index): 
        _,t2,t3 = super(DatasetSampler, self).__getitem__(index)
        p = torch.randperm(self.k)[:2]
        t1, t3 = t3[:,:,:,p[0]],t3[:,:,:,p[1]]
        if self.joint_transform: 
            t1, t3 = self.joint_transform(t1,t3)
        return t1,t2,t3

class DatasetSamplerWithOriginal(DatasetSampler): 
    def __getitem__(self, index): 
        t1,t2,t3 = super(DatasetSampler, self).__getitem__(index)
        p1, p2 = torch.randperm(self.k+1)[:2]
        if p1 == self.k: 
            t1_ = t1
            t3_ = t3[:,:,:,p2]
        elif p2 == self.k: 
            t1_ = t3[:,:,:,p1]
            t3_ = t1
        else: 
            t1_ = t3[:,:,:,p1]
            t3_ = t3[:,:,:,p2]
        if self.joint_transform: 
            t1_, t3_ = self.joint_transform(t1_,t3_)
        return t1_,t2,t3_

class DatasetSamplerGroup(DatasetSampler): 
    def __getitem__(self, index): 
        _,t2,t3 = super(DatasetSampler, self).__getitem__(index)
        if self.joint_transform: 
            t3 = [self.joint_transform(t3_) for t3_ in t3]
        return t3,t2,t3

class DatasetAll(TensorDataset): 
    def __init__(self, X, y, perturbations, k=20, return_index=False, 
                joint_transform=None): 
        if perturbations.size(0) < k: 
            raise ValueError
        perturbations = perturbations[:,:,:,:,:k]
        perturbations = perturbations.permute(0,4,1,2,3).reshape(-1,*X.size()[1:])
        super(DatasetAll, self).__init__(perturbations)
        self.Xy = TensorDataset(X,y)
        self.k = k
        self.return_index = return_index
        self.joint_transform=joint_transform

    def __getitem__(self, index):
        t3, = super(DatasetAll, self).__getitem__(index)
        # print(index, self.k, index//self.k)
        t1,t2 = self.Xy.__getitem__(int(index//self.k))
        if self.joint_transform is not None: 
            t1,t3 = self.joint_transform(t1,t3)
        if self.return_index:
            return t1,t2,t3,index
        else:
            return t1,t2,t3

def mnist_rts_limited(config, return_index=False):
    assert config.perturbation.test_type == "rts"
    assert config.perturbation.angle == 45
    assert config.perturbation.scale[0] == 0.7
    assert config.perturbation.scale[1] == 1.3
    assert config.perturbation.crop_sz == 42
    assert config.perturbation.padding == 7
    assert config.dataset.padding == 7

    t = transforms.Compose([
                transforms.Pad(config.dataset.padding),
                transforms.ToTensor()
            ])

    perturbation_data = torch.load('datasets/mnist_rts20.pth')
    train_X, train_y = torch.load('datasets/mnist_tensors.pth')
    train_X = F.pad(train_X, (7,7,7,7),mode='constant',value=0)
    if config.dataset.type == "mnist_rts_limited_enumerate":
        all_data = DatasetAll(train_X, train_y, perturbation_data,
                              k=config.dataset.k, return_index=True)
    else:
        all_data = DatasetSampler(train_X, train_y, perturbation_data,
                              k=config.dataset.k)

    # fix validation split
    seed = torch.get_rng_state()
    torch.manual_seed(0)
    train_split, val_split = torch.utils.data.random_split(all_data, [59000, 1000])
    torch.set_rng_state(seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if config.device == 'cuda' else {}
    train_loader = torch.utils.data.DataLoader(train_split,
        batch_size=config.training.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_split,
        batch_size=config.eval.batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(config.dataset.data_path, train=False, transform=t),
        batch_size=config.eval.batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader, val_loader

#### CIFAR10 dataloader ####

def cifar10(config):
    if config.dataset.transforms == 'none':
        transform_train = transform_test = transforms.ToTensor()
    elif config.dataset.transforms == 'cropflip':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

    train_set = datasets.CIFAR10(
        config.dataset.data_path,
        train=True,
        download=True,
        transform=transform_train
    )

    # fix validation split
    seed = torch.get_rng_state()
    torch.manual_seed(0)
    train_split, val_split = torch.utils.data.random_split(train_set, [49000, 1000])
    torch.get_rng_state()

    train_loader = torch.utils.data.DataLoader(
        train_split,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_split,
        batch_size=config.eval.batch_size,
        shuffle=False,
        num_workers=2
    )

    test_set = datasets.CIFAR10(
        config.dataset.data_path,
        train=False,
        download=True,
        transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=config.eval.batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, test_loader, val_loader

#### CIFAR10 corruptions dataset ####

def load_cifar10c(config, train=True):
    mode = "train" if train else "test"
    data_dir = os.path.join(config.dataset.data_path,f"cifar-10-c-{mode}")

    perturbation_data = []
    for corruption in config.dataset.corruptions:
        for severity in config.dataset.severity:
            perturbation_data.append(np.load(os.path.join(data_dir,f"{corruption}_{severity}.npy")))
    return torch.cat([torch.from_numpy(td).unsqueeze(-1) for td in perturbation_data],dim=-1)

def cifar10c(config): 
    kwargs = {'num_workers': 4, 'pin_memory': True} if config.device == 'cuda' else {} 
    train_perturb_data = load_cifar10c(config, train=True)

    if config.dataset.cropflip: 
        jt_train = TensorJointPadFlipAndCrop(
            config.dataset.cropflip.crop, 
            config.dataset.cropflip.pad,
            random_flip=config.dataset.cropflip.flip)
    else: 
        jt_train = None

    train_data = datasets.CIFAR10(config.dataset.data_path, train=True, download=True)
    train_X = torch.from_numpy(train_data.data)
    train_y = torch.Tensor(train_data.targets).long()

    train_perturb_data = train_perturb_data.permute(0,3,1,2,4).float()/255.0
    train_X = train_X.permute(0,3,1,2).float()/255.0

    if config.dataset.type == "cifar10c_all":
        DatasetType = DatasetAll
    elif config.dataset.type == "cifar10c_nooriginal": 
        DatasetType = DatasetSamplerNoOriginal
    elif config.dataset.type == "cifar10c_withoriginal": 
        DatasetType = DatasetSamplerWithOriginal
    elif config.dataset.type == "cifar10c": 
        DatasetType = DatasetSampler
    else: 
        raise NotImplementedError
    all_train_data = DatasetType(train_X, train_y, train_perturb_data, 
                              k=config.dataset.k, joint_transform=jt_train)


    test_perturb_data = load_cifar10c(config, train=False)

    test_data = datasets.CIFAR10(config.dataset.data_path, train=False, download=True)
    test_X = torch.from_numpy(test_data.data)
    test_y = torch.Tensor(test_data.targets).long()

    test_perturb_data = test_perturb_data.permute(0,3,1,2,4).float()/255.0
    test_X = test_X.permute(0,3,1,2).float()/255.0

    all_test_data = DatasetType(test_X, test_y, test_perturb_data,
                                k=config.dataset.k)

    
    val_sz = int(len(all_train_data)/50)
    train_sz = len(all_train_data) - val_sz

    seed = torch.get_rng_state()
    torch.manual_seed(0)
    train_split, val_split = torch.utils.data.random_split(all_train_data, [train_sz, val_sz])
    torch.set_rng_state(seed)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_split,
        batch_size=config.training.batch_size,
        shuffle=True,
        **kwargs,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_split,
        batch_size=config.eval.batch_size,
        shuffle=False,
        **kwargs,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=all_test_data,
        batch_size=config.eval.batch_size,
        shuffle=False,
        **kwargs,
    )

    return train_loader, test_loader, val_loader

#### Multi illumination dataset loader ####
# Largest dimension is 431 x 348

def group_np_loader(frames): 
    if frames.ndim == 3: 
        return Image.fromarray(frames)
    return [Image.fromarray(f) for f in frames]


def concat_np_dataset(root, view_list): 
    instances = []
    for fname in view_list: 
        instances.append(np.load(os.path.join(root, fname)))
    return np.concatenate(instances, 0)


class MultiIlluminationNumpy(DatasetFolder): 
    subset_list = {
        'train': [
            "dataset-train-mip-{}-batch-0.npy", 
            "dataset-train-mip-{}-batch-1.npy",
            "dataset-train-mip-{}-batch-2.npy",
            "dataset-train-mip-{}-batch-3.npy",
            "dataset-train-mip-{}-batch-4.npy"
        ], 
        'test': [
            "dataset-test-mip-{}-batch-0.npy"
        ], 
        'val': [
            "dataset-val-mip-{}-batch-0.npy"
        ]
    }

    materials_subset_list = {
        'train': [
            "dataset-train-materials-mip-{}-batch-0.npy", 
            "dataset-train-materials-mip-{}-batch-1.npy",
            "dataset-train-materials-mip-{}-batch-2.npy",
            "dataset-train-materials-mip-{}-batch-3.npy",
            "dataset-train-materials-mip-{}-batch-4.npy"
        ], 
        'test': [
            "dataset-test-materials-mip-{}-batch-0.npy"
        ], 
        'val': [
            "dataset-val-materials-mip-{}-batch-0.npy"
        ]
    }

    def __init__(self, root, subset='train', group=True, transform=None, target_transform=None, split='drylab', 
                        joint_transform=None, mip=5): 
        # override init
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        # classes, class_to_idx = self._find_classes(self.root)
        if split == 'drylab': 
            subset_list = self.subset_list
            target_list = self.materials_subset_list
        else: 
            raise ValueError
        view_list = [sl.format(mip) for sl in subset_list[subset]]
        samples = concat_np_dataset(self.root, view_list)

        targets = [sl.format(mip) for sl in target_list[subset]]
        targets = concat_np_dataset(self.root, targets)

        self.loader = group_np_loader
        self.samples = samples
        self.targets = targets
        self.group = group
        
        self.joint_transform=joint_transform
        
    def __len__(self):
        # return sum(len(s[0]) if isinstance(s[0], list) else 1 for s in self.samples)
        return super(MultiIlluminationNumpy, self).__len__()
                   
class MultiIlluminationSampler(MultiIlluminationNumpy): 
    def __init__(self, root, **kwargs):

        # override init
        super(MultiIlluminationSampler, self).__init__(root, group=True, **kwargs)
    
    def __getitem__(self, index): 
        data = self.samples[index]

        i,j = np.random.permutation(data.shape[0])[:2]
        sample0, sample1 = data[i], data[j]
        target = self.targets[index]

        if self.joint_transform is not None: 
            sample0, sample1, target = self.joint_transform(sample0, sample1, target)
        if self.transform is not None: 
            sample0, sample1 = self.transform(sample0), self.transform(sample1)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample0, target, sample1
                   
class MultiIlluminationFirst(MultiIlluminationNumpy): 
    def __init__(self, root, **kwargs):

        # override init
        super(MultiIlluminationFirst, self).__init__(root, group=True, **kwargs)
    
    def __getitem__(self, index): 
        data = self.samples[index]

        sample = data[0]
        target = self.targets[index]

        if self.joint_transform is not None: 
            sample, target = self.joint_transform(sample, target)
        if self.transform is not None: 
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, sample


class MultiIlluminationGroup(MultiIlluminationNumpy): 
    def __init__(self, root, **kwargs):

        # override init
        super(MultiIlluminationGroup, self).__init__(root, group=True, **kwargs)
    
    def __getitem__(self, index): 
        sample = self.samples[index]
        target = self.targets[index]

        if self.joint_transform is not None: 
            out = self.joint_transform(target, *sample)
            target, sample = out[0], out[1:]

        if self.transform is not None: 
            sample = torch.cat([self.transform(s).unsqueeze(0) for s in sample])

        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, sample


class MultiIlluminationAll(MultiIlluminationNumpy): 
    def __init__(self, root, **kwargs):
        # override init
        super(MultiIlluminationAll, self).__init__(root, group=True, **kwargs)
        self.k = 25
    
    def __getitem__(self, index): 
        n_scenes = self.k**2
        scene = index//n_scenes
        permutation = index%n_scenes
        sample1 = self.samples[scene, permutation//self.k]
        sample2 = self.samples[scene, permutation%self.k]
        target = self.targets[scene]

        if self.joint_transform is not None: 
            target, sample1, sample2 = self.joint_transform(target, sample1, sample2)

        if self.transform is not None: 
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample1, target, sample2

    def __len__(self):
        # return sum(len(s[0]) if isinstance(s[0], list) else 1 for s in self.samples)
        return super(MultiIlluminationNumpy, self).__len__()*(self.k**2)


def multi_illumination(config): 
    t = transforms.Compose([
        transforms.ToTensor()
    ])
    if config.dataset.mode == 'aug': 
        if config.dataset.mip == 2: 
            D = MultiIlluminationFolder
        else: 
            D = MultiIlluminationSampler
    elif config.dataset.mode == 'first': 
        D = MultiIlluminationFirst
    elif config.dataset.mode == 'group': 
        D = MultiIlluminationGroup
    elif config.dataset.mode == 'all': 
        D = MultiIlluminationAll
    else: 
        raise ValueError()

    if config.dataset.cropflip: 
        # jt_train = JointFlipAndCrop((100,150), center_crop=False, random_flip=True)
        # jt_test = JointFlipAndCrop((100,150), center_crop=True, random_flip=False)
        jt_train = JointPadFlipAndCrop(
            config.dataset.cropflip.crop, 
            config.dataset.cropflip.pad,
            random_flip=config.dataset.cropflip.flip)
        jt_test = None
    else: 
        jt_train, jt_test = None, None

    train_dataset = D(config.dataset.data_path, subset='train', transform=t,  
                        joint_transform=jt_train, split=config.dataset.split, 
                        mip=config.dataset.mip)
    test_dataset = D(config.dataset.data_path, subset='test', transform=t,  
                        joint_transform=jt_test, split=config.dataset.split, 
                        mip=config.dataset.mip)
    val_dataset = D(config.dataset.data_path, subset='val', transform=t,  
                        joint_transform=jt_test, split=config.dataset.split, 
                        mip=config.dataset.mip)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.training.batch_size, 
        shuffle=True, num_workers=2
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.eval.batch_size, 
        shuffle=False, num_workers=2
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.eval.batch_size, 
        shuffle=False, num_workers=2
    )
    return train_loader, test_loader, val_loader

loaders = {
    "mnist" : mnist_loaders,
    "mnist_rts_limited": mnist_rts_limited,
    "cifar10": cifar10,
    "cifar10c": cifar10c,
    "cifar10c_all": cifar10c, 
    "cifar10c_nooriginal": cifar10c, 
    "cifar10c_withoriginal": cifar10c, 
    "multi_illumination": multi_illumination
}
