"""
# Reads and prepares the dataset for attack
# Author: Nag Mani
# Created: 2/18/2019
"""

import os
import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
import numpy as np


def imagenet(batch_size, data_root= 'D:\\Research\\AE\\Adversarial_Examples\\data', limit=None, attack_name=None
             , **kwargs):

    def getclasses(data):
        classes = os.listdir(data + '/imagenet/val')
        classes.sort()
        f = open(data + "/synset_words.txt", "r")
        s2n = {}
        for line in f:
            parts = line.split(" ")

            s2n[parts[0]] = " ".join(parts[1:]).rstrip()

        return s2n

    synset_to_name = getclasses(data_root)

    valdir = os.path.join(data_root, 'imagenet/val')
    num_workers = kwargs.setdefault('num_workers', 2)
    kwargs.pop('input_size', None)
    _normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    np.random.seed(2019)
    train_ids = np.random.choice(range(20000, 25000), 1000, replace=False)
    test_ids = list(set(range(20000, 25000)) - set(train_ids))
    test_ids = test_ids[:500]
    print(len(train_ids), len(test_ids))

    train_dataset =datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0,0,0], std=[255,255,255])
    ]))

    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0,0,0], std=[255,255,255])
    ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(train_ids),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(test_ids),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    class_names = val_dataset.classes

    return train_loader, val_loader, class_names, synset_to_name


# def getBvB(batch_size, data_root='/tmp/public_dataset/pytorch', train=True, test=True, extras=False, **kwargs):
#     data_root = bird_or_bicycle.dataset.default_data_root()
#     num_workers = kwargs.setdefault('num_workers', 2)
#     kwargs.pop('input_size', None)
#
#     _normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                       std=[0.229, 0.224, 0.225])
#
#     traindirs = [os.path.join(data_root, partition)
#                  for partition in ['extras']]
#     # Use train as validation because it is IID with the test set
#     valdir = os.path.join(data_root, 'train')
#
#     testdir = os.path.join(data_root, 'test')
#
#     ds = []
#
#     train_dataset = [datasets.ImageFolder(
#         traindir,
#         transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             # _normalize,
#         ]))
#         for traindir in traindirs]
#     if len(train_dataset) == 1:
#         train_dataset = train_dataset[0]
#     else:
#         train_dataset = torch.utils.data.ConcatDataset(train_dataset)
#
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=batch_size, shuffle=True,
#         num_workers=num_workers, pin_memory=True)
#
#     val_loader = torch.utils.data.DataLoader(
#         datasets.ImageFolder(valdir, transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             # _normalize,
#         ])),
#         batch_size=batch_size, shuffle=False,
#         num_workers=num_workers, pin_memory=True)
#
#     test_loader = torch.utils.data.DataLoader(
#         datasets.ImageFolder(testdir, transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             # _redundant_normalize,
#         ])),
#         batch_size=batch_size, shuffle=False,
#         num_workers=num_workers, pin_memory=True)
#     ds.append(train_loader)
#     ds.append(val_loader)
#     ds.append(test_loader)
#     return ds


def getCIFAR10(batch_size, data_root, train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    # num_workers = kwargs.setdefault('num_workers', 2)
    kwargs.pop('input_size', None)
    _normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    test_indices = list(range(2000))
    TF = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
         # transforms.Normalize((0.0, 0.0, 0.0), (255, 255, 255))
         ]
    )

    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=True, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=True, **kwargs)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=False, download=True,
                transform=TF),
            # sampler=torch.utils.data.sampler.SubsetRandomSampler(test_indices),
            batch_size=batch_size, shuffle=False, **kwargs)

    if train and val:
        return train_loader, test_loader
    elif train and not val:
        return train_loader, []
    else:
        return test_loader, []


def getCIFAR100(batch_size, data_root, train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar100-data'))
    # num_workers = kwargs.setdefault('num_workers', 2)
    kwargs.pop('input_size', None)
    _redundant_normalize = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ]
    )
    TF = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
         # transforms.Normalize((0.0, 0.0, 0.0), (255, 255, 255))
         ]
    )
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=True, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=True, **kwargs)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=False, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=False, **kwargs)
    if train and val:
        return train_loader, test_loader
    elif train and not val:
        return train_loader, []
    else:
        return test_loader, []


def get_train_test_dataset(dataset, batch_size, train, dataroot='data/'):
    if dataset == 'cifar10':
        if train:
            train_loader, test_loader = getCIFAR10(batch_size, dataroot, train)
            class_dict = {0: 'plane', 1: 'car', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship',
                          9: 'truck'}
            return train_loader, test_loader, class_dict
        else:
            test_loader,_ = getCIFAR10(batch_size, dataroot, train)
            class_dict = {0: 'plane', 1: 'car', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship',
                          9: 'truck'}
            return test_loader, class_dict
    if dataset == 'cifar100':
        train_loader, test_loader = getCIFAR100(batch_size, data_root=dataroot, train=False, num_workers=4)
        return train_loader, test_loader
    if dataset == 'bvb':
        train_loader, val_loader, test_loader = getBvB(batch_size, data_root=dataroot, num_workers=4)
        return train_loader, val_loader, test_loader
    if dataset == 'imagenet':
        test_loader, labels_dict = imagenet(batch_size, data_root=dataroot, num_workers=8)
        return test_loader, labels_dict

    return [], []
