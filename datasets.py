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
import bird_or_bicycle


def getBvB(batch_size, data_root='/tmp/public_dataset/pytorch', train=True, test=True, extras=False, **kwargs):
    data_root = bird_or_bicycle.dataset.default_data_root()
    num_workers = kwargs.setdefault('num_workers', 2)
    kwargs.pop('input_size', None)

    _redundant_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])

    traindirs = [os.path.join(data_root, partition)
                 for partition in ['extras']]
    # Use train as validation because it is IID with the test set
    valdir = os.path.join(data_root, 'train')

    testdir = os.path.join(data_root, 'test')

    ds = []

    train_dataset = [datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # _redundant_normalize,
        ]))
        for traindir in traindirs]
    if len(train_dataset) == 1:
        train_dataset = train_dataset[0]
    else:
        train_dataset = torch.utils.data.ConcatDataset(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # _redundant_normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(testdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # _redundant_normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    ds.append(train_loader)
    ds.append(val_loader)
    ds.append(test_loader)
    return ds


def getCIFAR10(batch_size, data_root, train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    num_workers = kwargs.setdefault('num_workers', 2)
    kwargs.pop('input_size', None)
    _redundant_normalize = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ]
    )
    TF = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # _redundant_normalize
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            # _redundant_normalize
        ]),
    }
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=True, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=False, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getCIFAR100(batch_size, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar100-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    ds = []
    _redundant_normalize = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ]
    )
    TF = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # _redundant_normalize
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            # _redundant_normalize
        ]),
    }
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=True, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=False, download=True,
                transform=TF),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def get_train_test_dataset(dataset, batch_size, dataroot='/tmp/public_dataset/pytorch'):
    if dataset == 'cifar10':
        train_loader, test_loader = getCIFAR10(batch_size, data_root=dataroot, num_workers=4)
        return train_loader, test_loader
    if dataset == 'cifar100':
        train_loader, test_loader = getCIFAR100(batch_size, data_root=dataroot, num_workers=4)
        return train_loader, test_loader
    if dataset == 'imagenet':
        train_loader, val_loader, test_loader = getBvB(batch_size, data_root=dataroot, num_workers=4)
        return train_loader, val_loader, test_loader
    return [], []
