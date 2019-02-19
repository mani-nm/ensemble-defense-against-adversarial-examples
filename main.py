"""
# Driver to run adversarial examples and create
# AE samples from clean data and pre-trained models
# Author: Nag Mani
# Created: 2/18/2019
"""

import os
import argparse
import torch

import torch
import torch.nn as nn
from torchvision import models

import datasets
import model_evaluation

parser = argparse.ArgumentParser(description="Pytorch code: Adversarial Examples Generation")
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
parser.add_argument('--dataset', default='imagenet')
parser.add_argument('--data_dir', default='./data', help='Root data directory')
parser.add_argument('--arch', default='resnet152')
parser.add_argument('--gpu', type=int, default='0')
parser.add_argument('--adv', default='deepfool',  help='deepfool, bim, fgsm, cw')
parser.add_argument('--print_freq', type=int, default=10, help='Print frequency')

args = parser.parse_args()


def main():

    # Initialize default variables
    pretrained = False
    epochs = 30
    momentum = 0.9
    wd = 1e-4
    out_dir = 'ae_images/' + args.dataset + '/' + args.arch + '/'

    path = os.path.join(os.getcwd(), out_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(path)

    torch.cuda.manual_seed(2019)
    torch.cuda.set_device(args.gpu)

    # Loading the dataset
    train_loader, val_loader, test_loader = datasets.get_train_test_dataset(args.dataset, args.batch_size,'/data')

    # Loading pretrained model
    model = getattr(models, args.arch)(num_classes=2, pretrained=True)
    # prepend a BN layer w/o learnable params to perform data normalization
    # as we disabled data normalization in data iter in order to make the
    # interface compatible with attack APIs that requires data in [0.0, 1.0]
    # range.
    model = nn.Sequential(nn.BatchNorm2d(num_features=3, affine=False), model)
    model.cuda()
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), 0.1,
                                momentum=momentum,
                                weight_decay=wd)
    #
    # checkpoint = torch.load('saved_models/checkpoint.pth.tar')
    # start_epoch = checkpoint['epoch']
    # best_prec = checkpoint['best_prec1']
    # model.load_state_dict(checkpoint['state_dict'], strict=False)
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # print("=> Checkpoint Loaded")

    acc = model_evaluation.test_accuracy(test_loader, model, args)
    print("Final Accuracy:", acc)







if __name__=='__main__':
    main()

