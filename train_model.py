

import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
# import datasets
import time
import resnet
import os
import shutil

dataset = 'cifar10'
batch_size = 128
best_prec1 = 0
# lr = 0.1 * (batch_size / 256)
lr = 0.001
momentum = 0.9
weight_decay = 1e-4
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
arch = 'resnet34'


def train_epoch(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (x, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        x = x.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(x)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output, target)

        losses.update(loss.item(), x.size(0))
        top1.update(prec1.item(), x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t)'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))

    return model


def validate_epoch(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1 = accuracy(output, target)
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1))

        print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = torch.argmax(output, dim=1)
        correct = pred.eq(target)
        num_correct = correct.float().sum(0)
        return num_correct.mul_(100.0 / batch_size)


def save_checkpoint(state, is_best, filename='saved_models/'+arch+'checkpoint.pth.tar'):
  torch.save(state, filename)
  if is_best:
    shutil.copyfile(filename, 'saved_models/'+arch+'model_best.pth.tar')


def main():
    global lr, best_prec1
    epochs = 50
    model = resnet.ResNet50(10)
    # model = nn.Sequential(nn.BatchNorm2d(num_features=3, affine=False), pretrained)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 10)

    model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[25, 50, 75], gamma=0.2)

    # train_loader, val_loader, class_names = datasets.get_train_test_dataset(dataset, batch_size, True)
    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(root='data/cifar10-data', train=True, download=True
                                                                , transform=transform)
                                               , batch_size=batch_size, shuffle=True)
    test_indices = list(range(100))
    val_loader = torch.utils.data.DataLoader(datasets.CIFAR10(root='data/cifar10-data', train=False, download=True
                                                               , transform=transform)
                                              , sampler=torch.utils.data.sampler.SubsetRandomSampler(test_indices)
                                              , batch_size=batch_size)

    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch, lr)

        # train for one epoch
        train_epoch(train_loader, model, criterion, optimizer, epoch)
        # lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate_epoch(val_loader, model, criterion)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best)


if __name__=="__main__":
    main()


