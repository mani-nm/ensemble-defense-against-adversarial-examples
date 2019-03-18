import torch
import time
import shutil
import torch.nn as nn
import resnet
from datasets import get_train_test_dataset
import lib.adversary as adversary
import attacks
import numpy as np

dataset = 'cifar10'
batch_size = 128
best_prec1 = 0.0
# lr = 0.1 * (batch_size / 256)
lr = 0.001
momentum = 0.9
weight_decay = 1e-4
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
arch = 'resnet34'


def get_correct_pred_batchs(x, y, model):
    correct = 0
    correct_idx = []
    with torch.no_grad():
        images, labels = x.cuda(), y.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        for i in range(len(labels)):
            if predicted[i] == labels[i]:
                correct_idx.append(i)
    return correct_idx


def get_incorrect_pred_batchs(x, y, model):
    correct = 0
    incorrect_idx = []
    with torch.no_grad():
        images, labels = x.cuda(), y.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        for i in range(len(labels)):
            if predicted[i] != labels[i]:
                incorrect_idx.append(i)
    return incorrect_idx


def train_with_adv_exs(train_loader, model, test_model, criterion, optimizer, epoch, batch_size, num_classes):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    test_model.eval()
    # fgsm = attacks.FGSM()
    bim = attacks.BIM()
    # cw = adversary.cw()
    # df = attacks.DeepFool()
    end = time.time()
    for i, (x, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        x = x.cuda(non_blocking=True)
        # Online FGSM Attack
        # fgsm_data = fgsm(test_model, x.data.clone(), target.cpu(), 0.02)
        # inc_pred_ids_fgsm = get_incorrect_pred_batchs(fgsm_data, target, test_model)
        # fgsm_images = [fgsm_data[i] for i in inc_pred_ids_fgsm]
        # fgsm_images = torch.stack(fgsm_images)  # .squeeze()
        # fgsm_labels = [target[i] for i in inc_pred_ids_fgsm]
        # fgsm_labels = torch.stack(fgsm_labels)  # .squeeze()
        # fgsm_images = fgsm_images.cuda(non_blocking=True)

        # Online BIM Attack
        bim_data = bim(test_model, x.data.clone(), target.cpu(), num_classes)
        inc_pred_ids_bim = get_incorrect_pred_batchs(bim_data, target, test_model)
        bim_images = [bim_data[i] for i in inc_pred_ids_bim]
        bim_images = torch.stack(bim_images)  # .squeeze()
        bim_labels = [target[i] for i in inc_pred_ids_bim]
        bim_labels = torch.stack(bim_labels)  # .squeeze()
        bim_images = bim_images.cuda(non_blocking=True)

        # Online DF Attack
        # df_data = df(test_model, x.data.clone(), target.cpu(), range(batch_size),
        #              num_classes=10, max_iter=10)
        # df_data = adversary.deepfool(model,images[:16].data.clone(), labels[:16].data.cpu(),
        # num_classes, train_mode=False)
        # inc_pred_ids_df = get_incorrect_pred_batchs(df_data, target, test_model)
        # df_images = [df_data[i] for i in inc_pred_ids_df]
        # df_images = torch.stack(df_images)  # .squeeze()
        # df_labels = [target[i] for i in inc_pred_ids_df]
        # df_labels = torch.stack(df_labels)  # .squeeze()
        # df_images = df_images.cuda(non_blocking=True)

        # Online CW Attack
        # cw_data = adversary.cw(test_model, x.data.clone(), target.cpu(), 1.0, 'l2')
        # inc_pred_ids_cw = get_incorrect_pred_batchs(cw_data, target, test_model)
        # cw_images = [cw_data[i] for i in inc_pred_ids_cw]
        # cw_images = torch.stack(cw_images)  # .squeeze()
        # cw_labels = [target[i] for i in inc_pred_ids_cw]
        # cw_labels = torch.stack(cw_labels)  # .squeeze()
        # cw_images = cw_images.cuda(non_blocking=True)

        images = torch.cat((x, bim_images), dim=0)
        labels = torch.cat((target, bim_labels), dim=0)
        ids = np.arange(images.size()[0])
        np.random.shuffle(ids)
        labels = labels[ids]
        images = images[ids]
        # compute output
        labels = labels.cuda(non_blocking=True)
        output = model(images)
        loss = criterion(output, labels)

        # measure accuracy and record loss
        prec1 = accuracy(output, labels)

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


def save_checkpoint(state, is_best, filename='saved_models/' + arch + '_checkpoint_cifar10_trial.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'saved_models/' + arch + '_model_cifar10_best_trial.pth.tar')


def main():
    global lr, best_prec1
    model_path = 'pre_trained/resnet_cifar10.pth'
    epochs = 10
    model = resnet.ResNet34(10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    test_model = resnet.ResNet34(10)
    test_model.load_state_dict(torch.load(model_path, map_location=device))
    # model = nn.Sequential(nn.BatchNorm2d(num_features=3, affine=False), pretrained)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 10)
    model.cuda()
    test_model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=[25, 50, 75], gamma=0.2)

    # train_loader, val_loader, class_names = datasets.get_train_test_dataset(dataset, batch_size, True)
    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()

    train_loader, val_loader, _ = get_train_test_dataset('cifar10', batch_size, True)

    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch, lr)

        # train for one epoch
        train_with_adv_exs(train_loader, model, test_model, criterion, optimizer, epoch, batch_size, 10)
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


if __name__ == "__main__":
    main()
