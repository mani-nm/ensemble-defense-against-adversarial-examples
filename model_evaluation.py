"""
# Author: Nag Mani
# Created: 2/18/2019
"""

import time
import torch


def get_pred_single(x, model):
    x = x.cuda()
    outputs = model(x)
    _, predicted = torch.max(outputs.data, 1)

    return predicted


def get_correct_pred_batchs(x, y, model, args):
    start = time.time()
    correct = 0
    total = 0
    correct_idx =[]
    print("here")
    with torch.no_grad():
        images, labels = x.cuda(args.gpu), y.cuda(args.gpu)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for i in range(len(labels)):
            if predicted[i] == labels[i]:
                correct_idx.append(i)
        print("Total vs correct = {:.2f}/{:.2f}".format(correct, total))
    acc = 100 * correct / total
    print('Accuracy of the network on the {} test images: {:.2f} %' .format(total,
            100 * correct / total))
    end = time.time()
    print("Execution time: ", end-start)
    return correct_idx


def test_accuracy(test_loader, model, args):
    start = time.time()
    correct = 0
    total = 0
    model.eval()
    print("here")
    with torch.no_grad():
        for data in test_loader:

            images, labels = data
            images, labels = images.cuda(args.gpu), labels.cuda(args.gpu)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print("Total vs correct = {:.2f}/{:.2f}".format(correct, total))
    acc = 100 * correct / total
    print('Accuracy of the network on the {} test images: {:.2f} %' .format(total,
            100 * correct / total))
    end = time.time()
    print("Execution time: ", end-start)
    return acc


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))



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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

