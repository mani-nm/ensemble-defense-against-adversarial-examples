# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import matplotlib.pyplot as plt
from PIL import Image

import os
import tempfile
import numpy as np
import shutil
import sys
import matplotlib
# matplotlib.use('Agg')

import torch

# constants:
CHECKPOINT_FILE = 'checkpoint.torch'


# function that measures top-k accuracy:
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100. / batch_size))
    return res


# function that tries to load a checkpoint:
def load_checkpoint(checkpoint_folder):
    # read what the latest model file is:
    filename = os.path.join(checkpoint_folder, CHECKPOINT_FILE)
    if not os.path.exists(filename):
        return None

    # load and return the checkpoint:
    return torch.load(filename)


# function that saves checkpoint:
def save_checkpoint(checkpoint_folder, state):
    # make sure that we have a checkpoint folder:
    if not os.path.isdir(checkpoint_folder):
        try:
            os.makedirs(checkpoint_folder)
        except BaseException:
            print('| WARNING: could not create directory %s' % checkpoint_folder)
    if not os.path.isdir(checkpoint_folder):
        return False

    # write checkpoint atomically:
    try:
        with tempfile.NamedTemporaryFile(
                'w', dir=checkpoint_folder, delete=False) as fwrite:
            tmp_filename = fwrite.name
            torch.save(state, fwrite.name)
        os.rename(tmp_filename, os.path.join(checkpoint_folder, CHECKPOINT_FILE))
        return True
    except BaseException:
        print('| WARNING: could not write checkpoint to %s.' % checkpoint_folder)
        return False


# function that adjusts the learning rate:
def adjust_learning_rate(base_lr, epoch, optimizer, lr_decay, lr_decay_stepsize):
    lr = base_lr * (lr_decay ** (epoch // lr_decay_stepsize))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# adversary functions
# computes SSIM for a single block
def SSIM(x, y):
    x = x.resize_(x.size(0), x.size(1) * x.size(2) * x.size(3))
    y = y.resize_(y.size(0), y.size(1) * y.size(2) * y.size(3))
    N = x.size(1)
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    sigma_x = x.std(1)
    sigma_y = y.std(1)
    sigma_xy = ((x - mu_x.expand_as(x)) * (y - mu_y.expand_as(y))).sum(1) / (N - 1)
    ssim = (2 * mu_x * mu_y) * (2 * sigma_xy)
    ssim = ssim / (mu_x.pow(2) + mu_y.pow(2))
    ssim = ssim / (sigma_x.pow(2) + sigma_y.pow(2))
    return ssim


# mean SSIM using local block averaging
def MSSIM(x, y, window_size=16, stride=4):
    ssim = torch.zeros(x.size(0))
    L = x.size(2)
    W = x.size(3)
    x_inds = torch.arange(0, L - window_size + 1, stride).long()
    y_inds = torch.arange(0, W - window_size + 1, stride).long()
    for i in x_inds:
        for j in y_inds:
            x_sub = x[:, :, i:(i + window_size), j:(j + window_size)]
            y_sub = y[:, :, i:(i + window_size), j:(j + window_size)]
            ssim = ssim + SSIM(x_sub, y_sub)
    return ssim / x_inds.size(0) / y_inds.size(0)


# forwards input through model to get probabilities
def get_probs(model, imgs, output_prob=False):
    softmax = torch.nn.Softmax(1)
    # probs = torch.zeros(imgs.size(0), n_classes)
    imgsvar = torch.autograd.Variable(imgs, volatile=True) #.squeeze()
    output = model(imgsvar.cuda())
    if output_prob:
        probs = output.data.cpu()
    else:
        probs = softmax.forward(output).data.cpu()

    return probs


# calls get_probs to get predictions
def get_labels(model, input, output_prob=False):
    probs = get_probs(model, input, output_prob)
    _, label = probs.max(1)
    return label.squeeze()


def display_images(res_ex_df, dirnm_to_label='', class_names=''):
    cnt = 0
    plt.figure(figsize=(15, 20))
    # print(len(res_ex_df[0][0]))
    for i in range(4):
        for j, (orig, adv, ex) in enumerate(res_ex_df):
            cnt += 1
            if cnt > (2 * 4):
                break
            plt.subplot(4, 2, cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            # if j == 0:
            #     plt.ylabel("Itr: {}".format(iters[i]), fontsize=14)
            ex = ex.transpose((1, 2, 0))
            mean = np.array([0.4914, 0.4822, 0.4465]) #[0.485, 0.456, 0.406]
            std = np.array([0.2023, 0.1994, 0.2010]) #0.229, 0.224, 0.225
            ex = std * ex + mean
            if ex.shape[-1] == 3:
                img = Image.fromarray(np.uint8(ex * 255.), 'RGB')
            else:
                img = Image.fromarray(np.uint8(ex[:, :, 0] * 255.), 'L')

            ex = np.clip(ex, 0, 1)
            matplotlib.rcParams.update({'font.size': 22})
            if adv == orig:
                # plt.title("{}".format(dirnm_to_label[class_names[orig]]))
                plt.title("{}".format(class_names[orig]))
            else:
                # plt.title("{}".format(dirnm_to_label[class_names[adv]]))
                plt.title("{}".format(class_names[adv]))
            #         ex = invTrans(ex).numpy()
            #         ex =np.swapaxes(ex,0,2)
            #         ex = np.swapaxes(ex,0,1)
            #         print(ex.shape)
            plt.imshow(ex)

    #         plt.imshow(np.transpose(invTrans(ex))
    plt.tight_layout()
    plt.show()


def save_correct_and_incorrect_adv_images(x_adv, correct, image_ids, labels, results_dir):
    correct_dir = os.path.join(results_dir, 'correct_images')
    shutil.rmtree(correct_dir, ignore_errors=True)

    incorrect_dir = os.path.join(results_dir, 'incorrect_images')
    shutil.rmtree(incorrect_dir, ignore_errors=True)

    for i, image_np in enumerate(x_adv):
        if correct[i]:
            save_dir = correct_dir
        else:
            save_dir = incorrect_dir

        filename = "adv_%s_%s.png" % (labels[i], image_ids[i])
        save_image_to_png(image_np, os.path.join(save_dir, filename))


def save_image_to_png(image_np, filename):
    from PIL import Image

    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    if image_np.shape[-1] == 3:
        img = Image.fromarray(np.uint8(image_np * 255.), 'RGB')
    else:
        img = Image.fromarray(np.uint8(image_np[:, :, 0] * 255.), 'L')
    img.save(filename)


def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
