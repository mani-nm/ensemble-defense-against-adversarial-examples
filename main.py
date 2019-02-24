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
import torch.nn.functional as F
from torch.autograd import Variable

import datasets
import model_evaluation
import attacks
import multiprocessing

parser = argparse.ArgumentParser(description="Pytorch code: Adversarial Examples Generation")
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size')
parser.add_argument('--dataset', default='imagenet')
parser.add_argument('--data_dir', default='data', help='Root data directory (relative path)')
parser.add_argument('--arch', default='resnet152')
parser.add_argument('--gpu', type=int, default='0')
parser.add_argument('--adv', default='deepfool',  help='deepfool, bim, fgsm, cw')
parser.add_argument('--print_freq', type=int, default=1, help='Print frequency')

args = parser.parse_args()


def run_attack(net, data_loader, ):
    net.eval()
    total = 0
    correct = 0
    adv_examples = []
    for images, labels in data_loader:
        correctly_predicted_ids = model_evaluation.get_correct_pred_batchs(images, labels, net, args)

        for id in correctly_predicted_ids:
            df = attacks.DeepFool_single()
            pert_total, pert_image = df(images[id], labels[id], net, args)
            ae_pred = model_evaluation.get_pred_single(pert_image, net)
            total += 1
            if ae_pred == labels[id].item():
                correct += 1
            else:
                adv_examples.append((labels[id], ae_pred, pert_image))

    print("")



def main():

    # Initialize default variables
    N_WORKERS = multiprocessing.cpu_count() or 1
    pool = multiprocessing.Pool(N_WORKERS)
    pretrained = True
    epochs = 30
    momentum = 0.9
    wd = 1e-4
    out_dir = 'ae_images/' + args.dataset + '/' + args.arch + '/'
    path = os.path.join(os.getcwd(), out_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(path)
    data = os.path.join(os.getcwd(), args.data_dir)
    torch.cuda.manual_seed(2019)

    # Loading the dataset
    test_loader, labels_dict = datasets.get_train_test_dataset(args.dataset, args.batch_size, data)


    # Loading pretrained model
    if pretrained:
        model = getattr(models, args.arch)(pretrained=pretrained)
    else:
        model = getattr(models, args.arch)(pretrained=False)
    model.cuda()

    # preTrained_path = os.path.join(os.getcwd(), 'saved_models')
    # model_name = '.pth'
    # model_path = os.path.join(preTrained_path, model_name)

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), 0.1,
                                momentum=momentum,
                                weight_decay=wd)

    # print("Evaluating base model")
    torch.cuda.set_device(args.gpu)
    # model_evaluation.validate(test_loader, model, criterion, args)
    # model_evaluation.test_accuracy(test_loader, model, args)

    # If adversarial images has to be retrieved and stored then run the individual attack
    # Else to evaluate the attack performance run the model in batched mode

    model.eval()
    counter = 0
    correct = 0
    adv_examples = []
    for images, labels in test_loader:
        counter += 1
        correctly_predicted_ids = model_evaluation.get_correct_pred_batchs(images, labels, model, args)

        for id in correctly_predicted_ids:
            df = attacks.DeepFool_single()
            pert_total, pert_image = df(images[id], labels[id], model, args)
            ae_pred = model_evaluation.get_pred_single(pert_image, model)

            if ae_pred == labels[id].item():
                correct += 1
            else:
                adv_examples.append((labels[id], ae_pred, pert_image))

        if counter > 1:
            break
    print(len(adv_examples))
        # min_p, label_orig, label_pert, perturbed_data = deepfool(data, model, num_classes=2, max_iter=it)
        # avg_p += np.mean(min_p)








if __name__=='__main__':
    main()

