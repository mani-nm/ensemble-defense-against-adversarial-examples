"""
# Driver to run adversarial examples and create
# AE samples from clean data and pre-trained models
# Author: Nag Mani
# Created: 2/18/2019
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from foolbox.models import PyTorchModel
import matplotlib.pyplot as plt
import time
import os
from datasets import get_train_test_dataset
import resnet, densenet
from multiprocessing import Pool

import attacks
import lib.adversary as adversary
import lib.util as util


def imagenet(batch_size, data_root, limit=None, attack_name=None, **kwargs):
    def getclasses(data):
        classes = os.listdir(data + '/imagenet/val')
        classes.sort()
        f = open(data + "/synset_words.txt", "r")
        synset_to_name = {}
        for line in f:
            parts = line.split(" ")

            synset_to_name[parts[0]] = " ".join(parts[1:]).rstrip()

        return synset_to_name

    synset_to_name = getclasses(data_root)

    valdir = os.path.join(data_root, 'imagenet/val')
    num_workers = kwargs.setdefault('num_workers', 2)
    kwargs.pop('input_size', None)
    _normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    if attack_name == 'common_corruption':

        val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0,0,0], std=[255,255,255])
        ]))
    else:
        val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            _normalize,
        ]))

    if limit:
        np.random.seed(2019)
        indices = np.random.choice(range(20000, 25000), limit, replace=False)
        torch.manual_seed(limit)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
    else:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            # sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
    class_names = val_dataset.classes

    return val_loader, class_names, synset_to_name


def get_pred_single(x, model):
    x = x.cuda()
    outputs = model(x)
    _, predicted = torch.max(outputs.data, 1)

    return predicted


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


def test_accuracy(test_loader, model):
    start = time.time()
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print("Correct vs Total = {:.2f}/{:.2f} = {:.2f}".format(correct, total, correct / total))
    acc = 100 * correct / total
    print('Accuracy of the network on the {} test images: {:.2f} %'.format(total,
                                                                           100 * correct / total))
    end = time.time()
    print("Execution time: ", end - start)
    return acc


# def parallelize_deepfool(attack, model, images, labels, ids):
#     counter = 0
#     for id in ids:
#         pert_total, pert_image = attack(images[id], labels[id], model)
#         ae_pred = get_pred_single(pert_image, model)
#
#         if ae_pred == labels[id].item():
#             counter += 1
#     return counter


def run_deepfool_attack(test_loader, model, num_classes):
    start = time.time()
    model.eval()
    total = 0
    correct = 0
    adv_examples = []
    # perts = 0
    print_freq = 5
    # df = attacks.DeepFool()
    for i, (images, labels) in enumerate(test_loader):
        images = images.cuda()
        labels = labels.cuda()
        correctly_predicted_ids = get_correct_pred_batchs(images, labels, model)
        if len(correctly_predicted_ids)>0:
            total += len(correctly_predicted_ids)
            # adv_data = df(model, images.data.clone(), labels.cpu(), correctly_predicted_ids,
            #               num_classes=num_classes, max_iter=10)
            images = [images[i] for i in correctly_predicted_ids]
            images = torch.stack(images)  # .squeeze()
            labels = [labels[i] for i in correctly_predicted_ids]
            labels = torch.stack(labels)  # .squeeze()
            adv_data = adversary.deepfool(model,images.data.clone(), labels.data.cpu(), num_classes, train_mode=False)
            # labels = [labels[i] for i in correctly_predicted_ids]
            # labels = torch.stack(labels)# .squeeze()
            correct_ids = get_correct_pred_batchs(adv_data, labels, model)
            correct += len(correct_ids)
            adv_data = adv_data.cuda()
            adv_outputs = model(adv_data)
            _, adv_pred = torch.max(adv_outputs.data, 1)
            # print(adv_pred)
            for k in range(len(adv_pred)):
                if adv_pred[k].item() != labels[k].item():
                    clean_ex = images[k].squeeze().detach().cpu().numpy()
                    adv_examples.append((labels[k].item(), labels[k].item(), clean_ex))
                    adv_ex = adv_data[k].squeeze().detach().cpu().numpy()
                    adv_examples.append((labels[k].item(), adv_pred[k].item(), adv_ex))

            if i % print_freq == 0:
                print("Correct vs Total = {:.2f}/{:.2f} = {:.2f}".format(correct, total, correct / total))
    print('Accuracy of the network on the {} test images where {} can withstand attack: {:.2f} %'
          .format(total, correct, 100 * correct / total))
    end = time.time()
    print("Execution time: ", end - start)
    return adv_examples
    # for i, (images, labels) in enumerate(test_loader):
    #     correctly_predicted_ids = get_correct_pred_batchs(images, labels, model)
    #     # correct += p.map(parallelize_deepfool, (df, model, images, labels, correctly_predicted_ids))
    #     total += len(correctly_predicted_ids)
    #     for id in correctly_predicted_ids:
    #         # df = attacks.DeepFool_single()
    #         pert_total, pert_image = df(images[id], labels[id], model, num_classes)
    #         ae_pred = get_pred_single(pert_image, model)
    #
    #         if ae_pred == labels[id].item():
    #             correct += 1
    #         else:
    #             # adv_examples.append((labels[id], ae_pred, pert_image))
    #             perts += np.mean(pert_total)
    #     if i % print_freq == 0:
    #         print("Correct vs Total = {:.2f}/{:.2f} = {:.2f}".format(correct, total, correct / total))
    # print('Accuracy of the network on the {} test images where {} can withstand attack: {:.2f} %'.format(total, correct,
    #                                                                        100 * correct / total))
    # return adv_examples, perts


def run_one_pixel_attack(test_loader, model):
    model.eval()
    total = 0
    correct = 0
    adv_examples = []
    print_freq = 2
    df = attacks.OnePixelAttack()
    for i, (images, labels) in enumerate(test_loader):
        correctly_predicted_ids = get_correct_pred_batchs(images, labels, model)
        # correct += p.map(parallelize_deepfool, (df, model, images, labels, correctly_predicted_ids))
        total += len(correctly_predicted_ids)
        for id in correctly_predicted_ids:
            # df = attacks.DeepFool_single()
            pert_image = df(images[id], labels[id], model)
            ae_pred = get_pred_single(pert_image, model)

            if ae_pred == labels[id].item():
                correct += 1
            else:
                # adv_examples.append((labels[id], ae_pred, pert_image))
                clean_ex = images[id].squeeze().detach().cpu().numpy()
                adv_examples.append((labels[id].item(), labels[id].item(), clean_ex))
                adv_ex = pert_image.squeeze().detach().cpu().numpy()
                adv_examples.append((labels[id].item(), ae_pred.item(), adv_ex))  # np.add(adv_ex, clean_ex)
            if len(adv_examples) > 8:
                adv_examples.pop(0)
                adv_examples.pop(0)

        if i % print_freq == 0:
            print("Correct vs Total = {:.2f}/{:.2f} = {:.2f}".format(correct, total, correct / total))
        print('Out of {} test images, {} were predicted correctly, Accuracy: {:.2f} %'.format(total, correct,
                                                                                              100 * correct / total))
    return adv_examples


def _get_label_to_examples(test_loader, num_classes):
    label_to_examples={}
    for i in range(num_classes):
        label_to_examples[i] = []

    for x_np, y_np in test_loader:
        for x, label in zip(x_np, y_np):
            label_to_examples[label.item()].append(x)
    # print(label_to_examples.keys())
    return label_to_examples


def run_boundary_attack(test_loader, sample_loaders, model):
    model.eval()
    total = 0
    correct = 0
    adv_examples = []
    print_freq = 2
    shape = (224, 224, 3)
    label_to_examples = _get_label_to_examples(sample_loaders, 1000)
    # df = attacks.BoundaryAttackClass(model, shape, label_to_examples=label_to_examples)
    df = attacks.BoundaryAttackClass()
    for i, (images, labels) in enumerate(test_loader):
        correctly_predicted_ids = get_correct_pred_batchs(images, labels, model)
        # correct += p.map(parallelize_deepfool, (df, model, images, labels, correctly_predicted_ids))
        total += len(correctly_predicted_ids)
        for id in correctly_predicted_ids:
            pert_image = df(images[id], labels[id], model)
            ae_pred = get_pred_single(pert_image, model)

            if ae_pred == labels[id].item():
                correct += 1
            # else:
                # adv_examples.append((labels[id], ae_pred, pert_image))
                # clean_ex = images[id].squeeze().detach().cpu().numpy()
                # adv_examples.append((labels[id].item(), labels[id].item(), clean_ex))
                # adv_ex = pert_image.squeeze().detach().cpu().numpy()
                # adv_examples.append((labels[id].item(), ae_pred.item(), adv_ex))
        if i % print_freq == 0:
            print("Correct vs Total = {:.2f}/{:.2f} = {:.2f}".format(correct, total, correct / total))
    print('Accuracy of the network on the {} test images: {:.2f} %'.format(total,
                                                                           100 * correct / total))
    return adv_examples


def run_common_corruption_attack(test_loader, model):
    start = time.time()
    total = 0
    correct = 0
    adv_examples = []
    print_freq = 10
    j = 0
    df = attacks.CommonCorruptionsAttack()
    for i, (images, labels) in enumerate(test_loader):

        correctly_predicted_ids = get_correct_pred_batchs(images, labels, model)
        if len(correctly_predicted_ids)>0:
            total += len(correctly_predicted_ids)
            images_ss = np.array([images[j].data.numpy() for j in correctly_predicted_ids])
            labels_ss = [labels[j] for j in correctly_predicted_ids]
            # print(type(images_ss))
            # print(type(images_ss[0]))
            images_ss = torch.from_numpy(images_ss).permute(0, 2, 3, 1)
            pert_images = df(model, images_ss, labels_ss)
            for img, lbl, id in zip(pert_images, labels_ss, correctly_predicted_ids):
                # pert_image = df(pert_images[id], labels[id], model)
                # print(img.size())
                # print(type(pert_images[id]))
                ae_pred = get_pred_single(img.unsqueeze(0), model)
                # print(ae_pred.item(), lbl.item())
                if ae_pred.item() == lbl.item():
                    correct += 1
                else:
                    # adv_examples.append((labels[id], ae_pred, pert_image))
                    clean_ex = images[id].squeeze().detach().cpu().numpy()
                    adv_examples.append((labels[id].item(), labels[id].item(), clean_ex))
                    adv_ex = img.squeeze().detach().cpu().numpy()
                    adv_examples.append((labels[id].item(), ae_pred.item(), adv_ex)) #np.add(adv_ex, clean_ex)
                if len(adv_examples)>4:
                    adv_examples.pop(0)
                    adv_examples.pop(0)

            if i % print_freq == 0:
                print("Correct vs Total = {:.2f}/{:.2f} = {:.2f}".format(correct, total, correct / total))
    print('Out of {} test images, {} were predicted correctly, Accuracy: {:.2f} %'.format(total, correct,
                                                                           100 * correct / total))
    return adv_examples


def run_cw_attack(test_loader, model):
    start = time.time()
    total = 0
    correct = 0
    adv_examples = []
    print_freq = 10
    j = 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.cuda()
        labels = labels.cuda()
        correctly_predicted_ids = get_correct_pred_batchs(images, labels, model)
        if len(correctly_predicted_ids)>0:
            total += len(correctly_predicted_ids)
            images = [images[i] for i in correctly_predicted_ids]
            images = torch.stack(images)# .squeeze()
            labels = [labels[i] for i in correctly_predicted_ids]
            labels = torch.stack(labels)# .squeeze()
            # print(images.size())
            adv_data = adversary.cw(model, images.data.clone(), labels.cpu(), 1.0, 'linf') # linf
            correct_ids = get_correct_pred_batchs(adv_data, labels, model)
            correct += len(correct_ids)

            # adv_outputs = model(adv_data)
            # _, adv_pred = torch.max(adv_outputs.data, 1)
            # # print(adv_pred)
            # for k in range(len(adv_pred)):
            #     if adv_pred[k].item() != labels[k].item():
            #         clean_ex = images[k].squeeze().detach().cpu().numpy()
            #         adv_examples.append((labels[k].item(), labels[k].item(), clean_ex))
            #         adv_ex = adv_data[k].squeeze().detach().cpu().numpy()
            #         adv_examples.append((labels[k].item(), adv_pred[k].item(), adv_ex))

            if i % print_freq == 0:
                print("Correct vs Total = {:.2f}/{:.2f} = {:.2f}".format(correct, total, correct / total))
    print('Accuracy of the network on the {} test images where {} can withstand attack: {:.2f} %'
          .format(total, correct, 100 * correct / total))
    end = time.time()
    print("Execution time: ", end - start)
    return adv_examples


def run_fgsm_attack(test_loader, model):
    start = time.time()
    total = 0
    correct = 0
    adv_examples = []
    print_freq = 10
    df = attacks.FGSM()
    for i, (images, labels) in enumerate(test_loader):
        images = images.cuda()
        labels = labels.cuda()
        correctly_predicted_ids = get_correct_pred_batchs(images, labels, model)
        if len(correctly_predicted_ids)>0:
            total += len(correctly_predicted_ids)
            images = [images[i] for i in correctly_predicted_ids]
            images = torch.stack(images)# .squeeze()
            labels = [labels[i] for i in correctly_predicted_ids]
            labels = torch.stack(labels)# .squeeze()
            # print(images.size())
            adv_data = df(model, images.data.clone(), labels.cpu(), 0.02, type='ml') # linf
            correct_ids = get_correct_pred_batchs(adv_data, labels, model)
            correct += len(correct_ids)

            adv_outputs = model(adv_data)
            _, adv_pred = torch.max(adv_outputs.data, 1)
            # print(adv_pred)
            for k in range(len(adv_pred)):
                if adv_pred[k].item() != labels[k].item():
                    clean_ex = images[k].squeeze().detach().cpu().numpy()
                    adv_examples.append((labels[k].item(), labels[k].item(), clean_ex))
                    adv_ex = adv_data[k].squeeze().detach().cpu().numpy()
                    adv_examples.append((labels[k].item(), adv_pred[k].item(), adv_ex))

            if i % print_freq == 0:
                print("Correct vs Total = {:.2f}/{:.2f} = {:.2f}".format(correct, total, correct / total))
    print('Accuracy of the network on the {} test images where {} can withstand attack: {:.2f} %'
          .format(total, correct, 100 * correct / total))
    end = time.time()
    print("Execution time: ", end - start)
    return adv_examples


def run_bim_attack(test_loader, model, num_classes):
    start = time.time()
    total = 0
    correct = 0
    adv_examples = []
    print_freq = 10
    df = attacks.BIM()
    for i, (images, labels) in enumerate(test_loader):
        images = images.cuda()
        labels = labels.cuda()
        correctly_predicted_ids = get_correct_pred_batchs(images, labels, model)
        if len(correctly_predicted_ids)>0:
            total += len(correctly_predicted_ids)
            images = [images[i] for i in correctly_predicted_ids]
            images = torch.stack(images)# .squeeze()
            labels = [labels[i] for i in correctly_predicted_ids]
            labels = torch.stack(labels)# .squeeze()
            # print(images.size())
            adv_data = df(model, images.data.clone(), labels.cpu(), num_classes)
            correct_ids = get_correct_pred_batchs(adv_data, labels, model)
            correct += len(correct_ids)

            adv_outputs = model(adv_data)
            _, adv_pred = torch.max(adv_outputs.data, 1)
            # print(adv_pred)
            for k in range(len(adv_pred)):
                if adv_pred[k].item() != labels[k].item():
                    clean_ex = images[k].squeeze().detach().cpu().numpy()
                    adv_examples.append((labels[k].item(), labels[k].item(), clean_ex))
                    adv_ex = adv_data[k].squeeze().detach().cpu().numpy()
                    adv_examples.append((labels[k].item(), adv_pred[k].item(), adv_ex))

            if i % print_freq == 0:
                print("Correct vs Total = {:.2f}/{:.2f} = {:.2f}".format(correct, total, correct / total))
    print('Accuracy of the network on the {} test images where {} can withstand attack: {:.2f} %'
          .format(total, correct, 100 * correct / total))
    end = time.time()
    print("Execution time: ", end - start)
    return adv_examples


def run_illc_attack(test_loader, model, num_classes):
    start = time.time()
    total = 0
    correct = 0
    adv_examples = []
    print_freq = 10
    df = attacks.ILLC()
    for i, (images, labels) in enumerate(test_loader):
        images = images.cuda()
        labels = labels.cuda()
        correctly_predicted_ids = get_correct_pred_batchs(images, labels, model)
        if len(correctly_predicted_ids)>0:
            total += len(correctly_predicted_ids)
            images = [images[i] for i in correctly_predicted_ids]
            images = torch.stack(images)# .squeeze()
            labels = [labels[i] for i in correctly_predicted_ids]
            labels = torch.stack(labels)# .squeeze()
            # print(images.size())
            adv_data = df(model, images.data.clone(), labels.cpu(), num_classes)
            correct_ids = get_correct_pred_batchs(adv_data, labels, model)
            correct += len(correct_ids)

            adv_outputs = model(adv_data)
            _, adv_pred = torch.max(adv_outputs.data, 1)
            # print(adv_pred)
            for k in range(len(adv_pred)):
                if adv_pred[k].item() != labels[k].item():
                    clean_ex = images[k].squeeze().detach().cpu().numpy()
                    adv_examples.append((labels[k].item(), labels[k].item(), clean_ex))
                    adv_ex = adv_data[k].squeeze().detach().cpu().numpy()
                    adv_examples.append((labels[k].item(), adv_pred[k].item(), adv_ex))

            if i % print_freq == 0:
                print("Correct vs Total = {:.2f}/{:.2f} = {:.2f}".format(correct, total, correct / total))
    print('Accuracy of the network on the {} test images where {} can withstand attack: {:.2f} %'
          .format(total, correct, 100 * correct / total))
    end = time.time()
    print("Execution time: ", end - start)
    return adv_examples


def main():
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    epsilons = [0.01, .02, 0.03]
    iterations = [3, 4, 5, 6]
    data_dir = 'D:\\Research\\AE\\Adversarial_Examples\\data'

    # Get dataset
    # dataloaders, class_names, dirnm_to_label = imagenet(batch_size=32, data_root=data_dir, limit=32,
    #                                                     attack_name='common_corruption',
    #                                                     num_workers=2)  # , limit = 1000

    dataloaders, classes = get_train_test_dataset('cifar10', 128, False)
    print("success")
    # load a pretrained model
    # model_res50 = models.resnet50(pretrained=True)
    # model_res50.state_dict(torch.load('checkpoint.pth.tar')['state_dict'])
    # model_res50 = nn.Sequential(nn.BatchNorm2d(num_features=3, affine=False), model_res50)
    # num_ftrs = model_res50.fc.in_features
    # model_res50.fc = nn.Linear(num_ftrs, 10)
    # model_res50 = model_res50.to(device)

    ### Adversarial retraining
    # model_res34 = resnet.ResNet34(10)
    # chkpoint = torch.load('saved_models/resnet34checkpoint_adv_fgsm_ret_315.pth.tar')
    # model_res34.load_state_dict(chkpoint['state_dict'])
    # model_res34.cuda()
    # test_accuracy(dataloaders, model_res34)

    # dense_path = os.path.join('pre_trained', 'densenet_cifar.pth')
    # dense_model = densenet.DenseNet121()
    # dense_model = torch.nn.DataParallel(dense_model)
    # checkpoint = torch.load(dense_path)
    # dense_model.load_state_dict(checkpoint['net'])
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in checkpoint.items():
    #     print(k)
    #     name = k[7:]  # remove `module.`
    #     new_state_dict[name] = v
    # # load params
    # dense_model.load_state_dict(new_state_dict)
    # den_model = torch.load(dense_path, map_location=device)
    # den_model = torch.load(dense_path, map_location=device)
    # dense_model.cuda()

    resnet_path = os.path.join('pre_trained', 'resnet_cifar10.pth')
    res_model = resnet.ResNet34(num_c=10)
    res_model.load_state_dict(torch.load(resnet_path, map_location=device))
    res_model.cuda()
    # test_accuracy(dataloaders, res_model)
    # Evaluate the base model : Done for ResNet50
    # base_acc = test_accuracy(dataloaders, model_res50)
    # print(base_acc)

    # Running FGSM attack
    # print("Running FGSM:\n")
    # advs = run_fgsm_attack(dataloaders, res_model)
    # util.display_images(advs, class_names=classes)

    # Running BIM attack
    # print("Running BIM:\n")
    # advs = run_bim_attack(dataloaders, res_model, 10)
    # util.display_images(advs, class_names=classes)

    # Running ILLC attack
    # print("Running ILLC:\n")
    # advs = run_illc_attack(dataloaders, res_model, 10)
    # util.display_images(advs, class_names=classes)

    # Running DeepFool : Done
    # print("Running DeepFool:\n")
    # advs = run_deepfool_attack(dataloaders, res_model, 10)
    # util.display_images(advs, class_names=classes)
    # advs, perts = run_deepfool_attack(dataloaders, res_model, 10)
    # print("Average Perturbation: {:.02f}".format(perts))
    # for t, p, _ in advs:
    #     print("{} predicted as {}".format(dirnm_to_label[class_names[t]], dirnm_to_label[class_names[p.item()]]))

    # Running CWL2 attack
    print("Running CW:\n")
    advs = run_cw_attack(dataloaders, res_model)
    # util.display_images(advs, dirnm_to_label, class_names)




    # mn, mx = 0, 0
    # i= 0
    # for img, label in dataloaders:
    #     print(i)
    #     np_imgs = img.numpy()
    #     per_image_min = np.min(np_imgs)
    #     per_image_max = np.max(np_imgs)
    #     mn += per_image_min
    #     mx = per_image_max
    #     i += 1
    #     break
    #
    # print("Overall min: ", mn)
    # print("Overall max: ", mx)

    # Running Boundary Attack
    # testloaders, _, _ = imagenet(batch_size=32, data_root=data_dir, limit=128,
    #                                                     num_workers=4)  # , limit = 1000
    #
    # avdx = run_boundary_attack(dataloaders, testloaders, model_res50)
    # util.display_images(avdx, dirnm_to_label, class_names)

    # Running common corruptions

    # adx = run_common_corruption_attack(dataloaders, res_model)
    # util.display_images(adx, dirnm_to_label, class_names)


    # Running one_pixel attack
    # adv = run_one_pixel_attack(dataloaders, res_model)
    # util.display_images(adv, dirnm_to_label, class_names)

    # Run LBFGS Attack
    # adv = run_lbfgs_attack(dataloaders, model_res50)
    # util.display_images(adv, dirnm_to_label, class_names)


if __name__ == '__main__':
    main()
