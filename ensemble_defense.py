import torch
import time
import shutil
import torch.nn as nn
import resnet
from datasets import get_train_test_dataset
import attacks
import lib.adversary as adversary
import numpy as np

true_label = []

model1_prob = []
model2_prob = []
model3_prob = []
model4_prob = []
total_prob = []


def ensemble_analysis(y, prob1, prob2, prob3, prob4, tot):
    # print("Outside :", len(y), len(prob1), len(prob2), len(prob3),len(prob4),len(tot))
    for i in range(len(y)):
        # print("Inside: ", len(y[i]), len(prob1[i]), len(prob2[i]), len(prob3[i]), len(prob4[i]), len(tot[i]))
        # if y[i] != np.argmax(prob1[i]) or y[i] != np.argmax(prob2[i]) or y[i] != np.argmax(prob3[i]) or \
        #         y[i] != np.argmax(prob4[i]):
        true_label.append(y[i])
        model1_prob.append(prob1[i])
        model2_prob.append(prob2[i])
        model3_prob.append(prob3[i])
        model4_prob.append(prob4[i])
        total_prob.append(tot[i])


def get_ensemble_pred_batches(x, y, model1, model2, model3, model4): #, model3, model4, model5, model6
    correct = 0
    correct_idx = []
    with torch.no_grad():
        images, labels = x.cuda(), y.cuda()
        # o_base = base_model(images)
        pred_prob1 = model1(images)
        pred_prob2 = model2(images)
        pred_prob3 = model3(images)
        pred_prob4 = model4(images)
        # pred_prob5 = model5(images)
        # pred_prob6 = model6(images)
        # true_label.append(labels.cpu().numpy())
        # model1_prob.append(pred_prob1.cpu().numpy())
        # model2_prob.append(pred_prob2.cpu().numpy())
        # model3_prob.append(pred_prob3.cpu().numpy())
        # model4_prob.append(pred_prob4.cpu().numpy())

        final_pred = pred_prob1 + pred_prob2 + pred_prob3 + pred_prob4
        # total_prob.append(final_pred.cpu().numpy())
        # + pred_prob3 + pred_prob4 + pred_prob5 + pred_prob6
        # ensemble_analysis(labels.cpu().numpy(), pred_prob1.cpu().numpy(), pred_prob2.cpu().numpy()
        #                   , pred_prob3.cpu().numpy(), pred_prob4.cpu().numpy(), final_pred.cpu().numpy())
        predicted = torch.max(final_pred.data, 1)[1]
        correct += (predicted == labels).sum().item()
        for i in range(len(labels)):
            if predicted[i] == labels[i]:
                correct_idx.append(i)
    return correct_idx


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


def run_ens_fgsm_attack(test_loader, model, model1, model2, model3, model4): # , model3, model4, model5, model6
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
            adv_data = df(model, images.data.clone(), labels.cpu(), 0.02)
            correct_ids = get_ensemble_pred_batches(adv_data, labels, model1, model2, model3, model4)
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


def run_ens_bim_attack(test_loader, model, model1, model2, model3, model4 # , model3, model4, model5, model6
                       , num_classes):
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
            correct_ids = get_ensemble_pred_batches(adv_data, labels, model1, model2, model3, model4)
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


def run_ens_illc_attack(test_loader, model, model1, model2, model3, model4 # , model3, model4, model5, model6
                        , num_classes):
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
            correct_ids = get_ensemble_pred_batches(adv_data, labels, model1, model2, model3, model4)
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


def run_ens_deepfool_attack(test_loader, model, model1, model2, model3, model4 # , model3, model4, model5, model6
                            , num_classes):
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
            correct_ids = get_ensemble_pred_batches(adv_data, labels, model1, model2, model3, model4)
            correct += len(correct_ids)
            # adv_data = adv_data.cuda()
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


def run_ens_cw_attack1(test_loader, model, model1, model2, model3, model4 # , model3 model4, model5, model6
                      , num_classes, loss_str='l2'):
    start = time.time()
    total = 0
    correct = 0
    adv_examples = []
    print_freq = 10
    j = 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.cuda()
        labels = labels.cuda()
        # correctly_predicted_ids = get_correct_pred_batchs(images, labels, model)
        # if len(correctly_predicted_ids)>0:
        #     total += len(correctly_predicted_ids)
        #     images = [images[i] for i in correctly_predicted_ids]
        #     images = torch.stack(images)# .squeeze()
        #     labels = [labels[i] for i in correctly_predicted_ids]
        #     labels = torch.stack(labels)# .squeeze()
            # print(images.size())
        total += labels.size()[0]
        adv_data = adversary.cw(model, images.data.clone(), labels.cpu(), 1.0, loss_str, num_classes) # linf
        correct_ids = get_ensemble_pred_batches(adv_data, labels, model1, model2, model3, model4)
        correct += len(correct_ids)
        if i % print_freq == 0:
            print("Correct vs Total = {:.2f}/{:.2f} = {:.2f}".format(correct, total, correct / total))
    print('Accuracy of the network on the {} test images where {} can withstand attack: {:.2f} %'
          .format(total, correct, 100 * correct / total))
    end = time.time()
    print("Execution time: ", end - start)
    return adv_examples


def run_ens_cw_attack(test_loader, model, model1, model2, model3, model4 # , model3 model4, model5, model6
                      , num_classes, loss_str='l2'):
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
            adv_data = adversary.cw(model, images.data.clone(), labels.cpu(), 1.0, loss_str, num_classes) # linf
            correct_ids = get_ensemble_pred_batches(adv_data, labels, model1, model2, model3, model4)
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


def run_ensemble_test(base_model, model1, model2, model3, model4, test_loader): # , model3, model4, model5, model6
    start = time.time()
    correct = 0
    total = 0
    base_model.eval()
    model1.eval()
    model2.eval()
    # model3.eval()
    # model4.eval()
    # model5.eval()
    # model6.eval()

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()

            # o_base = base_model(images)
            # # _, p_base = torch.max(o_base.data, 1)
            pred_prob1 = model1(images)
            pred_prob2 = model2(images)
            pred_prob3 = model3(images)
            pred_prob4 = model4(images)
            # pred_prob5 = model5(images)
            # pred_prob6 = model6(images)
            final_pred = pred_prob1 + pred_prob2 + pred_prob3 + pred_prob4   # + pred_prob3 + pred_prob4 + pred_prob5 + pred_prob6
            predicted = torch.max(final_pred.data, 1)[1]
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print("Correct vs Total = {:.2f}/{:.2f} = {:.2f}".format(correct, total, correct / total))
    acc = 100 * correct / total
    print('Accuracy of the network on the {} test images: {:.2f} %'.format(total,
                                                                           100 * correct / total))
    end = time.time()
    print("Execution time: ", end - start)
    return acc


def main():

    dataloaders, classes = get_train_test_dataset('cifar10', 64, False)
    print("Data Loaded successfully")

    # Loading Base model
    base_res = resnet.ResNet34(10)
    base_res.load_state_dict(torch.load('pre_trained/resnet_cifar10.pth'))
    base_res.cuda()

    # Loading FGSM trained model
    # fgsm_res = resnet.ResNet34(10)
    # chkpoint = torch.load('saved_models/resnet34_checkpoint_cifar10_fgsm_317.pth.tar')
    # fgsm_res.load_state_dict(chkpoint['state_dict'])
    # fgsm_res.cuda()

    # Loading ILLC trained model
    # illc_res = resnet.ResNet34(10)
    # chkpoint = torch.load('saved_models/resnet34_checkpoint_cifar10_illc_406.pth.tar')
    # illc_res.load_state_dict(chkpoint['state_dict'])
    # illc_res.cuda()

    # Loading BIM trained model
    bim_res = resnet.ResNet34(10)
    chkpoint = torch.load('saved_models/resnet34_checkpoint_cifar10_trial.pth.tar')
    bim_res.load_state_dict(chkpoint['state_dict'])
    bim_res.cuda()

    # Loading DeepFool trained model
    df_res = resnet.ResNet34(10)
    chkpoint = torch.load('saved_models/resnet34_checkpoint_cifar10_df.pth.tar')
    df_res.load_state_dict(chkpoint['state_dict'])
    df_res.cuda()

    # Loading CWL2 trained model
    cw_res = resnet.ResNet34(10)
    chkpoint = torch.load('saved_models/resnet34_checkpoint_cifar10_cw_317.pth.tar')
    cw_res.load_state_dict(chkpoint['state_dict'])
    cw_res.cuda()

    # Loading CWLinf trained model
    cwinf_res = resnet.ResNet34(10)
    chkpoint = torch.load('saved_models/resnet34_checkpoint_cifar10_cwinf_406.pth.tar')
    cwinf_res.load_state_dict(chkpoint['state_dict'])
    cwinf_res.cuda()

    a = run_ensemble_test(base_res, cwinf_res,  bim_res, df_res, cw_res, dataloaders)
    # print()
    # print("Evaluating FGSM Attack")
    # a = run_ens_fgsm_attack(dataloaders, base_res, cwinf_res,  bim_res, df_res, cw_res)
    # # illc_res, df_res, cw_res, cwinf_res
    # print()
    # print("Evaluating BIM Attack")
    # a = run_ens_bim_attack(dataloaders, base_res, cwinf_res,  bim_res, df_res, cw_res, 10)
    # print()
    # print("Evaluating ILLC Attack")
    # a = run_ens_illc_attack(dataloaders, base_res, cwinf_res,  bim_res, df_res, cw_res, 10)
    # print()
    # print("Evaluating CWL2 Attack")
    # a = run_ens_cw_attack1(dataloaders, base_res, cwinf_res,  bim_res, df_res, cw_res, 10)
    # print()
    # print("Evaluating CWLinf Attack")
    # a = run_ens_cw_attack(dataloaders, base_res, cwinf_res,  bim_res, df_res, cw_res, 10, 'linf')
    # print()
    # print("Evaluating DeepFool Attack")
    # a = run_ens_deepfool_attack(dataloaders, base_res, cwinf_res,  bim_res, df_res, cw_res, 10)
    # print()
    # print(true_label[10][0])
    # print()
    # print(model1_prob[10][0])
    # print()
    # print(model2_prob[10][0])
    # print()
    # print(model3_prob[10][0])
    # print()
    # print(model4_prob[10][0])
    # print()
    # print(total_prob[10][0])
    true_label_np = np.array(true_label)
    model1_prob_np = np.array(model1_prob)
    model2_prob_np = np.array(model2_prob)
    model3_prob_np = np.array(model3_prob)
    model4_prob_np = np.array(model4_prob)
    total_prob_np = np.array(total_prob)
    # print(true_label_np.shape, model1_prob_np.shape, model2_prob_np.shape, model3_prob_np.shape, model4_prob_np.shape)
    np.savetxt('data/true_label.csv',true_label_np, delimiter=',')
    np.savetxt('data/model1_prob.csv', model1_prob_np, delimiter=',')
    np.savetxt('data/model2_prob.csv', model2_prob_np, delimiter=',')
    np.savetxt('data/model3_prob.csv', model3_prob_np, delimiter=',')
    np.savetxt('data/model4_prob_np.csv', model4_prob_np, delimiter=',')
    np.savetxt('data/total_prob_np.csv', total_prob_np, delimiter=',')


if __name__ == "__main__":
    main()
