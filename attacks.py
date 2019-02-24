import copy
import numpy as np
import torch
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients


class Attack(object):
    name = None

    # TODO: Refactor this out of this object
    _stop_after_n_datapoints = None  # An attack can optionally run on only a subset of the dataset

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class CleanData(Attack):
    # Also known as the "null attack". Just returns the unaltered clean image
    name = 'clean'

    def __call__(self, model_fn, images_batch_nhwc, y_np):
        del y_np, model_fn  # unused
        return images_batch_nhwc


class DeepFool_single(Attack):

    name = 'DeepFool'

    def __call__(self, image, label, net, args, num_classes=10, overshoot=0.02, max_iter=10):

        image = image.cuda(args.gpu)
        f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
        I = (np.array(f_image)).flatten().argsort()[::-1]
        I = I[0:num_classes]
        label = I[0]
        input_shape = image.detach().cpu().numpy().shape
        pert_image = copy.deepcopy(image)
        w = np.zeros(input_shape)
        r_tot = np.zeros(input_shape)
        loop_i = 0

        x = Variable(pert_image[None, :], requires_grad=True)

        fs = net.forward(x)
        k_i = label
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        while k_i == label and loop_i < max_iter:

            pert = np.inf
            fs[0, I[0]].backward(retain_graph=True)
            grad_orig = x.grad.data.cpu().numpy().copy()

            for k in range(1, num_classes):
                zero_gradients(x)

                fs[0, I[k]].backward(retain_graph=True)
                cur_grad = x.grad.data.cpu().numpy().copy()

                # set new w_k and new f_k
                w_k = cur_grad - grad_orig
                f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

                pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())

                # determine which w_k to use
                if pert_k < pert:
                    pert = pert_k
                    w = w_k

            r_i = (pert + 1e-4) * w / np.linalg.norm(w)
            r_tot = np.float32(r_tot + r_i)
            pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot).cuda()

            x = Variable(pert_image, requires_grad=True)
            fs = net.forward(x)
            k_i = np.argmax(fs.data.cpu().numpy().flatten())

            loop_i += 1

        return r_tot, pert_image


class DeepFool(Attack):

    name = 'DeepFool'

    def __call__(self, images, labels, net, args, num_classes=10, overshoot=0.02, max_iter=10):
        batch_size = len(images)
        pert_imgs = [None] * batch_size

        for idx, image in enumerate(images):
            f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
            I = (np.array(f_image)).flatten().argsort()[::-1]
            I = I[0:num_classes]
            label = I[0]
            input_shape = image.detach().cpu().numpy().shape
            pert_image = copy.deepcopy(image)
            w = np.zeros(input_shape)
            r_tot = np.zeros(input_shape)
            loop_i = 0

            x = Variable(pert_image[None, :], requires_grad=True)
            fs = net.forward(x)
            fs_list = [fs[0, I[k]] for k in range(num_classes)]
            k_i = label

            while k_i == label and loop_i < max_iter:

                pert = np.inf
                fs[0, I[0]].backward(retain_graph=True)
                grad_orig = x.grad.data.cpu().numpy().copy()

                for k in range(1, num_classes):
                    zero_gradients(x)

                    fs[0, I[k]].backward(retain_graph=True)
                    cur_grad = x.grad.data.cpu().numpy().copy()

                    # set new w_k and new f_k
                    w_k = cur_grad - grad_orig
                    f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

                    pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())

                    # determine which w_k to use
                    if pert_k < pert:
                        pert = pert_k
                        w = w_k

                r_i = (pert + 1e-4) * w / np.linalg.norm(w)
                r_tot = np.float32(r_tot + r_i)
                pert_image = image + (1 + overshoot) * torch.from_numpy(r_tot).cuda()

            pert_imgs[idx] = pert_image

        return r_tot, pert_image

