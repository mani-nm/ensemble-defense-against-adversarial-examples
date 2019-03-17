import copy
import numpy as np
import torch
import random
from torch.autograd import Variable
import tensorflow as tf
from imagenet_c import corrupt
import stadv
import torch.nn.functional as F
import multiprocessing
from cleverhans.attacks import SPSA
from cleverhans.model import Model
import lib.util as util
import foolbox
from foolbox.attacks import BoundaryAttack
from foolbox.models import PyTorchModel
from torch.autograd.gradcheck import zero_gradients
from scipy.optimize import differential_evolution


class Attack(object):
    name = None

    # TODO: Refactor this out of this object
    _stop_after_n_datapoints = None  # An attack can optionally run on only a subset of the dataset

    def __call__(self, **kwargs):
        raise NotImplementedError()


class CleanData(Attack):
    # Also known as the "null attack". Just returns the unaltered clean image
    name = 'clean'

    def __call__(self, model_fn, images_batch_nhwc, y_np):
        del y_np, model_fn  # unused
        return images_batch_nhwc


class FGSM(Attack):
    name = "Fast Gradient Sign Method"

    def __init__(self, min_pixel=-2.42906570435, max_pixel=2.75373125076):
        self.min_pixel = min_pixel
        self.max_pixel = max_pixel

    def __call__(self, model, images, labels, epsilon, type='ml'):
        model.eval()
        images, labels = images.cuda(), labels.cuda()
        images, labels = Variable(images, volatile=True), Variable(labels)
        images.requires_grad = True
        output = model(images)
        # _, top_ = output.topk(10, 1)
        # # argmax = torch.tensor(top2[:, 0], requires_grad=True)
        # argmax = top_[:, 0]
        # for j in range(top_.size(0)):
        #     if argmax[j] == labels[j]:
        #         argmax[j] = top_[j, np.random.randint(0, 9, 1)]
        # argmax = Variable(argmax.data)
        loss = F.nll_loss(output, labels)
        model.zero_grad()
        loss.cuda()
        loss.backward()
        data_grad = images.grad.data
        sign_data_grad = data_grad.sign()
        pert_images = torch.add(images.data, epsilon*sign_data_grad)  #, adv_noise
        pert_images = torch.clamp(pert_images, self.min_pixel, self.max_pixel)
        return pert_images


class ILLC(Attack):
    name: "Iterative Least Likely Method"

    def __init__(self, min_pixel=-2.42906570435, max_pixel=2.75373125076):
        self.min_pixel = min_pixel
        self.max_pixel = max_pixel

    def __call__(self, model, images, labels, num_classes, alpha=0.01):
        model.eval()
        # adv_noise = 0.01
        images, labels = images.cuda(), labels.cuda()
        labels = Variable(labels)
        pert_imgs = Variable(images.data, requires_grad=True)
        output = model(pert_imgs)
        _, top_ = output.topk(num_classes, 1)
        argmax = top_[:, 0]
        for j in range(top_.size(0)):
            if argmax[j] == labels[j]:
                argmax[j] = top_[j, num_classes-1]
        argmax = Variable(argmax.data)
        i = 0
        loss = F.nll_loss(output, labels)
        loss.backward()
        gradient = torch.sign(pert_imgs.grad.data)
        while i < 5:
            pert_imgs = torch.add(pert_imgs.data,  alpha * gradient) #adv_noise,
            pert_imgs = torch.clamp(pert_imgs, self.min_pixel, self.max_pixel)  #
            pert_imgs = Variable(pert_imgs, requires_grad=True)
            model.zero_grad()
            output = model(pert_imgs)
            loss = F.nll_loss(output, argmax)
            loss.cuda()
            loss.backward()
            gradient = torch.sign(pert_imgs.grad.data)
            i += 1
        return pert_imgs


class BIM(Attack):
    name = "BIM"

    def __init__(self, min_pixel=-2.42906570435, max_pixel=2.75373125076):
        self.min_pixel = min_pixel
        self.max_pixel = max_pixel

    def __call__(self, model, images, labels, num_classes, alpha=0.01):
        model.eval()
        # adv_noise = 0.01
        images, labels = images.cuda(), labels.cuda()
        labels = Variable(labels)

        pert_imgs = Variable(images.data, requires_grad=True)
        output = model(pert_imgs)
        loss = F.nll_loss(output, labels)
        loss.backward()
        gradient = torch.sign(pert_imgs.grad.data)
        i = 0
        while i < 5:
            pert_imgs = torch.add(pert_imgs.data, alpha*gradient) #, adv_noise
            pert_imgs = torch.clamp(pert_imgs, self.min_pixel, self.max_pixel)
            pert_imgs = Variable(pert_imgs, requires_grad=True)
            model.zero_grad()
            output = model(pert_imgs)
            loss = F.nll_loss(output, labels)
            loss.cuda()
            loss.backward() #retain_graph=True
            gradient = torch.sign(pert_imgs.grad.data)
            i += 1
        return pert_imgs


class DeepFool(Attack):
    name = 'DeepFool'

    def deepfool_single(self, image, label, net, num_classes=1000, overshoot=0.02, max_iter=2):
        image = image.cuda()
        f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
        I = (np.array(f_image)).flatten().argsort()[::-1]
        I = I[0:num_classes]
        # label = I[0]
        input_shape = image.detach().cpu().numpy().shape
        pert_image = copy.deepcopy(image)
        w = np.zeros(input_shape)
        r_tot = np.zeros(input_shape)
        loop_i = 0

        x = Variable(pert_image[None, :], requires_grad=True)

        fs = net.forward(x)
        k_i = label
        fs[0, I[0]].backward(retain_graph=True)
        # grad_orig = x.grad.data.cpu().numpy().copy()

        while k_i == label and loop_i < max_iter:

            pert = np.inf
            fs[0, I[0]].backward(retain_graph=True)
            grad_orig = x.grad.data.cpu().numpy().copy()
            tgt = np.random.randint(1, num_classes - 1, size=1)
            if tgt == label.item():
                tgt = tgt + 1
            for k in tgt:
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

    def __call__(self, model, images, labels, ids, num_classes=1000, overshoot=0.02, max_iter=2):
        adv_ex = torch.Tensor(len(ids), images.size()[1], images.size()[2], images.size()[3])
        i = 0
        for id in ids:
            _, pert_image = self.deepfool_single(images[id], labels[id], model, num_classes)
            adv_ex[i] = pert_image
            i += 1
        return adv_ex


class OnePixelAttack(Attack):
    name = 'One Pixel Attack'

    def __call__(self, image, label, net, popsize=10, d=1, max_iter=10):
        pred_adv = 0
        prob_adv = 0
        shape = image.size()
        pert_image = copy.deepcopy(image)

        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        def preprocess(img):
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img.astype(np.float32)
            if img.shape[0] != 3:
                img = img.transpose(2, 0, 1)
            # img = (img - mean)/std
            return img

        def perturb(x):
            adv_img = copy.deepcopy(image)
            adv_img = adv_img.numpy()
            adv_img = adv_img.transpose(1, 2, 0)
            # calculate pixel locations and values
            pixs = np.array(np.split(x, len(x) / 5)).astype(int)
            # print(pixs.shape, pixs)
            # print()
            loc = (pixs[:, 0], pixs[:, 1])
            # print(loc, "\n")
            val = pixs[:, 2:]
            # print(val.shape, val)
            # print(adv_img[loc].shape)
            adv_img[loc] = val
            adv_img = adv_img.transpose(2, 0, 1)
            return adv_img

        def optimize(x):
            adv_img = perturb(x)

            inp = Variable(torch.from_numpy(preprocess(adv_img)).float().unsqueeze(0))
            net.cuda()
            out = net(inp.cuda())
            out = out.cpu()
            prob = softmax(out.data.numpy()[0])
            return prob[label.item()]

        def callback(x, convergence):
            global pred_adv, prob_adv
            adv_img = perturb(x)
            net.cpu()
            inp = Variable(torch.from_numpy(preprocess(adv_img)).float().unsqueeze(0))
            out = net(inp)
            # out = out.cpu()
            prob = softmax(out.data.numpy())[0]
            pred_adv = np.argmax(prob)
            # prob_adv = prob[pred_adv]
            if pred_adv != label.item():  # and prob_adv >= 0.9:
                print('Attack successful..')
                # print('Prob [%s]: %f' % (cifar10_class_names[pred_adv], prob_adv))
                print()
                return True
            # else:
            #     # print('Attack failed...')
            #     continue

        bounds = [(0, shape[0] - 1), (0, shape[1]), (0, 1), (0, 1),
                  (0, 1)] * d  # (-2.1179, 2.64), (-2.1179, 2.64), (-2.1179, 2.64)
        result = differential_evolution(optimize, bounds, maxiter=max_iter, popsize=popsize, tol=1e-5,
                                        callback=callback)

        adv_img = perturb(result.x)
        inp = Variable(torch.from_numpy(preprocess(adv_img)).float().unsqueeze(0))

        return inp


# class SpatialTransAttack(Attack):
#     name = 'Spatially Transformed Attack'
#
#     def __call__(self, inp, label, net):
#         images = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='images')
#         flows = tf.placeholder(tf.float32, [None, 2, 28, 28], name='flows')
#         targets = tf.placeholder(tf.int64, shape=[None], name='targets')
#         tau = tf.placeholder_with_default(
#             tf.constant(0., dtype=tf.float32),
#             shape=[], name='tau'
#         )
#
#         perturbed_images = stadv.layers.flow_st(images, flows, 'NHWC')

import matplotlib.pyplot as plt


def corrupt_float32_image(x, corruption_name, severity):
    """Convert to uint8 and back to conform to corruption API"""
    x = np.copy(x)  # We make a copy to avoid changing things in-place
    x = (x * 255).astype(np.uint8)

    corrupt_x = corrupt(
        x,
        corruption_name=corruption_name,
        severity=severity)
    return corrupt_x.astype(np.float32) / 255.


def _corrupt_float32_image_star(args):
    return corrupt_float32_image(*args)


class CommonCorruptionsAttack(Attack):
    name = "common_corruptions"

    def __init__(self, severity=1):
        self.corruption_names = [
            'gaussian_noise',
            # 'shot_noise',
            # 'impulse_noise',
            # 'defocus_blur',
            # 'glass_blur',
            # 'motion_blur',
            # 'zoom_blur',
            # 'snow', # Snow does not work in python 2.7
            # # 'frost', # Frost is not working correctly
            'fog',
            'brightness',
            'contrast',
            'elastic_transform',
            'pixelate',
            'jpeg_compression',
            'speckle_noise',
            'gaussian_blur',
            'spatter',
            'saturate'
        ]
        self.severity = severity
        self.pool = multiprocessing.Pool(8)  # len(self.corruption_names)

    def __call__(self, model_fn, images_batch_nhwc, y_np):
        # assert images_batch_nhwc.shape[1:] == (224, 224, 3), \
        #     "Image shape must equal (N, 224, 224, 3)"
        batch_size = len(images_batch_nhwc)
        # Keep track of the worst corruption for each image
        worst_corruption = copy.deepcopy(images_batch_nhwc)
        worst_corruption = worst_corruption.permute(0, 3, 1, 2)
        worst_loss = [np.NINF] * batch_size

        # Iterate through each image in the batch
        for batch_idx, x in enumerate(images_batch_nhwc):
            corrupt_args = [(x.data.numpy(), corruption_name, self.severity)
                            for corruption_name in self.corruption_names]
            corrupt_x_batch = self.pool.map(_corrupt_float32_image_star, corrupt_args)
            corrupt_x_batch = torch.tensor(corrupt_x_batch).permute(0, 3, 1, 2)
            # logits_batch = model_fn(corrupt_x_batch.cuda())
            # label = y_np[batch_idx]

            # This is left un-vectorized for readability
            for (logits, corrupt_x) in zip(logits_batch, corrupt_x_batch):
                correct_logit, wrong_logit = logits[label], logits[(label + 12) % 1000]

                # We can choose different loss functions to optimize in the
                # attack. For now, optimize the magnitude of the wrong logit
                # because we use this as our confidence threshold
                # loss = wrong_logit
                loss = wrong_logit - correct_logit

                if loss > worst_loss[batch_idx]:
                    worst_corruption[batch_idx] = corrupt_x
                    worst_loss[batch_idx] = loss

        return corrupt_x_batch


class BoundaryAttackClass(Attack):
    name = "Boundary Attack"

    def __call__(self, image, label, model):
        mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
        std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
        Model = PyTorchModel(model, bounds=(0, 1), num_classes=1000, preprocessing=(mean, std))
        image, label = foolbox.utils.imagenet_example(data_format='channels_first')
        image = image / 255
        print(np.argmax(Model.predictions(image)), label)

        return adv


# class BoundaryAttackClass(Attack):
#     name = "Boundary Attack"
#
#     def __init__(self, model, image_shape_hwc, max_l2_distortion=4, label_to_examples=None):
#         if label_to_examples is None:
#             label_to_examples = {}
#
#         self.max_l2_distortion = max_l2_distortion
#
#         class Model:
#             def bounds(self):
#                 return [-1, 1]
#
#             def predictions(self, img):
#                 return model(img[np.newaxis, :, :, :])[0]
#
#             def batch_predictions(self, img):
#                 return model(img)
#
#         self.label_to_examples = label_to_examples
#
#         h, w, c = image_shape_hwc
#         mse_threshold = max_l2_distortion ** 2 / (h * w * c)
#         try:
#             # Foolbox 1.5 allows us to use a threshold the attack will abort after
#             # reaching. Because we only care about a distortion of less than 4, as soon
#             # as we reach it, we can just abort and move on to the next image.
#             self.attack = BoundaryAttack(model=Model(), threshold=mse_threshold)
#         except:
#             # Fall back to the original implementation.
#             print("WARNING: Using foolbox version < 1.5 will cause the "
#                   "boundary attack to perform more work than is required. "
#                   "Please upgrade to version 1.5")
#             self.attack = BoundaryAttack(model=Model())
#
#     def __call__(self, x_np, y_np, model):
#         # r = []
#         k = []
#         for i in self.label_to_examples.keys():
#             if len(self.label_to_examples[i]) > 0:
#                 k.append(i)
#
#         other = random.choice(k)
#         initial_adv = random.choice(self.label_to_examples[other])
#         try:
#             # with util.suppress_stdout():  # Foolbox is extremely verbose, so we silence it
#             adv = self.attack(x_np, y_np,
#                               log_every_n_steps=100,  # Reduce verbosity of the attack
#                               starting_point=initial_adv
#                               )
#             distortion = np.sum((x_np - adv) ** 2) ** .5
#             if distortion > self.max_l2_distortion:
#                 # project to the surface of the L2 ball
#                 adv = x_np + (adv - x_np) / distortion * self.max_l2_distortion
#
#         except AssertionError as error:
#             if str(error).startswith("Invalid starting point provided."):
#                 print("WARNING: The model misclassified the starting point (the target) "
#                       "from BoundaryAttack. This means that the attack will fail on this "
#                       "specific point (but is likely to succeed on other points.")
#                 adv = x_np  # Just return the non-adversarial point
#             else:
#                 raise error
#
#         # r.append(adv)
#         return adv

# class CleverhansPyfuncModelWrapper(Model):
#     nb_classes = 2
#     num_classes = 2
#
#     def __init__(self, model_fn):
#         """
#         Wrap a callable function that takes a numpy array of shape (N, C, H, W),
#         and outputs a numpy vector of length N, with each element in range [0, 1].
#         """
#         self.model_fn = model_fn
#
#     def fprop(self, x, **kwargs):
#         logits_op = tf.py_func(self.model_fn, [x], tf.float32)
#         return {'logits': logits_op}
#
#
# class SpsaAttack(Attack):
#     name = 'spsa'
#
#     def __init__(self, model, image_shape_hwc, epsilon=(16. / 255),
#                  num_steps=200, batch_size=32, is_debug=False):
#         self.graph = tf.Graph()
#
#         with self.graph.as_default():
#             self.sess = tf.Session(graph=self.graph)
#
#             self.x_input = tf.placeholder(tf.float32, shape=(1,) + image_shape_hwc)
#             self.y_label = tf.placeholder(tf.int32, shape=(1,))
#
#             self.model = model
#             attack = SPSA(CleverhansPyfuncModelWrapper(self.model), sess=self.sess)
#             self.x_adv = attack.generate(
#                 self.x_input,
#                 y=self.y_label,
#                 epsilon=epsilon,
#                 num_steps=num_steps,
#                 early_stop_loss_threshold=-1.,
#                 batch_size=batch_size,
#                 is_debug=is_debug)
#
#         self.graph.finalize()
#
#     def __call__(self, model, x_np, y_np):  # (4. / 255)):
#         if model != self.model:
#             raise ValueError('Cannot call spsa attack on different models')
#         del model  # unused except to check that we already wired it up right
#
#         with self.graph.as_default():
#             all_x_adv_np = []
#             for i in range(len(x_np)):
#                 x_adv_np = self.sess.run(self.x_adv, feed_dict={
#                     self.x_input: np.expand_dims(x_np[i], axis=0),
#                     self.y_label: np.expand_dims(y_np[i], axis=0),
#                 })
#                 all_x_adv_np.append(x_adv_np)
#             return np.concatenate(all_x_adv_np)
#
#
# class Loss_flow(torch.nn.Module):
#     def __init__(self, neighbours=np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])):
#         super(Loss_flow, self).__init__()
#
#         filters = []
#         for i in range(neighbours.shape[0]):
#             for j in range(neighbours.shape[1]):
#                 if neighbours[i][j] == 1:
#                     filter = np.zeros((1, neighbours.shape[0], neighbours.shape[1]))
#                     filter[0][i][j] = -1
#                     filter[0][neighbours.shape[0] // 2][neighbours.shape[1] // 2] = 1
#                     filters.append(filter)
#         filters = np.array(filters)
#         self.filters = torch.from_numpy(filters).float()
#
#     def forward(self, f):
#         # TODO: padding
#         '''
#         f - f.size() =  [1, h, w, 2]
#             f[0, :, :, 0] - u channel
#             f[0, :, :, 1] - v channel
#         '''
#         f_u = f[:, :, :, 0].unsqueeze(1)
#         f_v = f[:, :, :, 1].unsqueeze(1)
#
#         diff_u = F.conv2d(f_u, self.filters)[0][0]  # don't use squeeze
#         diff_u_sq = torch.mul(diff_u, diff_u)
#
#         diff_v = F.conv2d(f_v, self.filters)[0][0]  # don't use squeeze
#         diff_v_sq = torch.mul(diff_v, diff_v)
#
#         dist = torch.sqrt(torch.sum(diff_u_sq, dim=0) + torch.sum(diff_v_sq, dim=0))
#         return torch.sum(dist)
#
#
# def CWLoss(logits, target, kappa=0, num_classes=10):
#     # inputs to the softmax function are called logits.
#     # https://arxiv.org/pdf/1608.04644.pdf
#     target = torch.ones(logits.size(0)).type(logits.type()).fill_(target)
#     target_one_hot = torch.eye(num_classes).type(logits.type())[target.long()]
#
#     # workaround here.
#     # subtract large value from target class to find other max value
#     # https://github.com/carlini/nn_robust_attacks/blob/master/l2_attack.py
#     real = torch.sum(target_one_hot * logits, 1)
#     other = torch.max((1 - target_one_hot) * logits - (target_one_hot * 10000), 1)[0]
#     kappa = torch.zeros_like(other).fill_(kappa)
#
#     return torch.sum(torch.max(other - real, kappa))
#
#
# class SpatialAttack(Attack):
#     name = 'Spatially Transformed Attack'
#
#     def __call__(self, image, label, net, num_classes=1000, max_iter=10):
#         pert_image = copy.deepcopy(image)
#         x = Variable(image.unsqueeze(0), requires_grad=True)
#
#         theta = torch.tensor([[1, 0, 0], [0, 1, 0]]).unsqueeze(0).float()  # identity transformation
#         grid = F.affine_grid(theta, x.size())  # flow = 0. This is base grid
#         # grid.size() = (1, h, w, 2)
#
#         f = Variable(torch.zeros_like(grid).float(), requires_grad=True)
#         torch.nn.init.normal_(f, mean=0.456, std=0.224)
#
#         grid_new = grid + f
#         grid_new = grid_new.clamp(min=-1, max=1)
#         x_new = F.grid_sample(x, grid_new, mode='bilinear')
#
#         optimizer = torch.optim.SGD([f, ], lr=0.005)  # optimizer = torch.optim.LBFGS([f, ], lr=lr)
#         loss_flow = Loss_flow()
#         loss_adv = CWLoss
#         i = 0
#         while i < max_iter:
#             optimizer.zero_grad()
#
#             logits = net(x_new.cuda())  # .detach() for LBFGS
#             # pred = np.argmax(logits.data.numpy())
#             # _, pred = torch.max(logits.data, 1)
#             logits = logits.cpu()
#             loss = loss_adv(logits, label, num_classes=num_classes) + num_classes * loss_flow(f)
#             loss.backward()
#             optimizer.step()
#
#             # update variables and predict on adversarial image
#             grid_new = grid + f
#             grid_new = grid_new.clamp(min=-1, max=1)
#             x_new = F.grid_sample(x, grid_new, mode='bilinear')
#
#             pred_adv = net(x_new.cuda())
#             pred_adv = np.argmax(pred_adv.cpu().data.numpy())
#             if label.item() != pred_adv:
#                 break
#             i += 1
#
#         return x_new

from foolbox.attacks import LBFGSAttack
from foolbox.criteria import Misclassification
from foolbox.models import PyTorchModel


class LBFGS(Attack):
    name = "LBFGS"

    def __call__(self, image, label, model):
        mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
        std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
        fmodel = PyTorchModel(model, bounds=(0, 1), num_classes=1000, preprocessing=(mean, std))
        criterion = Misclassification()
        attack = LBFGSAttack(fmodel, criterion)
        images, labels = foolbox.utils.samples('imagenet', )
        adversarial = attack(image, label=label)
        return adversarial


def get_pred_single(x, model):
    x = x.cuda()
    outputs = model(x)
    _, predicted = torch.max(outputs.data, 1)

    return predicted
