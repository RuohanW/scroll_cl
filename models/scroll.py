# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from utils.args import *
from models.utils.continual_model import ContinualModel
import math
from utils.buffer import Buffer
import torch
import numpy as np
from utils.status import progress_bar

from scipy import linalg
from utils.tsa import resnet_tsa, get_dataloader, TensorDataset, tensor_aug_transform

from collections import defaultdict
from torchvision import transforms
from PIL import Image

from utils.conf import base_path

import torch.nn.functional as F


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via'
                                        ' Progressive Neural Networks.')
    add_management_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='Ridge Regression L2 penality')
    parser.add_argument('--lr_cl', type=float, default=0.1,
                        help='Learning rate for classifier')
    parser.add_argument('--er_iters', type=int, default=40,
                        help='ER iterations')
    parser.add_argument('--buffer_method', type=str, default="get_indices_minimum_dist_mean",
                        help="Buffering method.")
    parser.add_argument('--scale_init', type=int, default=1,
                        help="Scale initialisation.")
    add_experiment_args(parser)
    return parser

def cuda_to_np(tensor):
    return tensor.cpu().detach().numpy()


resize_transform = transforms.Compose(
        [
         transforms.ToPILImage(),
         lambda x: x.resize((84, 84), resample=Image.LANCZOS),
         transforms.ToTensor(),
         lambda x: x*2 - 1])

def batch_resize(xs):
    xs_cpu = xs.cpu().detach()
    ret = [resize_transform(xs_cpu[i]) for i in range(xs_cpu.shape[0])]
    ret = torch.stack(ret)
    return ret

def np_one_hot(ind, n_ways):
    ret = np.zeros((ind.shape[0], n_ways))
    ret[np.arange(ind.shape[0]), ind] = 1
    return ret


def get_indices_minimum_dist_mean(samples_per_class, feats):
    idxs = []

    mean_feat = feats.mean(0, keepdim=True)

    running_sum = torch.zeros_like(mean_feat)
    for i in range(min(samples_per_class, feats.shape[0])):
        cost = (mean_feat - (feats + running_sum) / (i + 1)).norm(2, 1)

        idxs.append(cost.argmin().item())

        running_sum += feats[idxs[i]:idxs[i] + 1]
        feats[idxs[i]] = feats[idxs[i]] + float('inf')
    return idxs

def get_indices_random(samples_per_class, feats):
    idxs = list(np.random.choice(feats.shape[0], min(samples_per_class, feats.shape[0]), replace=False))
    return idxs

def get_indices_minimum_dist_mean_live(samples_per_class, feats, mean_feat):
    idxs = []

    running_sum = torch.zeros_like(mean_feat)
    for i in range(min(samples_per_class, feats.shape[0])):
        cost = (mean_feat - (feats + running_sum) / (i + 1)).norm(2, 1)

        idxs.append(cost.argmin().item())

        running_sum += feats[idxs[i]:idxs[i] + 1]
        feats[idxs[i]] = feats[idxs[i]] + float('inf')
    return idxs

def get_indices_random_live(samples_per_class, feats, mean_feat):
    idxs = list(np.random.choice(feats.shape[0], min(samples_per_class, feats.shape[0]), replace=False))
    return idxs


def fill_buffer(self, mem_buffer: Buffer, dataset, t_idx: int, get_index_func) -> None:
        """
        Adds examples from the current task to the memory buffer
        by means of the herding strategy. Copied from icarl.py.
        :param mem_buffer: the memory buffer
        :param dataset: the dataset from which take the examples
        :param t_idx: the task index
        """

        mode = self.net.training
        self.net.eval()
        samples_per_class = mem_buffer.buffer_size // len(self.classes_so_far)
        classes_with_extra = mem_buffer.buffer_size % len(self.classes_so_far)

        # Choose classes to have extra
        extra_classes = list(np.random.choice(len(self.classes_so_far), classes_with_extra, replace=False))
        non_extra_classes = [i for i in range(len(self.classes_so_far)) if i not in extra_classes]

        if t_idx > 0:
            # 1) First, subsample prior classes
            buf_x, buf_y = mem_buffer.get_all_data()

            # If adding few classes when there are many classes,
            # we will add extra for a class that does not have so many samples in the buffer.
            # This is because the quotient may not change, while the random extra class changes.
            # So we check for this and choose another class to have extra.
            cannot_extra_class = []
            for _y in buf_y.unique():
                y = _y.detach().cpu().numpy()
                idx = (buf_y == _y)
                if len(idx[idx]) < samples_per_class + 1:
                    cannot_extra_class.append(y)
            for y in cannot_extra_class:
                if y in non_extra_classes:
                    non_extra_classes.remove(y)
            for y in cannot_extra_class:
                if y in extra_classes:
                    extra_classes.remove(y)
                    new_extra_class = np.random.choice(non_extra_classes, 1)[0]
                    non_extra_classes.remove(new_extra_class)
                    extra_classes.append(new_extra_class)

            mem_buffer.empty()
            for _y in buf_y.unique():
                add_extra = 1 if _y.detach().cpu().numpy() in extra_classes else 0
                idx = (buf_y == _y)
                _y_x, _y_y = buf_x[idx], buf_y[idx]
                mem_buffer.add_data(
                    examples=_y_x[:samples_per_class + add_extra],
                    labels=_y_y[:samples_per_class + add_extra]
                )

        # 2) Then, fill with current tasks
        loader = dataset.train_loader
        norm_trans = batch_resize
        if dataset.SETTING == "class-il":
            classes_start, classes_end = t_idx * dataset.N_CLASSES_PER_TASK, (t_idx + 1) * dataset.N_CLASSES_PER_TASK
        elif dataset.SETTING == "domain-il":
            classes_start, classes_end = 0, dataset.N_CLASSES_PER_TASK
        else:
            raise Exception

        a_x, a_y, a_f = [], [], []
        for x, y, not_norm_x in loader:
            mask = (y >= classes_start) & (y < classes_end)
            x, y, not_norm_x = x[mask], y[mask], not_norm_x[mask]
            if not x.size(0):
                continue
            x, y, not_norm_x = (a.to(self.device) for a in [x, y, not_norm_x])
            a_x.append(not_norm_x.to('cpu'))
            a_y.append(y.to('cpu'))
            feats = self.net.embed(norm_trans(not_norm_x).to(self.device))
            feats = F.normalize(feats, p=2., dim=1)
            a_f.append(feats.cpu())
        a_x, a_y, a_f = torch.cat(a_x), torch.cat(a_y), torch.cat(a_f)

        # 2.2 Compute class means
        for _y in a_y.unique():
            add_extra = 1 if _y.detach().cpu().numpy() in extra_classes else 0
            idx = (a_y == _y)
            _x, _y = a_x[idx], a_y[idx]
            feats = a_f[idx]

            idxs = get_index_func(samples_per_class + add_extra, feats)

            mem_buffer.add_data(
                examples=norm_trans(_x[idxs]).to(self.device),
                labels=_y[idxs].to(self.device),
            )

        self.net.train(mode)


def fill_buffer_live(self, mem_buffer: Buffer, not_aug_inputs, labels, x_feat, class_means, t_idx: int, get_index_live_func) -> None:
    """
    Adds examples from the current task to the memory buffer
    by means of the herding strategy. Copied from icarl.py.
    :param mem_buffer: the memory buffer
    :param dataset: the dataset from which take the examples
    :param t_idx: the task index
    """

    mode = self.net.training
    self.net.eval()

    prior_classes = dict()

    if not mem_buffer.is_empty():
        # 1) First, subsample prior classes
        buf_x, buf_y, buf_feat = mem_buffer.get_all_data()

        mem_buffer.empty()
        for _y in buf_y.unique():
            idx = (buf_y == _y)
            _y_x, _y_y, _y_feat = buf_x[idx], buf_y[idx], buf_feat[idx]
            y = int(_y.detach().cpu().numpy())
            if y not in prior_classes:
                prior_classes[y] = [torch.tensor([], device=self.device), torch.tensor([], device=self.device), torch.tensor([], device=self.device)]
            prior_classes[y][0] = torch.concat([prior_classes[y][0], _y_x])
            prior_classes[y][1] = torch.concat([prior_classes[y][1], _y_y])
            prior_classes[y][2] = torch.concat([prior_classes[y][2], _y_feat])

    # 2) Then, fill with current batch
    norm_trans = batch_resize
    # x_feat after normalize in observe originally, now we have constant term but should not cause any problems

    # 2.2 Compute class means
    new_classes = dict()
    for _y in labels.unique():
        idx = (labels == _y)
        _y_x, _y_y, _y_feats = norm_trans(not_aug_inputs[idx]).to(self.device), labels[idx], x_feat[idx]
        y = int(_y.detach().cpu().numpy())
        if y not in new_classes:
            new_classes[y] = [torch.tensor([], device=self.device), torch.tensor([], device=self.device), torch.tensor([], device=self.device)]
        new_classes[y][0] = torch.concat([new_classes[y][0], _y_x])
        new_classes[y][1] = torch.concat([new_classes[y][1], _y_y])
        new_classes[y][2] = torch.concat([new_classes[y][2], _y_feats])

    all_classes = dict()
    for i in prior_classes:
        all_classes[i] = prior_classes[i]
    for i in new_classes:
        if i in all_classes:
            all_classes[i][0] = torch.concat([all_classes[i][0], new_classes[i][0]])
            all_classes[i][1] = torch.concat([all_classes[i][1], new_classes[i][1]])
            all_classes[i][2] = torch.concat([all_classes[i][2], new_classes[i][2]])
        else:
            all_classes[i] = new_classes[i]

    class_samples = []
    for i in all_classes:
        class_samples.append((len(all_classes[i][0]), i))
    class_samples.sort()
    remaining_buffer_size = mem_buffer.buffer_size
    classes = len(all_classes)
    buffer_class_samples = dict()
    for curr_class_info in class_samples:
        curr_class_qty, curr_class = curr_class_info
        curr_buffer_class_sample = min((remaining_buffer_size + classes - 1) // classes, curr_class_qty)
        buffer_class_samples[curr_class] = curr_buffer_class_sample
        remaining_buffer_size -= curr_buffer_class_sample
        classes -= 1

    for _y in all_classes:
        idxs = get_index_live_func(buffer_class_samples[_y], all_classes[_y][2].clone().detach(), class_means[_y].unsqueeze(0))

        mem_buffer.add_data(
            examples=all_classes[_y][0][idxs].to(self.device),
            labels=all_classes[_y][1][idxs].to(self.device),
            logits=all_classes[_y][2][idxs]
        )

    self.net.train(mode)


def residual_adapt(classifier, scale, device, batch_size=250, max_iter=40, lr=0.1, lr_cl=0.01, tsa_net=None, buffer=None):
    tsa_net.reset()
    tsa_net.to(device)

    # Get params list for optimiser
    alpha_params = [v for k, v in tsa_net.named_parameters() if 'alpha' in k]

    params = []
    params.append({'params': alpha_params})
    params.append({'params': [scale], 'lr': 1.})
    params.append({'params': list(classifier.parameters()), 'lr': lr_cl})

    # Initialise optimiser
    # TODO: Check if correct optimiser
    optimizer = torch.optim.Adadelta(params, lr=lr)

    # all_inputs, all_labels = buffer.get_data(len(buffer.examples), transform=lambda x: x)
    all_inputs, all_labels, all_feats = buffer.get_data(len(buffer.examples), transform=lambda x: x)

    dataset = TensorDataset(all_inputs.cpu(), all_labels.cpu(), tensor_aug_transform)

    dataloader = get_dataloader(dataset, batch_size, 4, shuffle=True, drop_last=True)

    # Get number of epochs
    epoch_iter = len(dataloader)

    num_epochs = math.floor(max_iter / epoch_iter + 1)

    # Train
    for i in range(num_epochs):
        for batch_x, batch_y, _ in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            context_features = tsa_net.embed(batch_x)
            context_features = F.normalize(context_features, p=2., dim=1)
            logits = classifier(context_features)
            loss = F.cross_entropy(logits * scale.to(device), batch_y)
            loss.backward()
            optimizer.step()
        progress_bar(i, num_epochs, 1, 'G', loss.item())


def full_train(classifier, scale, device, batch_size=250, max_iter=40, lr=0.1, lr_cl=0.01, tsa_net=None, buffer=None):
    # Initialise model
    tsa_net = tsa_net.to(device)
    tsa_net.train()

    # Get params list for optimiser
    params = []
    params.append({'params': list(tsa_net.parameters())})
    # params.append({'params': [scale], 'lr': 1.})
    params.append({'params': list(classifier.parameters()), 'lr': lr_cl})

    # Initialise optimiser
    # TODO: Check if correct optimiser
    optimizer = torch.optim.Adadelta(params, lr=lr)

    # Create dataset from buffer
    all_inputs, all_labels, all_feats = buffer.get_data(len(buffer.examples), transform=lambda x: x)
    # all_inputs, all_labels = buffer.get_data(len(buffer.examples), transform=lambda x: x)
    dataset = TensorDataset(all_inputs.cpu(), all_labels.cpu(), tensor_aug_transform)
    dataloader = get_dataloader(dataset, batch_size, 4, shuffle=True, drop_last=True)
    epoch_iter = len(dataloader)

    num_epochs = math.floor(max_iter / epoch_iter)

    for i in range(num_epochs):
        losses = []
        for batch_x, batch_y, _ in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()

            context_features = tsa_net.embed(batch_x)
            context_features = F.normalize(context_features, p=2., dim=1)
            logits = classifier(context_features)
            loss = F.cross_entropy(logits * scale.to(device), batch_y)

            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        progress_bar(i, num_epochs, 1, 'G', loss.item())


class Scroll(ContinualModel):
    NAME = 'scroll'
    COMPATIBILITY = ['class-il', 'task-il', "domain-il"]

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.task = 0

        self.centroids = defaultdict(int)
        self.cov_mat = 0.

        self.tsa_net = self.net

        self.classifier = None

        self.scale_init = self.args.scale_init

        self.class_counts = dict()
        self.mean_feat = dict()

    def observe(self, inputs, labels, not_aug_inputs):
        if not hasattr(self, 'classes_so_far'):
            self.register_buffer('classes_so_far', labels.unique().to('cpu'))
        else:
            self.register_buffer('classes_so_far', torch.cat((
                self.classes_so_far, labels.to('cpu'))).unique())
        #this is crucial. The training pipeline keeps switching the net back to training mode, which we don't need
        #batch norm will get wrong stats if in training mode
        self.net.eval()
        # no_aug_inputs are 32 by 32
        aug_resized_inputs = batch_resize(not_aug_inputs).to(self.device)
        with torch.no_grad():
            x_feat = self.net.embed(aug_resized_inputs)
            x_feat = F.normalize(x_feat, p=2., dim=1)
            x_feat_t = x_feat
            x_feat = cuda_to_np(x_feat).astype(np.float64)
            x_feat = np.concatenate([x_feat, np.ones((x_feat.shape[0], 1))], axis=1)
            y_np = cuda_to_np(labels)

            self.cov_mat += np.matmul(x_feat.T, x_feat)

            for x, y in zip(x_feat, y_np):
                y_int = int(y.item())
                self.centroids[y_int] += x


        for _y in labels.unique():
            idx = (labels == _y)
            _y_x, _y_y, _y_feats = not_aug_inputs[idx], labels[idx], x_feat_t[idx]
            y = int(_y.detach().cpu().numpy())
            if y not in self.mean_feat:
                self.mean_feat[y] = 0
                self.class_counts[y] = 0
            self.mean_feat[y] = ((self.mean_feat[y] * self.class_counts[y]) + _y_feats.sum(0)) / (self.class_counts[y] + len(_y_feats))
            self.class_counts[y] += len(_y_feats)

        with torch.no_grad():
            fill_buffer_live(self, self.buffer, not_aug_inputs, labels, x_feat_t, self.mean_feat, self.task, eval(self.args.buffer_method + "_live"))

        return 0

    def least_square(self):
        tmp = self.centroids
        centroids = []
        for i in range(len(tmp)):
            centroids.append(tmp[i])
        centroids = np.stack(centroids, axis=0)
        c_sum = np.sum(centroids, axis=0, keepdims=True)
        c_target = (2 * centroids - c_sum).transpose()
        # c_target = centroids.transpose()
        nc = c_target.shape[0]
        alpha = self.args.alpha
        eye = np.eye(nc) * alpha
        cov = self.cov_mat + eye
        w = linalg.solve(cov, c_target, sym_pos=True, overwrite_a=True)
        return w.T

    def end_task(self, dataset):
        # with torch.no_grad():
        #     fill_buffer(self, self.buffer, dataset, self.task, eval(self.args.buffer_method))

        self.task += 1
        if not (self.task == dataset.N_TASKS):
            return
        # d = dict()
        # data, labels = self.buffer.get_data(2000, lambda x: x)
        # for i in labels:
        #     i = int(i.detach().cpu().numpy())
        #     d[i] = d.get(i, 0) + 1
        # print(d)
        weights = self.least_square()
        # print(weights.shape)
        # print(weights)
        weights, bias = np.split(weights, [512], axis=1)
        num_cls, feat_dim = weights.shape

        self.classifier = torch.nn.Linear(feat_dim, num_cls)
        self.classifier.weight.data = torch.Tensor(weights.astype(np.float32))
        self.classifier.bias.data = torch.Tensor(bias.T.astype(np.float32))
        self.classifier.to(self.device)

        self.net.eval()

        self.scale = torch.nn.Parameter(torch.Tensor(1))
        self.scale.requires_grad = True
        self.scale.data.fill_(self.scale_init)

        batch_size = self.args.minibatch_size if self.args.minibatch_size < self.args.buffer_size else self.args.buffer_size

        if len(self.buffer) <=500:
            opt = lambda x: x
            opt.tsa_alpha = True
            opt.tsa_beta = False
            self.tsa_net = resnet_tsa(self.net, opt)
            self.tsa_net.reset()  # Initialise adapters and pa
            self.tsa_net.to(self.device)

            residual_adapt(self.classifier, self.scale, self.device, batch_size=batch_size, max_iter=self.args.er_iters,
                           lr=self.args.lr, lr_cl=self.args.lr_cl, buffer=self.buffer, tsa_net=self.tsa_net)
        else:
            full_train(self.classifier, self.scale, self.device, batch_size=batch_size, max_iter=self.args.er_iters,
                           lr=self.args.lr, lr_cl=self.args.lr_cl, buffer=self.buffer, tsa_net=self.tsa_net)


    def forward(self, xs):
        with torch.no_grad():
            if self.classifier is not None:
                target_features = self.tsa_net.embed(xs)
                target_features = F.normalize(target_features, p=2., dim=1)
                logits = self.classifier(target_features)
                return logits
            else:
                return self.net(xs)

