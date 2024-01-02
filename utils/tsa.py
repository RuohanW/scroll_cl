'''
tsa.py
Created by Wei-Hong Li [https://weihonglee.github.io]
This code allows you to attach task-specific parameters, including adapters, pre-classifier alignment (PA) mapping
from 'Universal Representation Learning from Multiple Domains for Few-shot Classification'
(https://arxiv.org/pdf/2103.13841.pdf), to a pretrained backbone.
It only learns attached task-specific parameters from scratch on the support set to adapt
the pretrained model for previously unseen task with very few labeled samples.
'Cross-domain Few-shot Learning with Task-specific Adapters.' (https://arxiv.org/pdf/2107.00358.pdf)
'''

import torch
import torch.nn as nn
import math
import copy
import torch.nn.functional as F

from torchvision import transforms

from torch.utils.data import Dataset, DataLoader

import numpy as np
import random

from utils.status import progress_bar

def init_worker_fn(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(dataset, batch_size, n_workers, shuffle=True, drop_last=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=n_workers,
                      pin_memory=True, worker_init_fn=init_worker_fn)


def prototype_loss(support_embeddings, support_labels,
                   query_embeddings, query_labels, distance='cos'):
    n_way = len(query_labels.unique())

    prots = compute_prototypes(support_embeddings, support_labels, n_way).unsqueeze(0)
    embeds = query_embeddings.unsqueeze(1)

    if distance == 'l2':
        logits = -torch.pow(embeds - prots, 2).sum(-1)    # shape [n_query, n_way]
    elif distance == 'cos':
        logits = F.cosine_similarity(embeds, prots, dim=-1, eps=1e-30) * 10
    elif distance == 'lin':
        logits = torch.einsum('izd,zjd->ij', embeds, prots)
    elif distance == 'corr':
        logits = F.normalize((embeds * prots).sum(-1), dim=-1, p=2) * 10

    return cross_entropy_loss(logits, query_labels)


def cross_entropy_loss(logits, targets):
    preds = logits.argmax(1)
    labels = targets.type(torch.long)
    loss = F.cross_entropy(logits, labels, reduction="mean")
    acc = torch.eq(preds, labels).float().mean()
    stats_dict = {'loss': loss.item(), 'acc': acc.item()}
    pred_dict = {'preds': preds.cpu().numpy(), 'labels': labels.cpu().numpy()}
    return loss, stats_dict, pred_dict


def compute_prototypes(embeddings, labels, n_way):
    prots = torch.zeros(n_way, embeddings.shape[-1]).type(
        embeddings.dtype).to(embeddings.device)
    for i in range(n_way):
        if torch.__version__.startswith('1.1'):
            prots[i] = embeddings[(labels == i).nonzero(), :].mean(0)
        else:
            prots[i] = embeddings[(labels == i).nonzero(as_tuple=False), :].mean(0)
    return prots


class conv_tsa(nn.Module):
    def __init__(self, orig_conv, opt):
        super(conv_tsa, self).__init__()
        # the original conv layer
        self.conv = copy.deepcopy(orig_conv)
        self.conv.weight.requires_grad = False
        planes, in_planes, _, _ = self.conv.weight.size()
        stride, _ = self.conv.stride

        self.tsa_alpha = opt.tsa_alpha
        # task-specific adapters
        if self.tsa_alpha:
            self.alpha = nn.Parameter(torch.ones(planes, in_planes, 1, 1)) # initialization done at resnet_tsa.reset()
            self.alpha.requires_grad = True

    def forward(self, x):
        y = self.conv(x)
        if self.tsa_alpha:
            # residual adaptation in matrix form
            y = y + F.conv2d(x, self.alpha, stride=self.conv.stride)
        return y

class pa(nn.Module):
    """
    pre-classifier alignment (PA) mapping from 'Universal Representation Learning from Multiple Domains for Few-shot Classification'
    (https://arxiv.org/pdf/2103.13841.pdf)
    """
    def __init__(self, feat_dim):
        super(pa, self).__init__()
        # define pre-classifier alignment mapping
        self.weight = nn.Parameter(torch.ones(feat_dim, feat_dim, 1, 1))
        self.weight.requires_grad = True

    def forward(self, x):
        if len(list(x.size())) == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.conv2d(x, self.weight.to(x.device)).flatten(1)
        return x

class resnet_tsa(nn.Module):
    """ Attaching task-specific adapters (alpha) and/or PA (beta) to the ResNet backbone """
    def __init__(self, orig_resnet, opt):
        super(resnet_tsa, self).__init__()
        # freeze the pretrained backbone
        for k, v in orig_resnet.named_parameters():
                v.requires_grad=False

        # attaching task-specific adapters (alpha) to each convolutional layers
        # note that we only attach adapters to residual blocks in the ResNet
        for block in orig_resnet.layer1:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_tsa(m, opt)
                    setattr(block, name, new_conv)

        for block in orig_resnet.layer2:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_tsa(m, opt)
                    setattr(block, name, new_conv)

        for block in orig_resnet.layer3:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_tsa(m, opt)
                    setattr(block, name, new_conv)

        for block in orig_resnet.layer4:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_tsa(m, opt)
                    setattr(block, name, new_conv)

        self.backbone = orig_resnet

        # attach pre-classifier alignment mapping (beta)
        feat_dim = orig_resnet.layer4[-1].bn2.num_features
        beta = pa(feat_dim)
        setattr(self, 'beta', beta)

    def forward(self, x):
        return self.backbone.forward(x=x)

    def embed(self, x):
        return self.backbone.embed(x)

    def get_state_dict(self):
        """Outputs all the state elements"""
        return self.backbone.state_dict()

    def get_parameters(self):
        """Outputs all the parameters"""
        return [v for k, v in self.backbone.named_parameters()]

    def reset(self):
        # initialize task-specific adapters (alpha)
        for k, v in self.backbone.named_parameters():
            if 'alpha' in k:
                v.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device) * 0.0001

        # initialize pre-classifier alignment mapping (beta)
        v = self.beta.weight
        self.beta.weight.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device)


tensor_aug_transform = transforms.Compose(
    [
        # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomCrop(84, padding=6),
        transforms.RandomHorizontalFlip(),
    ]
)

class TensorDataset(Dataset):
    def __init__(self, data, labels, transform):
        super().__init__()
        self.transform = transform

        self.data = data
        self.labels = labels

        self._len = self.data.shape[0]

    def __getitem__(self, item):
        idx = item
        img = self.transform(self.data[idx])
        target = self.labels[idx]

        return img, target, item

    def __len__(self):
        return self._len


class WTensorDataset(Dataset):
    def __init__(self, data, labels, weights, transform):
        super().__init__()
        self.transform = transform

        self.data = data
        self.labels = labels
        self.weights = weights

        self._len = self.data.shape[0]

    def __getitem__(self, item):
        idx = item
        img = self.transform(self.data[idx])
        target = self.labels[idx]
        weights = self.weights[idx]

        return img, target, weights, item

    def __len__(self):
        return self._len



def tsa(context_images, context_labels, model, classifier, scale, opt, max_iter=40, lr=0.1, lr_beta=1, lr_cl=0.1, batch_size=256, distance='cos'):
    """
    Optimizing task-specific parameters attached to the ResNet backbone,
    e.g. adapters (alpha) and/or pre-classifier alignment mapping (beta)
    """
    model.eval()
    alpha_params = [v for k, v in model.named_parameters() if 'alpha' in k]
    beta_params = [v for k, v in model.named_parameters() if 'beta' in k]
    params = []
    if opt.tsa_alpha:
        params.append({'params': alpha_params})
    if opt.tsa_beta:
        params.append({'params': beta_params, 'lr': lr_beta})

    params.append({'params': [scale], 'lr':1.})

    params.append({'params': list(classifier.parameters()), 'lr': lr_cl})

    optimizer = torch.optim.Adadelta(params, lr=lr)

    dataset = TensorDataset(context_images.cpu(), context_labels.cpu(), tensor_aug_transform)

    # target_iter = int(len(dataset) * max_iter / batch_size)

    dataloader = get_dataloader(dataset, batch_size, 4, shuffle=True, drop_last=False)
    epoch_iter = len(dataloader)

    # num_epochs = math.floor(target_iter / epoch_iter +1)
    num_epochs = math.floor(max_iter / epoch_iter + 1)
    # num_epochs = 100
    print(num_epochs)

    losses = []

    for i in range(num_epochs):
        for batch_x, batch_y, _ in dataloader:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            optimizer.zero_grad()
            model.zero_grad()

            if opt.tsa_alpha:
                # adapt features by task-specific adapters
                context_features = model.embed(batch_x)
                context_features = F.normalize(context_features, p=2., dim=1)
            if opt.tsa_beta:
                # adapt feature by PA (beta)
                aligned_features = model.beta(context_features)
            else:
                aligned_features = context_features

            # aligned_features = F.normalize(aligned_features, p=2., dim=1) #this seems important, acts as nonlinearity
            logits = classifier(aligned_features)
            loss = F.cross_entropy(logits * scale.cuda(), batch_y)
            # loss, stat, _ = prototype_loss(aligned_features, context_labels,
            #                                aligned_features, context_labels, distance=distance)

            loss.backward()
            optimizer.step()
        progress_bar(i, num_epochs, 1, 'G', loss.item())
        losses.append(loss.item())

    all_inputs, all_labels = buffer.get_data(len(buffer.examples), transform=lambda x: x)

    dataset = TensorDataset(all_inputs.cpu(), all_labels.cpu(), tensor_aug_transform)

    dataloader = get_dataloader(dataset, batch_size, 4, shuffle=True, drop_last=False)

    # Get number of epochs
    epoch_iter = len(dataloader)

    num_epochs = math.floor(max_iter / epoch_iter + 1)

    # Train
    all_losses = []
    all_accs = []
    iters = 0
    for i in range(num_epochs):
        losses = []
        for batch_x, batch_y, _ in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()

            if tsa_alpha:
                # adapt features by task-specific adapters
                context_features = tsa_net.embed(batch_x)
                if normalise:
                    context_features = F.normalize(context_features, p=2.,
                                                   dim=1)  # this seems important, acts as nonlinearity
            if tsa_beta:
                # adapt feature by PA (beta)
                aligned_features = tsa_net.beta(context_features)
            else:
                aligned_features = context_features

            logits = classifier(aligned_features)
            loss = F.cross_entropy(logits * scale.to(device), batch_y)

            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return losses
