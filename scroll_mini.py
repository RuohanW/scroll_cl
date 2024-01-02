import os.path

import numpy
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import torch.optim
from utils.conf import base_path
from utils.tsa import get_dataloader, TensorDataset, resnet_tsa
import numpy as np
import scipy
import math

from sklearn.metrics import accuracy_score

from datasets.mini_imagenet import ImageNet, default_transform, normalize
from backbone.resnet12 import resnet12

from utils.conf import base_path

mimg_model_path = os.path.expanduser(f"{base_path()}/res12_mini_45")


tensor_aug_transform = transforms.Compose(
    [
        lambda x: Image.fromarray(x),
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomCrop(84, padding=6),
        transforms.RandomHorizontalFlip(),
        normalize,
    ]
)


def np_one_hot(ind, n_ways, default_val=0):
    ret = np.ones((ind.shape[0], n_ways)) * default_val
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


def fill_buffer(dataset, x_feat, buffer_size, get_index_func):
    labels = dataset.get_labels()
    uniq_labels = np.unique(labels)

    num_per_cls = buffer_size // uniq_labels.shape[0]
    out_x, out_y = [], []
    for _y in uniq_labels:
        idx = (labels == _y)
        _x = dataset.data[idx]
        feats = x_feat[idx]
        idx = get_index_func(num_per_cls, feats)

        tmp = _x[idx]
        out_x.append(tmp)
        out_y.append((_y * np.ones(tmp.shape[0])).astype(np.int32))
    return np.concatenate(out_x, axis=0), np.concatenate(out_y, axis=0)


def get_repr(dataset, model, device="cuda"):
    data_loader = get_dataloader(dataset, batch_size=128, n_workers=4, shuffle=False, drop_last=False)

    # Get representation
    model.eval()
    x_feat = []
    with torch.no_grad():
        for batch_x, _ in data_loader:
            curr_x_feat = model.embed(batch_x.to(device))
            curr_x_feat = F.normalize(curr_x_feat, p=2., dim=1)
            x_feat.append(curr_x_feat.cpu().numpy())

    x_feat = np.concatenate(x_feat, axis=0)
    return x_feat


def save_ls_square_data(name, dataset, model, device="cuda"):
    x_feat_no_bias = get_repr(dataset, model, device=device)
    x_feat = np.concatenate([x_feat_no_bias, np.ones((x_feat_no_bias.shape[0], 1))], axis=1)

    # Get number of classes
    labels = dataset.get_labels()
    num_cls = np.unique(labels).shape[0]

    # Check if covariance and b has been calculated before. If so, load from file.
    cov_mat_file_path = base_path() + f"{name}_cov_mat_with_bias"
    b_file_path = base_path() + f"{name}_b_with_bias"

    # if os.path.isfile(cov_mat_file_path) and os.path.isfile(b_file_path):
    #     return torch.load(cov_mat_file_path), torch.load(b_file_path), x_feat_no_bias

    # Calculate covariance and b
    cov_mat = np.matmul(x_feat.T, x_feat)
    targets = dataset.get_labels().astype(np.int)
    y_one_hot = np_one_hot(targets, num_cls, default_val=-1)
    b = np.matmul(x_feat.T, y_one_hot)

    # print(cov_mat.shape)
    # print(cov_mat)
    #
    # print(b.shape)
    # print(b)

    # Save covariance and b to file
    torch.save(cov_mat, cov_mat_file_path)
    torch.save(b, b_file_path)

    return cov_mat, b, x_feat_no_bias


def save_proto_data(save_path, name, feat, num_cls):
    mat_feat = np.reshape(feat, (num_cls, -1, feat.shape[-1]))
    feat_avg = np.mean(mat_feat, axis=1)

    t_path = f"{save_path}/{name}_train_proto_mean"
    if not os.path.exists(t_path):
        torch.save(feat_avg, t_path)
    return feat_avg


def residual_adapt(tsa_net, mat_cov, b, num_cls, train_buffer, alpha=0.2, device="cuda", batch_size=250, max_iter=40, lr=0.1,
                   lr_cl=0.01, scale_init=10, proto_w=None, classifier_init=True):
    # Get classifier weights from Ridge Regression
    if proto_w is None:
        eye = np.eye(mat_cov.shape[0]) * alpha
        cov = mat_cov + eye
        w = scipy.linalg.solve(cov, b, sym_pos=True, overwrite_a=True).T
        w, bias = np.split(w, [-1], axis=1)
    else:
        w = proto_w
        bias = numpy.zeros(num_cls)

    classifier = torch.nn.Linear(w.shape[1], num_cls)
    if classifier_init:
        classifier.weight.data = torch.Tensor(w.astype(np.float32))
        classifier.bias.data = torch.Tensor(bias.T.astype(np.float32))
    classifier = classifier.to(device)

    # Initialise scale
    scale = torch.nn.Parameter(torch.Tensor(1))
    scale.requires_grad = True
    scale.data.fill_(scale_init)

    # Get params list for optimiser
    alpha_params = [v for k, v in tsa_net.named_parameters() if 'alpha' in k]

    params = [
        {'params': alpha_params},
        {'params': [scale], 'lr': 1.},
        {'params': list(classifier.parameters()), 'lr': lr_cl}
    ]

    # Initialise optimiser
    # TODO: Check if correct optimiser
    optimizer = torch.optim.Adadelta(params, lr=lr)
    dataloader = get_dataloader(train_buffer, batch_size, 4, shuffle=True, drop_last=True)

    # Get number of epochs
    epoch_iter = len(dataloader)
    num_epochs = math.floor(max_iter / epoch_iter)

    epoch_loss = []
    for i in range(num_epochs):
        losses = []
        for batch_x, batch_y, _ in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()

            context_features = tsa_net.embed(batch_x)
            context_features = F.normalize(context_features, p=2., dim=1)  # this seems important, acts as nonlinearity
            logits = classifier(context_features)
            loss = F.cross_entropy(logits * scale.to(device), batch_y)

            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        epoch_loss.append(np.mean(losses))

    return tsa_net, classifier, epoch_loss


def test_result(dataset, tsa_net, classifier, device="cuda", change_model_training_state=False, normalise=True):
    # Load test data and evaluate
    test_loader = get_dataloader(dataset, batch_size=256, n_workers=4, shuffle=False, drop_last=False)

    if change_model_training_state:
        training_state = tsa_net.training
        tsa_net.eval()

    ret = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            x_feat = tsa_net.embed(batch_x.to(device))
            x_feat = F.normalize(x_feat, p=2., dim=1)
            logits = classifier(x_feat)
            ret.append(logits)

    if change_model_training_state:
        tsa_net.train(training_state)

    ret = torch.cat(ret, dim=0)
    pred = torch.argmax(ret, dim=1).cpu().numpy()
    acc = accuracy_score(dataset.get_labels(), pred)
    # print(np.unique(pred))
    # print(np.unique(dataset.get_labels()))
    return acc


def save_buffered(save_path, name, dataset, x_feat, buf_size, get_index_func):
    buf_x, buf_y = fill_buffer(dataset, x_feat, buf_size, get_index_func)
    save_data = {
        "buf_input": buf_x,
        "buf_label": buf_y,
    }
    torch.save(save_data, f"{save_path}/{name}_{buf_size}")


def partial_reload(model, state_dict):
    cur_dict = model.state_dict()
    partial_dict = {}
    for k, v in state_dict.items():
        if k in cur_dict and cur_dict[k].shape == v.shape:
            partial_dict[k] = v
    print(f"number of matched tensors: {len(partial_dict)}")
    print(partial_dict.keys())
    cur_dict.update(partial_dict)
    model.load_state_dict(cur_dict)


def change_param_prefix(params, old_prefix, new_prefix):
    _len = len(old_prefix)
    names = list(params.keys())
    for name in names:
        if name.startswith(old_prefix):
            new_name = f"{new_prefix}{name[_len:]}"
            params[new_name] = params[name]
            del params[name]

    return params


def load_model(file_path):
    model = resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5)
    save_dicts = torch.load(file_path)
    model_params = save_dicts["model"]
    model_params = change_param_prefix(model_params, "backbone.", "")
    partial_reload(model, model_params)

    return model


def main_routine_mini_residual(device="cuda"):
    test_data = ImageNet(os.path.expanduser("~/workspace/metaL_data/"), default_transform, partition="test")

    num_cls = np.unique(test_data.get_labels()).shape[0]
    backbone = load_model(mimg_model_path)
    backbone.eval()
    opt = lambda x:x
    opt.tsa_alpha = True
    opt.tsa_beta = False

    model = resnet_tsa(backbone, opt)
    model.reset()
    model = model.to(device)

    buf_size = 200
    mat_cov = torch.load(f"{base_path()}/mimg_cov_mat_with_bias")
    b = torch.load(f"{base_path()}/mimg_b_with_bias")
    buffer = torch.load(f"{base_path()}/mimg_{buf_size}")
    buf_x = buffer["buf_input"]
    buf_y = buffer["buf_label"]

    dataset = "mimg"
    # feat_avg = torch.load(f"{base_path()}/{dataset}_train_proto_mean")

    print(f"Buffer Size: {len(buf_x)}")
    buf_dataset = TensorDataset(buf_x, buf_y.astype(np.long), tensor_aug_transform)

    accs = []
    for i in range(20):
        model.reset()
        backbone.eval()
        seq_model, classifier, epoch_loss = residual_adapt(model, mat_cov, b, num_cls, buf_dataset, alpha=0.001,
                                                           device=device,
                                                           batch_size=50, max_iter=300, lr=0.003, lr_cl=0.5,
                                                           scale_init=1,
                                                           proto_w=None, classifier_init=True)
        acc = test_result(test_data, seq_model, classifier, device=device)

        accs.append(acc)
        print(f"Run {i} Acc: {accs[-1]}")
        if i == 2 or i == 9 or i == 19:
            print("=" * 50)
            print(f"Avg acc: {np.mean(accs)}, std: {np.std(accs)}")
            print("=" * 50)
    print("=" * 50)
    print(f"Final avg acc: {np.mean(accs)}, std: {np.std(accs)}")
    print("=" * 50)


def prep_stat_and_buffer_mini():
    train_data = ImageNet(os.path.expanduser("~/workspace/metaL_data/"), default_transform, partition="train")
    num_cls = np.unique(train_data.get_labels()).shape[0]
    model = load_model(mimg_model_path)
    _, _, x_feat = save_ls_square_data("mimg", train_data, model.cuda())
    # save_proto_data(base_path(), "mimg", x_feat, num_cls)
    save_buffered(base_path(), "mimg", train_data, torch.Tensor(x_feat), 200, get_indices_minimum_dist_mean)
    # save_buffered(base_path(), "mimg", train_data, torch.Tensor(x_feat), 2000, get_indices_random)


if __name__ == '__main__':
    prep_stat_and_buffer_mini()
    main_routine_mini_residual()



