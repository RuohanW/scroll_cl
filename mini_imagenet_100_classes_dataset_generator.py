import numpy as np
import os
import pickle

data_root = os.path.expanduser("~/workspace/metaL_data/")
split_ratio = 0.8


def _load_from_file(data_root, partition, split_ratio):
    if partition == "train":
        file_path = f"miniImageNet/miniImageNet_category_split_train_phase_train.pickle"
        expected_classes = 64
        expected_data = 38400
    if partition == "val":
        file_path = f"miniImageNet/miniImageNet_category_split_val.pickle"
        expected_classes = 16
        expected_data = 9600
    if partition == "test":
        file_path = f"miniImageNet/miniImageNet_category_split_test.pickle"
        expected_classes = 20
        expected_data = 12000
    with open(os.path.join(data_root, file_path), "rb") as f:
        data = pickle.load(f, encoding="latin1")
        labels = np.array(data["labels"])
        data = np.array(data["data"]).astype("uint8")

    idx_sorted = np.argsort(labels)
    data = data[idx_sorted]
    labels = labels[idx_sorted]
    uniq_labels = np.unique(labels)
    d_shape = data.shape

    assert len(uniq_labels) == expected_classes
    assert len(data) == expected_data
    assert len(labels) == expected_data

    mat_data = data.reshape(uniq_labels.shape[0], -1, *d_shape[1:])
    mat_lab = labels.reshape(uniq_labels.shape[0], -1)
    num_per_cls = mat_data.shape[1]
    split = int(round(num_per_cls * split_ratio))
    data_train = mat_data[:, :split]
    labels_train = mat_lab[:, :split]
    data_test = mat_data[:, split:]
    labels_test = mat_lab[:, split:]

    data_train = data_train.reshape(-1, *d_shape[1:])
    labels_train = labels_train.reshape(-1)
    data_test = data_test.reshape(-1, *d_shape[1:])
    labels_test = labels_test.reshape(-1)

    return data_train, labels_train, data_test, labels_test


data_metatrain_train, labels_metatrain_train, data_metatrain_test, labels_metatrain_test = _load_from_file(data_root, "train", split_ratio)
data_metaval_train, labels_metaval_train, data_metaval_test, labels_metaval_test = _load_from_file(data_root, "val", split_ratio)
data_metatest_train, labels_metatest_train, data_metatest_test, labels_metatest_test = _load_from_file(data_root, "test", split_ratio)

print("Done")

print(data_metatrain_train.shape)
print(data_metatrain_test.shape)

data_pretraining_train = data_metatrain_train
labels_pretraining_train = labels_metatrain_train
data_pretraining_val = data_metaval_train
labels_pretraining_val = labels_metaval_train
data_scroll_train = np.concatenate([data_metaval_train, data_metatest_train])
labels_scroll_train = np.concatenate([labels_metaval_train, labels_metatest_train])
data_train = np.concatenate([data_metatrain_train, data_metaval_train, data_metatest_train])
labels_train = np.concatenate([labels_metatrain_train, labels_metaval_train, labels_metatest_train])
data_test = np.concatenate([data_metatrain_test, data_metaval_test, data_metatest_test])
labels_test = np.concatenate([labels_metatrain_test, labels_metaval_test, labels_metatest_test])

print(data_pretraining_train.shape)
print(labels_pretraining_train.shape)

pretraining_data = dict()
pretraining_data["data"] = data_pretraining_train
pretraining_data["labels"] = labels_pretraining_train

file_path = "miniImageNet/pretraining_data.pickle"
with open(os.path.join(data_root, file_path), "wb") as f:
    pickle.dump(pretraining_data, f)

print(data_pretraining_val.shape)
print(labels_pretraining_val.shape)

pretraining_data_val = dict()
pretraining_data_val["data"] = data_pretraining_val
pretraining_data_val["labels"] = labels_pretraining_val

file_path = "miniImageNet/pretraining_data_val.pickle"
with open(os.path.join(data_root, file_path), "wb") as f:
    pickle.dump(pretraining_data_val, f)

print(data_scroll_train.shape)
print(labels_scroll_train.shape)

scroll_data = dict()
scroll_data["data"] = data_scroll_train
scroll_data["labels"] = labels_scroll_train

file_path = "miniImageNet/scroll_data.pickle"
with open(os.path.join(data_root, file_path), "wb") as f:
    pickle.dump(scroll_data, f)


print(data_train.shape)
print(labels_train.shape)

train_data = dict()
train_data["data"] = data_train
train_data["labels"] = labels_train

file_path = "miniImageNet/train_data.pickle"
with open(os.path.join(data_root, file_path), "wb") as f:
    pickle.dump(train_data, f)


print(data_test.shape)
print(labels_test.shape)

test_data = dict()
test_data["data"] = data_test
test_data["labels"] = labels_test

file_path = "miniImageNet/test_data.pickle"
with open(os.path.join(data_root, file_path), "wb") as f:
    pickle.dump(test_data, f)
