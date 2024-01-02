from torch.utils.data.dataset import Dataset
import numpy as np
import os, pickle

from torchvision import transforms

from PIL import Image
imgnet_mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
imgnet_std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
# imgnet_mean = [0.485, 0.456, 0.406]
# imgnet_std = [0.229, 0.224, 0.225]

normalize = transforms.Normalize(mean=imgnet_mean, std=imgnet_std)
normalize01 = lambda x: x*2 - 1

default_transform = transforms.Compose(
    [lambda x: Image.fromarray(x), transforms.ToTensor(), normalize]
)


class ImageNet(Dataset):
    def __init__(self, data_root, transform, split_ratio=0.8, partition="train"):
        super().__init__()
        self.transform = transform

        self.data, self.labels = self._load_from_file(data_root, partition, split_ratio)
        self.n_cls = np.max(self.labels) + 1
        self._real_len = len(self.data)

    def __getitem__(self, item):
        img = self.transform(self.data[item])
        target = self.labels[item]

        return img, target

    def __len__(self):
        return self._real_len

    def get_labels(self):
        return self.labels

    def _load_from_file(self, data_root, partition, split_ratio):
        # file_path = f"miniImageNet/miniImageNet_category_split_test.pickle"
        file_path = f"miniImageNet/{partition}_data.pickle"
        with open(os.path.join(data_root, file_path), "rb") as f:
            data = pickle.load(f, encoding="latin1")
            labels = np.array(data["labels"])
            data = np.array(data["data"]).astype("uint8")

        idx_sorted = np.argsort(labels)
        data = data[idx_sorted]
        labels = labels[idx_sorted]
        uniq_labels = np.unique(labels)
        d_shape = data.shape
        # mat_data = data.reshape(uniq_labels.shape[0], -1, *d_shape[1:])
        # mat_lab = labels.reshape(uniq_labels.shape[0], -1)
        # num_per_cls = mat_data.shape[1]
        # split = int(round(num_per_cls * split_ratio))
        # if partition == "train":
        #     data = mat_data[:, :split]
        #     labels = mat_lab[:, :split]
        # else:
        #     data = mat_data[:, split:]
        #     labels = mat_lab[:, split:]

        data = data.reshape(uniq_labels.shape[0], -1, *d_shape[1:])
        labels = labels.reshape(uniq_labels.shape[0], -1)

        data = data.reshape(-1, *d_shape[1:])
        labels = labels.reshape(-1)

        # print(np.unique(labels))
        # if partition == "test":
        #     data = data[labels >= 64]
        #     labels = labels[labels >= 64]
        # print(data.shape)
        # print(labels.shape)

        return data, labels