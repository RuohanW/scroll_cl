import logging

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

base_transform = transforms.Compose(
        [
         lambda x: Image.fromarray(x, mode='RGB'),
         lambda x: x.resize((224, 224), resample=Image.LANCZOS),
         transforms.ToTensor(),
         lambda x: x * 2 - 1])


class TensorCIFAR(Dataset):
    def __init__(self, cifar_data):
        super().__init__()
        if cifar_data is not None:
            data = cifar_data.data
            targets = cifar_data.targets

            self.data = torch.stack([base_transform(x) for x in data])
            self.targets = torch.Tensor(targets)

    def __getitem__(self, item):
        return self.data[item], self.targets[item]

    def __len__(self):
        return self.data.shape[0]

    def get_class(self, i):
        idx = self.targets == i
        return self.data[idx]

    def get_range(self, i, j):
        idx = torch.logical_and(self.targets >=i, self.targets<j)
        return self.data[idx], self.targets[idx]

    def get_labels(self):
        return torch.unique(self.targets)

    def load_data(self, file_path):
        state_dict = torch.load(file_path)
        self.data = state_dict["data"]
        self.targets = state_dict["targets"]

        # print(self.data.shape)

    def save_data(self, file_path):
        train_data = {
            "data": self.data,
            "targets": self.targets,
        }
        torch.save(train_data, file_path)
