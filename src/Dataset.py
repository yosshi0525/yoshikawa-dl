from torch.utils.data import Dataset
from torch import Tensor
import numpy as np


class MyDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        return self.data[index], self.target[index]
