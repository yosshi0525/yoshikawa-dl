from torch.utils.data import Dataset
import numpy as np


class MyDataset(Dataset):
    def __init__(self, mnist_dataset, train=True):
        self.train = train
        self.dataset = mnist_dataset
        self.transform = mnist_dataset.transform

        if self.train:
            self.train_data = self.dataset.train_data
            self.train_labels = self.dataset.train_labels
            self.train_label_set = set(self.train_labels.numpy())
            self.label_to_indices = {
                label: np.where(self.train_labels.numpy() == label)[0]
                for label in self.train_label_set
            }
        else:
            self.test_data = self.dataset.test_data
            self.test_labels = self.dataset.test_labels
            self.test_label_set = set(self.test_labels.numpy())
            self.label_to_indices = {
                label: np.where(self.test_labels.numpy() == label)[0]
                for label in self.test_label_set
            }

            # シャッフルしないので、先にペアを決めておく
            positive_pairs = [
                [
                    i,
                    np.random.choice(self.label_to_indices[self.test_labels[i].item()]),
                    1,
                ]
                for i in range(0, len(self.test_data), 2)
            ]

            negative_pairs = [
                [
                    i,
                    np.random.choice(
                        self.label_to_indices[
                            np.random.choice(
                                list(
                                    self.test_label_set
                                    - set([self.test_labels[i].item()])
                                )
                            )
                        ]
                    ),
                    0,
                ]
                for i in range(1, len(self.test_data), 2)
            ]

            self.test_pairs = positive_pairs + negative_pairs
