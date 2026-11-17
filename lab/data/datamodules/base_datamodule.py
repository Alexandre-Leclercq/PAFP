import os
from abc import ABC, abstractmethod
from torch.utils.data import Subset, Dataset, DataLoader
import torch
import pandas as pd
import lightning as L


class BaseDataModule(L.LightningDataModule):
    def __init__(self,
                 dataset_path: str,
                 reduce_mode: bool,
                 batch_size: int,
                 train_val_test_shuffle: list[bool],
                 persistent_workers: bool,
                 img_size: int,
                 n_channel: int,
                 n_class: int,
                 n_worker: int = 1):
        super().__init__()
        self.batch_size = batch_size
        self.img_size = img_size
        self.n_channel = n_channel
        self.n_class = n_class
        self.n_worker = n_worker
        self.dataset_path = dataset_path
        self.reduce_mode = reduce_mode
        self.train_val_test_shuffle = train_val_test_shuffle
        self.persistent_workers = persistent_workers
        self._is_setup: bool = False
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def get_n_class(self):
        return self.n_class

    def get_train_class_distribution(self) -> torch.Tensor:
        if self.train_set is None:
            return torch.tensor([])
        return compute_class_distribution_set(self.train_set, self.get_n_class())

    def get_val_class_distribution(self) -> torch.Tensor:
        if self.val_set is None:
            return torch.tensor([])
        return compute_class_distribution_set(self.val_set, self.get_n_class())

    def get_test_class_distribution(self) -> torch.Tensor:
        if self.test_set is None:
            return torch.tensor([])
        return compute_class_distribution_set(self.test_set, self.get_n_class())

    def get_len_train_set(self):
        if self.train_set is None:
            return 0
        else:
            return len(self.train_set)

    def get_len_val_set(self):
        if self.val_set is None:
            return 0
        else:
            return len(self.val_set)

    def get_len_test_set(self):
        if self.test_set is None:
            return 0
        else:
            return len(self.test_set)

    def get_n_channel(self):
        return self.n_channel

    def get_img_size(self):
        return self.img_size

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_n_worker(self, n_worker):
        self.n_worker = n_worker

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            shuffle=self.train_val_test_shuffle[0],
            num_workers=self.n_worker,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=self.persistent_workers
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size,
            shuffle=self.train_val_test_shuffle[1],
            num_workers=1,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=self.persistent_workers
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.batch_size,
            shuffle=self.train_val_test_shuffle[2],
            num_workers=self.n_worker,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=self.persistent_workers
        )


# Utility function #
def compute_class_distribution_set(data_set: Dataset | Subset, n_class: int) -> torch.Tensor:
    assert data_set is not None
    if hasattr(data_set, 'targets'):
        targets = data_set.targets
    elif isinstance(data_set, Subset) and hasattr(data_set.dataset, 'targets'):
        # print('is a subset of the dataset')
        full_dataset_targets = data_set.dataset.targets  # get the targets from the original dataset (not the subset)
        if isinstance(full_dataset_targets, pd.Series):
            targets = full_dataset_targets.iloc[list(data_set.indices)]
        else:
            targets = full_dataset_targets[data_set.indices]  # filter using the indices of the subset
    elif isinstance(data_set.dataset, Subset) and hasattr(data_set.dataset.dataset, 'targets'):
        # print('is a subset of the dataset')
        full_dataset_targets = data_set.dataset.dataset.targets  # get the targets from the original dataset (not the subset)
        targets = full_dataset_targets[data_set.data_set.indices]  # filter using the indices of the subset
    else:
        raise AttributeError("The dataset don't have a targets attribute.")
    # print(f'distrib: {targets.unique(return_counts=True)}')
    # I don't use the unique method that allow to count unique_values,
    # because I need to count 0 for the class not represented
    distribution = torch.zeros(n_class, dtype=torch.int)
    for i in range(n_class):
        distribution[i] = (targets == i).sum().item()
    return distribution
