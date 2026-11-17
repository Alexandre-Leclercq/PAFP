import torch
from .base_datamodule import BaseDataModule
from lab.data.datasets import FedIsic2019
from torch.utils.data import random_split, Subset
from typing import List, Literal, Optional



class FedIsic2019DataModule(BaseDataModule):
    def __init__(self,
                 dataset_path: str,
                 batch_size: int,
                 train_val_test_shuffle: List[bool],
                 train_val_split: List[float],
                 persistent_workers: bool,
                 img_size: int,
                 n_channel: int = 3,
                 n_class: int = 8,
                 n_worker: int = 1,
                 normalize: Optional[Literal['tanh', 'imagenet', 'fed-isic2019']]='imagenet',
                 reduce_mode: bool = False,
                 center: Optional[int] = None,
                 pooled: bool = True,
                 target: Literal['melanoma', 'center']='melanoma'
                 ):
        super().__init__(
            dataset_path=dataset_path,
            reduce_mode=reduce_mode,
            batch_size=batch_size,
            train_val_test_shuffle=train_val_test_shuffle,
            n_channel=n_channel,
            n_class=n_class,
            n_worker=n_worker,
            persistent_workers=persistent_workers,
            img_size=img_size
        )

        assert pooled == (center is None)

        self.train_val_split = train_val_split
        self.center = center
        self.pooled = pooled
        self.normalize = normalize

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        # make sure setup isn't called multiple time in FL
        if self._is_setup:
            return

        # Assign train/val datasets for use in dataloaders
        # if stage == "fit":
        print('Fed-ISIC2019 setup')
        p_train, p_val = self.train_val_split
        dataset_train = FedIsic2019(dataset_path=self.dataset_path, center=self.center, pooled=self.pooled, train=True,
                                    normalize=self.normalize)
        dataset_test = FedIsic2019(dataset_path=self.dataset_path, center=self.center, pooled=self.pooled, train=False,
                                   normalize=self.normalize)

        if self.reduce_mode:
            train_indices = range(100)  # only use 100 samples of original dataset
            test_indices = range(20)
        else:
            train_indices = range(len(dataset_train))
            test_indices = range(len(dataset_test))

        train_set_indices, val_set_indices = random_split(
            train_indices, [p_train, p_val], generator=torch.Generator()
        )

        self.train_set = Subset(dataset_train, train_set_indices)
        self.val_set = Subset(dataset_train, val_set_indices)
        self.test_set = Subset(dataset_test, test_indices)


        # compute the class weight
        weights = []
        targets_train_set = self.train_set.dataset.targets.iloc[list(self.train_set.indices)]
        class_distribution = {i: targets_train_set[targets_train_set == i].size for i in range(self.n_class)}
        for k, v in class_distribution.items():
            if class_distribution[k] == 0:
                weights.append(0)
            else:
                weights.append(sum(class_distribution.values()) / (self.n_class * class_distribution[k]))
        self.class_weights = torch.tensor(weights, dtype=torch.float)
        print(f'weights: {weights}')

        self._is_setup = True
