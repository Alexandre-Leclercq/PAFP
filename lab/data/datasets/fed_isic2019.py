import os.path
from torch.utils.data import Dataset
import albumentations
from typing import Literal, Optional
import random
from PIL import Image
import pandas as pd
import numpy as np
import torch

"""
inspired by https://github.com/owkin/FLamby/blob/main/flamby/datasets/fed_isic2019/dataset.py
"""
class FedIsic2019(Dataset):
    def __init__(self,
                 dataset_path: str,
                 center: int = 0,
                 train: bool = True,
                 pooled: bool = False,
                 normalize:Optional[Literal['imagenet', 'tanh', 'fed-isic2019']]='imagenet',
                 target: Literal['melanoma', 'center']='melanoma'
                 ):
        sz = 200
        assert pooled == (center is None)
        if train:
            if normalize is None:
                norm = albumentations.Normalize(p=0.0)
            elif normalize=='fed-isic2019': #use precomputed mean and std for each fed-isic2019 client training set
                mean = (
                    (0.5850, 0.5027, 0.4810),
                    (0.7867, 0.4538, 0.4683),
                    (0.6688, 0.5334, 0.5719),
                    (0.7177, 0.5613, 0.5702),
                    (0.6536, 0.5329, 0.4901),
                    (0.6622, 0.4619, 0.4650),
                )
                std = (
                    (0.2542, 0.2397, 0.2445),
                    (0.1957, 0.1831, 0.1932),
                    (0.1967, 0.2006, 0.2218),
                    (0.1860, 0.1864, 0.2020),
                    (0.1766, 0.1687, 0.1724),
                    (0.2553, 0.2188, 0.2360),
                )
                norm = albumentations.Normalize(mean=mean[center], std=std[center], max_pixel_value=255)
            elif normalize=='imagenet':
                norm = albumentations.Normalize()
            elif normalize=='tanh':
                norm = albumentations.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255)
            else:
                raise NotImplementedError(f'Unknown norm: {normalize}')

            self.augmentations = albumentations.Compose(
                [
                    albumentations.RandomScale(0.07),
                    albumentations.Rotate(50),
                    albumentations.RandomBrightnessContrast(0.15, 0.1),
                    albumentations.HorizontalFlip(p=0.5),
                    albumentations.Affine(shear=0.1),
                    albumentations.RandomCrop(sz, sz),
                    albumentations.CoarseDropout((1, 8), (16,16), (16,16)),
                    norm
                ]
            )
        else:
            self.augmentations = albumentations.Compose(
                [
                    albumentations.CenterCrop(sz, sz),
                    albumentations.Normalize(),
                ]
            )

        train_test_split_path = os.path.join(dataset_path, 'train_test_split.csv')
        images_path = os.path.join(dataset_path, 'ISIC_2019_Training_Input_preprocessed')
        self.center = center
        self.train_test = "train" if train else "test"
        df = pd.read_csv(train_test_split_path)

        if pooled:
            df2 = df.query("fold == '" + self.train_test + "' ").reset_index(drop=True)
        else:
            assert center in range(6)
            self.key = self.train_test + "_" + str(self.center)
            df2 = df.query("fold2 == '" + self.key + "' ").reset_index(drop=True)

        images = df2.image.tolist()
        self.image_paths: list = [
            os.path.join(images_path, image_name + ".jpg")
            for image_name in images
        ]
        if target == 'melanoma':
            self.targets: pd.Series = df2.target
        elif target == 'center':
            self.targets: pd.Series = df2.center
        else:
            raise ValueError(f'Unknown target: {target}')
        self.centers:pd.Series = df2.center

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = np.array(Image.open(image_path))
        target = self.targets.iloc[idx]

        # Image augmentations
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(target, dtype=torch.int64),
        )
