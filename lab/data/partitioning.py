from torchvision.datasets import MNIST, FashionMNIST, CIFAR100, SVHN
import numpy as np
from torchvision.transforms import Compose, Lambda
import torch
from hydra.utils import instantiate
from torch.utils.data import random_split
from lightning import LightningDataModule


def partition_data(
        dataset: str,
        dataset_dir: str,
        method: str,
        n_partitions: int,
        reduce_mode: bool,
        config_datamodule,
        partitioning_parameters) -> list[LightningDataModule]:
    """
    the goal of this function is to provide a harmonize every dataset behavior to the partition function.
    Thus, for each dataset we map the expected parameters to the dataset settings.
    """
    assert method in ['IID', 'LDA_LABEL', 'LDA_QUANTITY', 'FEATURE_NOISE']  # list supported method

    if dataset == 'MNIST':  # torchvision dataset
        print('partitioning MNIST dataset')
        train_dataset = MNIST(root=dataset_dir, train=True, download=True)
        train_indexes = np.arange(len(train_dataset))
        # dataset.classes contain str value that doesn't map the value in the targets. Thus, we manually set classes list
        # for example : '0 - zero' for the class 0, etc.
        train_classes = np.arange(len(train_dataset.classes))
        train_targets = train_dataset.targets.numpy()
        train_partitions, transforms = partition_dataset(train_indexes, train_classes, train_targets, n_partitions,
                                                         method, partitioning_parameters)
        test_dataset = MNIST(root=dataset_dir, train=False, download=True)
        test_indexes = np.arange(len(test_dataset))
        # dataset.classes contain str value that doesn't map the value in the targets. Thus, we manually set classes list
        test_classes = np.arange(len(train_dataset.classes))
        test_targets = test_dataset.targets.numpy()
        test_partitions, _ = partition_dataset(test_indexes, test_classes, test_targets, n_partitions, method,
                                                   partitioning_parameters)

        return [ instantiate(config_datamodule, train_indices=train_partition, test_indices=test_partition,
                             reduce_mode=reduce_mode, feature_transform=transforms[i])
                 for i, (train_partition, test_partition) in enumerate(zip(train_partitions, test_partitions))]
    elif dataset == 'FMNIST':
        print('partitioning FMNIST dataset')

        #   Train SET
        train_dataset = FashionMNIST(root=dataset_dir, train=True, download=True)
        train_indexes = np.arange(len(train_dataset))
        # dataset.classes contain str value that doesn't map the value in the targets. Thus, we manually set classes list
        # for example : '0 - zero' for the class 0, etc.
        train_classes = np.arange(len(train_dataset.classes))
        train_targets = train_dataset.targets.numpy()
        train_partitions, transforms = partition_dataset(train_indexes, train_classes, train_targets, n_partitions,
                                                         method, partitioning_parameters)
        #    Test SET
        test_dataset = FashionMNIST(root=dataset_dir, train=False, download=True)
        test_indexes = np.arange(len(test_dataset))
        # dataset.classes contain str value that doesn't map the value in the targets. Thus, we manually set classes list
        test_classes = np.arange(len(train_dataset.classes))
        test_targets = test_dataset.targets.numpy()
        test_partitions, _ = partition_dataset(test_indexes, test_classes, test_targets, n_partitions, method,
                                                   partitioning_parameters)

        return [ instantiate(config_datamodule, train_indices=train_partition, test_indices=test_partition,
                             reduce_mode=reduce_mode, feature_transform=transforms[i])
                 for i, (train_partition, test_partition) in enumerate(zip(train_partitions, test_partitions))]
    elif dataset == 'CIFAR10':
        print('partitioning CIFAR10 dataset')
        indexes = None
        classes = None
        targets = None
        raise NotImplementedError
    elif dataset == 'CIFAR100':
        print('partitioning CIFAR100 dataset')
        full_dataset = CIFAR100(root=dataset_dir, train=True, download=True)
        indexes = np.arange(len(full_dataset))
        classes = None
        targets = None
        raise NotImplementedError
    elif dataset == 'SVHN':
        print('partitioning SVHN dataset')
        full_dataset = SVHN(root=dataset_dir, split='train', download=True)
        indexes = np.arange(len(full_dataset))
        classes = None
        targets = None
        raise NotImplementedError

    # dataset with manifest
    elif dataset == 'UPENN_GBM':
        print('partitioning UPENN-GBM dataset')
        # datamodule: UPennGBMDataModule = instantiate(config_datamodule)
        # datamodule.manifest
        indexes = None
        classes = None
        targets = None
        raise NotImplementedError
    elif dataset == 'LUMIERE':
        print('partitioning LUMIERE dataset')
        indexes = None
        classes = None
        targets = None
        raise NotImplementedError
    else:
        raise ValueError(f'unknown dataset: {dataset}')

    # partitions = partition_function(full_dataset, method, n_partitions, partitioning_parameters)

    # return [instantiate(config_datamodule, indices=partition, reduce_mode=reduce_mode) for partition in partitions]


def partition_dataset(
        indexes: np.array,
        classes: np.array,
        targets: np.array,
        n_partitions: int,
        method: str,
        partitioning_parameters
) -> tuple[list[list[int]], list]:
    """

    :param indexes: dataset indexes
    :param classes: list of unique classes
    :param targets: list of all classes targets associate to all indexes
    :return:
    """
    assert len(indexes) == len(targets)
    transforms = [Compose([])] * n_partitions  # init transforms list with empty transformations
    print(f'{method} partitioning')

    if method == 'IID':
        # get the indices of the dataset and shuffle them
        dataset_index = torch.randperm(len(indexes)).tolist()
        split_proportion = [1 / n_partitions] * n_partitions
        partitions = random_split(dataset_index, split_proportion)
        partitions = [list(partition) for partition in partitions]
    elif method == 'LDA_LABEL':
        # based on the implementation of NIID-Bench : https://github.com/Xtra-Computing/NIID-Bench
        min_size = 0  # the size of the smallest partitions
        min_require_size = 10  # a partition can't be less than 10 samples

        N = len(indexes)

        partitions: list[list] = [[] for _ in range(n_partitions)]
        while min_size < min_require_size:
            partitions: list[list] = [[] for _ in range(n_partitions)]
            for k in classes:
                # print(f'targets: {targets}')
                # print(f'k: {k}')
                # print(f'type(k): {type(k)}')
                # print(f'np.where(targets == k): {np.where(targets == k)}')
                k_class_index = np.where(targets == k)[0]
                # print(f'k_class_index: {k_class_index}')
                # return
                np.random.shuffle(k_class_index)
                k_proportion = np.random.dirichlet(np.repeat(partitioning_parameters['alpha'], n_partitions))
                #print(f"class {k} proportion for each client: {k_proportion}")
                k_proportion = np.array([p * (len(partition_j) < N/n_partitions) for p, partition_j in zip(k_proportion, partitions)])
                k_proportion = k_proportion / k_proportion.sum()
                print(f"class {k}: {k_proportion}")
                k_proportion = (np.cumsum(k_proportion) * len(k_class_index)).astype(int)[:-1]
                # print(f"class {k} split threeshold: {k_proportion}")
                # print(np.split(k_class_index, k_proportion))
                partitions = [partition_j + idx.tolist() for partition_j, idx in zip(partitions, np.split(k_class_index, k_proportion))]
                min_size = min([len(partition_j) for partition_j in partitions])
            # print()

        # assert distribution stay the same
        for j, partition_j in enumerate(partitions):
            unique_values, count_values = np.unique(targets[partition_j], return_counts=True)
            print(f'partition {j}: ', end='')
            for k in classes:
                print(f'{k}: {np.where(targets[partition_j] == k)[0].size}\t', end='')
            print(f'\nn = {len(partition_j)}\n\n')

        for j in range(n_partitions):
            np.random.shuffle(partitions[j]) # shuffle all the samples of the partition

    elif method == 'LDA_QUANTITY':
        pass  # should be implemented if needed
    elif method == 'FEATURE_NOISE':
        partitions, _ = partition_dataset(indexes, classes, targets, n_partitions,'IID', partitioning_parameters)

        def create_noise_transform(partition_index, total_partitions):
            def transform_ftc(x):
                noise_factor = partitioning_parameters['scale'] * partition_index / total_partitions
                return x + noise_factor * torch.randn_like(x)
            return transform_ftc

        for i in range(len(partitions)):
            print(f'partition {i}: sigma = {partitioning_parameters["scale"] * i / n_partitions}')
            noise_transform_i = create_noise_transform(i, n_partitions)
            transforms[i] = Compose([
                # randn is N(0,I). Thus, i x rand --> N(0, i x I)
                Lambda(noise_transform_i),
            ])
        return partitions, transforms
    else:
        raise ValueError(f'unknown method: {method}')
    return partitions, transforms
