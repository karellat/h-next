import logging
import os
import warnings
from abc import ABC
from typing import Any, Optional, Callable, Dict, Tuple
import torch
from PIL import Image
import numpy as np

from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_url, download_and_extract_archive
from torchvision import transforms


class MnistRotTestVisionDataset(VisionDataset):
    # NOTE: To be connected to hugging faces (double-blind review)
    """
    mnist-rot-test pytorch faster reimplementation of ../custom_datasets.py hugging faces dataset
    Based on:
    https://pytorch.org/vision/main/_modules/torchvision/datasets/mnist.html#MNIST
    """
    _VERSION = "v3"
    _URL = "https://owncloud.cesnet.cz/index.php/s/q2BYzg8Uzcc8O4g/download"
    _MD5SUM = "f1789ad4651263d3667a17fb651acc95"
    _FILE_NAME = f"mnist-rot-test-uint8-{_VERSION}.npz"

    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __len__(self) -> int:
        return len(self.data)

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def __init__(self,
                 root: str,
                 train: Optional[bool] = None,
                 split: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = True):
        self.processed_file = None
        assert (train is None) or (split is None), "Use only one parameter split or train"
        super().__init__(root=root,
                         transform=transform,
                         target_transform=target_transform)

        if train is None:
            assert split in ['train', 'test'], "Supporting two options for split. (train, test)"
            self.train = (split == 'train')
            self.split = split
        else:
            self.train = train
            self.split = 'train' if train else 'test'

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.data, self.targets = self._load_data()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _load_data(self):
        with np.load(self._downloaded_file) as file:
            data = torch.from_numpy(file[f"{self.split}_images"])
            targets = torch.from_numpy(file[f"{self.split}_labels"])
        return data, targets

    @property
    def _downloaded_file(self):
        return os.path.join(self.root, self._FILE_NAME)

    def _check_exists(self) -> bool:
        return check_integrity(self._downloaded_file, md5=self._MD5SUM)

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist already."""

        if self._check_exists():
            return
        logging.debug(f"Downloading {self._URL}")
        download_url(self._URL,
                     root=self.root,
                     filename=self._FILE_NAME,
                     md5=self._MD5SUM)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"


class Cifar10RotTestVisionDataset(VisionDataset):
    # NOTE: To be connected to hugging faces (double-blind review)
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Based on https://pytorch.org/vision/main/_modules/torchvision/datasets/cifar.html#CIFAR10
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    _VERSION = "v0"
    _URL = "https://owncloud.cesnet.cz/index.php/s/Denv319G7GwulEv/download"
    _MD5SUM = "ba0336e0a4ffd97dc4c01d5e0eb15d49"
    _FILE_NAME = f"cifar10-rot-test-{_VERSION}.npz"

    def __init__(
            self,
            root: str,
            train: bool = None,
            split: Optional[str] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = True,
    ) -> None:
        assert (train is None) or (split is None), "Use only one parameter split or train"
        if transform is None:
            warnings.warn("No transform used, please convert to Tensor.")
            transform = transforms.ToTensor()
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        if train is None:
            assert split in ['train', 'test'], "Supporting two options for split. (train, test)"
            self.train = (split == 'train')
            self.split = split
        else:
            self.train = train
            self.split = 'train' if train else 'test'

        self.data, self.targets, self.classes = self._load_data()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def _downloaded_file(self):
        return os.path.join(self.root, self._FILE_NAME)

    def _check_integrity(self):
        return check_integrity(self._downloaded_file, md5=self._MD5SUM)

    def _check_exists(self) -> bool:
        return os.path.exists(self._downloaded_file)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"

    def download(self) -> None:
        """Download the CIFAR data if it doesn't exist already."""

        if self._check_exists():
            return
        logging.debug(f"Downloading {self._URL}")
        download_url(self._URL,
                     root=self.root,
                     filename=self._FILE_NAME,
                     md5=self._MD5SUM)

    def _load_data(self):
        with np.load(self._downloaded_file) as file:
            data = file[f"{self.split}_images"].transpose((0, 2, 3, 1))
            targets = file[f"{self.split}_labels"]
            classes = file['classes']
        return data, targets, classes


class RotatedMnistVisionDataset(VisionDataset, ABC):
    _URL = "http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip"
    _MD5SUM = "0f9a947ff3d30e95cd685462cbf3b847"
    _ZIP_FILE = "mnist_rotation_new.zip"
    _TEST_FILE = "mnist_all_rotation_normalized_float_test.amat"
    _TRAIN_FILE = "mnist_all_rotation_normalized_float_train_valid.amat"

    def __init__(self,
                 root: str,
                 train: Optional[bool] = None,
                 split: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = True):
        self.processed_file = None
        assert (train is None) or (split is None), "Use only one parameter split or train"
        super().__init__(root=root,
                         transform=transform,
                         target_transform=target_transform)

        if train is None:
            assert split in ['train', 'test'], "Supporting two options for split. (train, test)"
            self.train = (split == 'train')
            self.split = split
        else:
            self.train = train
            self.split = 'train' if train else 'test'

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.data, self.targets = self._load_data()

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist already."""

        if self._check_exists():
            return
        logging.debug(f"Downloading {self._URL}")
        download_and_extract_archive(self._URL,
                                     download_root=self.root,
                                     md5=self._MD5SUM)

    def _check_exists(self) -> bool:
        return (
                check_integrity(os.path.join(self.root, self._ZIP_FILE), md5=self._MD5SUM)
                and
                os.path.exists(self._downloaded_file)
        )

    @property
    def _downloaded_file(self):
        if self.train:
            return os.path.join(self.root, self._TRAIN_FILE)
        else:
            return os.path.join(self.root, self._TEST_FILE)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"

    def _load_data(self):
        raw = np.loadtxt(self._downloaded_file)
        data = raw[:, :-1].reshape(-1, 28, 28)
        data = (data * 255.0).astype(np.uint8)
        targets = (raw[:, -1]).astype(np.uint8)
        return data, targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
    def __len__(self) -> int:
        return self.data.shape[0]

