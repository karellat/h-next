import os.path
import sys
from abc import ABC
from typing import Optional
import torchvision.datasets
from loguru import logger
from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from custom_datasets import MnistRotTestVisionDataset, Cifar10RotTestVisionDataset, RotatedMnistVisionDataset

TEST_INVARIANCE_KEY = "test_90"

class MnistRotTest(LightningDataModule, ABC):
    def __init__(self,
                 data_dir: str = "./data",
                 pad: int = 10,
                 batch_size: int = 32,
                 test_batch_size: int = 256,
                 upscale=False,
                 scale_factor: int = 2,
                 scale_mode="BILINEAR",
                 limit_train_samples=None,
                 normalize=False,
                 num_workers=12):
        assert os.path.exists(data_dir), f"Dataset folder \"{data_dir}\" not found."
        super().__init__()
        assert hasattr(InterpolationMode, scale_mode)
        scale_mode = getattr(InterpolationMode, scale_mode)
        self.save_hyperparameters(ignore=['input_shape',
                                          'data_dir',
                                          'num_workers',
                                          'input_shape',
                                          'test_batch_size'])
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.valid_ds = None  # Multiple checking multiple angles
        self.test_ds = None
        self.train_ds = None

        # Splitting indices
        indices = list(range(60000))
        self.valid_indices = indices[:10000]
        self.train_indices = indices[10000:]

        # Limit train samples
        if limit_train_samples is not None:
            assert limit_train_samples <= len(self.train_indices), f"Limit is higher than train size. {limit_train_samples}"
            assert limit_train_samples > 0, f"Limit must be higher than zero."
            self.train_indices = self.train_indices[:limit_train_samples]

        self._input_shape = [batch_size, 1, 32 + 2 * pad, 32 + 2 * pad]
        # Transformations
        _transforms = [
            transforms.ToTensor(),
            transforms.Pad(padding=pad, fill=0, padding_mode='constant')
        ]
        if normalize:
            _transforms.append(transforms.Normalize((0.1307,), (0.3081,)))

        if upscale:
            self._input_shape[2] *= scale_factor
            self._input_shape[3] *= scale_factor
            upscale_transforms = [transforms.Resize(size=self._input_shape[2],
                                                    interpolation=scale_mode)]
        else:
            upscale_transforms = []

        self.base_transforms = transforms.Compose(_transforms + upscale_transforms)
        self.rotation_transforms = {
            '0': self.base_transforms,
            '90': transforms.Compose(
                [*_transforms,
                 transforms.RandomRotation(degrees=(90, 90), interpolation=InterpolationMode.BILINEAR),
                 *upscale_transforms]
            ),
            '45': transforms.Compose(
                [*_transforms,
                 transforms.RandomRotation(degrees=(45, 45), interpolation=InterpolationMode.BILINEAR),
                 *upscale_transforms]),
            'rd': transforms.Compose(
                [*_transforms,
                 transforms.RandomRotation(degrees=(0, 359), interpolation=InterpolationMode.BILINEAR),
                 *upscale_transforms]
            )
        }

        self.num_workers = num_workers

    @property
    def sample_input_shape(self):
        return self._input_shape

    @property
    def _dataset(self):
        return MnistRotTestVisionDataset

    def prepare_data(self):
        self._dataset(root=self.data_dir,
                      train=True,
                      download=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds,
                          batch_size=self.test_batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)

    def setup(self, stage: Optional[str]):
        self.train_ds = self._dataset(root=self.data_dir,
                                      split='train',
                                      transform=self.base_transforms)
        # Test/Valid datasets
        self.valid_ds = {}
        for k, _transforms in self.rotation_transforms.items():
            self.valid_ds[k] = self._dataset(root=self.data_dir,
                                             split='train',
                                             transform=_transforms)
        self.test_ds = self._dataset(root=self.data_dir,
                                     split='test',
                                     transform=self.base_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          sampler=SubsetRandomSampler(self.train_indices),
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)

    def val_dataloader(self):
        loaders = {}
        for k, ds in self.valid_ds.items():
            ds_name = f"val_{k}" if k != '0' else 'val'
            loaders[ds_name] = DataLoader(ds,
                                          sampler=SubsetRandomSampler(self.valid_indices),
                                          batch_size=self.test_batch_size,
                                          num_workers=self.num_workers,
                                          shuffle=False),
        return CombinedLoader(loaders, mode="max_size_cycle")


class Cifar10RotTest(LightningDataModule, ABC):
    # TODO: Merge with mnist-rot-test
    def __init__(self,
                 data_dir: str = "./data",
                 pad: int = 0,
                 batch_size: int = 32,
                 test_batch_size: int = 256,
                 num_workers=12):
        assert os.path.exists(data_dir), f"Dataset folder \"{data_dir}\" not found."
        super().__init__()
        self.save_hyperparameters(ignore=['input_shape',
                                          'data_dir',
                                          'num_workers',
                                          'input_shape',
                                          'test_batch_size'])
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.valid_ds = None  # Multiple checking multiple angles
        self.test_ds = None
        self.train_ds = None

        # Splitting indices
        indices = list(range(50000))
        self.valid_indices = indices[:8000]
        self.train_indices = indices[8000:]

        # Transformations
        _transforms = [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self._input_shape = [batch_size, 3, 32 + 2 * pad, 32 + 2 * pad]

        self.base_transforms = transforms.Compose(_transforms)
        self.rotation_transforms = {
            '0': self.base_transforms,
            '90': transforms.Compose([
                transforms.RandomRotation(degrees=(90, 90), interpolation=InterpolationMode.BILINEAR),
                *_transforms]),
            '45': transforms.Compose([
                transforms.RandomRotation(degrees=(45, 45), interpolation=InterpolationMode.BILINEAR),
                *_transforms]),
            'rd': transforms.Compose([
                transforms.RandomRotation(degrees=(0, 359), interpolation=InterpolationMode.BILINEAR),
                *_transforms])
        }

        self.num_workers = num_workers

    @property
    def sample_input_shape(self):
        return self._input_shape

    @property
    def _dataset(self):
        return Cifar10RotTestVisionDataset

    def prepare_data(self):
        self._dataset(root=self.data_dir,
                      train=True,
                      transform=self.base_transforms,
                      download=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds,
                          batch_size=self.test_batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)

    def setup(self, stage: Optional[str]):
        self.train_ds = self._dataset(root=self.data_dir,
                                      split='train',
                                      download=True,
                                      transform=self.base_transforms)
        # Test/Valid datasets
        self.valid_ds = {}
        for k, _transforms in self.rotation_transforms.items():
            self.valid_ds[k] = self._dataset(root=self.data_dir,
                                             split='train',
                                             transform=_transforms)
        self.test_ds = self._dataset(root=self.data_dir,
                                     split='test',
                                     transform=self.base_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          sampler=SubsetRandomSampler(self.train_indices),
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)

    def val_dataloader(self):
        loaders = {}
        for k, ds in self.valid_ds.items():
            ds_name = f"val_{k}" if k != '0' else 'val'
            loaders[ds_name] = DataLoader(ds,
                                          sampler=SubsetRandomSampler(self.valid_indices),
                                          batch_size=self.test_batch_size,
                                          num_workers=self.num_workers,
                                          shuffle=False),
        return CombinedLoader(loaders, mode="max_size_cycle")


class RotatedMnist(LightningDataModule, ABC):

    def __init__(self,
                 data_dir: str = "./data",
                 pad: int = 10,
                 batch_size: int = 32,
                 test_batch_size: int = 256,
                 upscale=False,
                 scale_factor: int = 2,
                 scale_mode="BILINEAR",
                 normalize=False,
                 num_workers=12):
        assert os.path.exists(data_dir), f"Dataset folder \"{data_dir}\" not found."
        super().__init__()
        assert hasattr(InterpolationMode, scale_mode)
        scale_mode = getattr(InterpolationMode, scale_mode)
        self.save_hyperparameters(ignore=['input_shape',
                                          'data_dir',
                                          'num_workers',
                                          'input_shape',
                                          'test_batch_size'])
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.valid_ds = None  # Multiple checking multiple angles
        self.test_ds = None
        self.train_ds = None

        # Splitting indices
        indices = list(range(12000))
        self.valid_indices = indices[:2000]
        self.train_indices = indices[2000:]

        self._input_shape = [batch_size, 1, 28 + 2 * pad, 28 + 2 * pad]
        # Transformations
        _transforms = [
            transforms.ToTensor(),
            transforms.Pad(padding=pad, fill=0, padding_mode='constant')
        ]
        if normalize:
            _transforms.append(transforms.Normalize((0.1307,), (0.3081,)))

        if upscale:
            self._input_shape[2] *= scale_factor
            self._input_shape[3] *= scale_factor
            upscale_transforms = [transforms.Resize(size=self._input_shape[2],
                                                    interpolation=scale_mode)]
        else:
            upscale_transforms = []

        self.base_transforms = transforms.Compose(_transforms + upscale_transforms)
        self.rotation_transforms = {
            '0': self.base_transforms,
            '90': transforms.Compose([*_transforms,
                                      transforms.RandomRotation(degrees=(90, 90),
                                                                interpolation=InterpolationMode.BILINEAR),
                                      *upscale_transforms]),
            '45': transforms.Compose([*_transforms,
                                      transforms.RandomRotation(degrees=(45, 45),
                                                                interpolation=InterpolationMode.BILINEAR),
                                      *upscale_transforms]),
            'rd': transforms.Compose([*_transforms,
                                      transforms.RandomRotation(degrees=(0, 359),
                                                                interpolation=InterpolationMode.BILINEAR),
                                      *upscale_transforms])
        }

        self.num_workers = num_workers

    @property
    def sample_input_shape(self):
        return self._input_shape

    @property
    def _dataset(self):
        return RotatedMnistVisionDataset

    def prepare_data(self):
        self._dataset(root=self.data_dir,
                      train=True,
                      download=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds,
                          batch_size=self.test_batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)

    def setup(self, stage: Optional[str]):
        self.train_ds = self._dataset(root=self.data_dir,
                                      split='train',
                                      transform=self.base_transforms)
        # Test/Valid datasets
        self.valid_ds = {}
        for k, _transforms in self.rotation_transforms.items():
            self.valid_ds[k] = self._dataset(root=self.data_dir,
                                             split='train',
                                             transform=_transforms)
        self.test_ds = self._dataset(root=self.data_dir,
                                     split='test',
                                     transform=self.base_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          sampler=SubsetRandomSampler(self.train_indices),
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)

    def val_dataloader(self):
        loaders = {}
        for k, ds in self.valid_ds.items():
            ds_name = f"val_{k}" if k != '0' else 'val'
            loaders[ds_name] = DataLoader(ds,
                                          sampler=SubsetRandomSampler(self.valid_indices),
                                          batch_size=self.test_batch_size,
                                          num_workers=self.num_workers,
                                          shuffle=False),
        return CombinedLoader(loaders, mode="max_size_cycle")


def get_datamodule(dataset_name: str):
    if hasattr(sys.modules[__name__], dataset_name):
        return getattr(sys.modules[__name__], dataset_name)
    elif dataset_name == "mnist-rot-test":
        return MnistRotTest
    elif dataset_name == "cifar10-rot-test":
        return Cifar10RotTest
    elif dataset_name == "rotated-mnist":
        return RotatedMnist
    else:
        raise RuntimeError(f"Unknown dataset")


def _get_samples(degree=0, interpolation=InterpolationMode.BILINEAR, padding=2, up_size=64, batch_size=8):
    _transforms = [
        transforms.ToTensor(),
        transforms.Pad(padding=padding)
    ]
    if degree != 0:
        _transforms.append(
            transforms.RandomRotation(degrees=(degree, degree), interpolation=interpolation)
        )
    if up_size != 28 + 2 * padding:
        _transforms.append(
            transforms.Resize(size=[up_size, up_size], interpolation=interpolation))
    else:
        logger.debug("No upsizing.")

    ds = torchvision.datasets.MNIST(root='./data',
                                    train=False,
                                    download=True,
                                    transform=transforms.Compose(_transforms))
    return next(iter(DataLoader(dataset=ds,
                                batch_size=batch_size,
                                shuffle=False)))
