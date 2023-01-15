import os
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.datamodules.components.imgs_dataset import ImagesDataset
from src.datamodules.components.custom_len_dataset import CustomLenDataset
from src.datamodules.transforms.transform_img_gen import get_transform


class AutoEncoderDataModule(LightningDataModule):
    """
    Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
            self,
            data_dir: str = "data/",
            val_len: int = 1000,
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            transform_params={}
    ):
        super().__init__()

        self.val_len = val_len
        self.transform_params = transform_params
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning separately when using `trainer.fit()` and `trainer.test()`!
        The `stage` can be used to differentiate whether the `setup()` is called before trainer.fit()` or `trainer.test()`."""
        train_real_transform = get_transform(**self.transform_params, is_train=True, apply_strong=False)
        val_real_transform = get_transform(**self.transform_params, is_train=False, apply_strong=False)

        self.train_real_dataset = ImagesDataset(train_real_transform, os.path.join(self.data_dir, 'train_real'))
        self.val_dataset = ImagesDataset(val_real_transform, os.path.join(self.data_dir, 'val_real'))

    def train_dataloader(self):
        return DataLoader(self.train_real_dataset,
                       batch_size=self.batch_size,
                       shuffle=True,
                       drop_last=True,
                       pin_memory=self.pin_memory,
                       num_workers=self.num_workers)


    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          drop_last=False,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)
