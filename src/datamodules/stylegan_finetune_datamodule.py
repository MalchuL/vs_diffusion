from typing import Optional

from torch.utils.data import DataLoader

from src.datamodules.components.repeat_dataset import RepeatDataset
from src.datamodules.imagegen_datamodule import ImageGenDataModule


class ImageGenFinetuneDataModule(ImageGenDataModule):
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

    def __init__(self, data_dir: str = "data/", val_len: int = 1000, length: int = 1000, batch_size: int = 64, val_batch_size=None,
                 num_workers: int = 0, pin_memory: bool = False, transform_params={}):
        super().__init__(data_dir, val_len, batch_size, num_workers, pin_memory, transform_params)
        self.length = length
        if val_batch_size is None:
            self.val_batch_size = self.batch_size
        else:
            self.val_batch_size = val_batch_size

    def setup(self, stage: Optional[str] = None):
        super().setup(stage)
        self.train_real_dataset = RepeatDataset(self.train_real_dataset, length=self.length)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.val_batch_size,
                          shuffle=False,
                          drop_last=False,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)