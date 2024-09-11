from typing import Dict

import torch.utils.data as Data
import pytorch_lightning as pl

from .build_dataset import build_datasets


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        exp: str,
        dataset: Dict,

        batch_size: int,
        train_shuffle: bool,
        num_workers: int,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.exp = exp

        self.train_dataset, self.val_dataset, self.test_dataset = build_datasets(exp, dataset)

        self.batch_size = batch_size
        self.train_shuffle = train_shuffle
        self.num_workers = num_workers

    def train_dataloader(self):
        return Data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.train_shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        return Data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return Data.DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers)