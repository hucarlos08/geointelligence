import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

class LandsatDataModule(pl.LightningDataModule):
    def __init__(self, train_file, test_file, batch_size=32, num_workers=4):
        super().__init__()
        self.train_file = train_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Load training and test dataset
        self.train_dataset = LandsatDataset(self.train_file)
        self.test_dataset = LandsatDataset(self.test_file)

        # Optionally split train dataset into train/validation sets
        train_size = int(0.8 * len(self.train_dataset))
        val_size = len(self.train_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(self.train_dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
