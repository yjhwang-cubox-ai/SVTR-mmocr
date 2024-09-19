import lightning as L
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, random_split
from lightning.pytorch.demos.boring_classes import RandomDataset
from dataset import TNGoDataset
from config import Config

class SvtrDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.batch_size: int = 256
        self.train_dataset = None
        self.val_dataset = None
        self.collate_fn = None
        self.config = Config()

    def setup(self, stage=None):
        json_list = self.config.dataset_config['train_json']
        train_datasets = TNGoDataset(json_list, mode='train')
        train_set_size = int(len(train_dataset) * 0.9)
        val_set_size = len(train_dataset) - train_set_size

        # split the dataset
        seed = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(train_datasets, [train_set_size, val_set_size], generator=seed)


        print(json_list)



        # make assignments here (val/train/test split)
        # called on every process in DDP
        pass
    def train_dataloader(self):
        return data.DataLoader(self.train_dataset)

    def val_dataloader(self):
        return data.DataLoader(self.val_dataset)

    def test_dataloader(self):
        return data.DataLoader(self.test_dataset)

if __name__ == '__main__':
    dm = SvtrDataModule()
    dm.setup()
    print('done')
