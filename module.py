import lightning as L
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, random_split
from lightning.pytorch.demos.boring_classes import RandomDataset
from dataset import TNGODataset
from config import Config

from tps_preprocessor import STN
from svtr_encoder import SVTREncoder
from svtr_decoder import SVTRDecoder
from loss import CTCModuleLoss
from dictionary import Dictionary

class SVTRModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.dictionary = Dictionary(dict_file=self.config.global_config['character_dict_path'], with_padding=True, with_unknown=True)
        self.preprocessor = STN(in_channels=3)
        self.encoder = SVTREncoder()
        self.decoder = SVTRDecoder(in_channels=192, dictionary=None)
        self.criterion = CTCModuleLoss(dictionary=self.dictionary)

    def training_step(self, batch, batch_idx):
        image, text = batch['image'], batch['text']
        x = self.preprocessor(image)
        x = self.encoder(x)
        x = self.decoder.forward_train(x)
        loss = self.criterion(x, text)
        self.log('train_loss', loss)
        return loss
    def validation_step(self, batch, batch_idx):
        image, text = batch['image'], batch['text']
        x = self.preprocessor(image)
        x = self.encoder(x)
        x = self.decoder.forward_test(x)
        loss = self.criterion(x, text)
        self.log('val_loss', loss)
        return loss
    def test_step(self, batch, batch_idx):
        pass

class SvtrDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.batch_size: int = 256
        self.train_dataset = None
        self.val_dataset = None
        self.collate_fn = None
        self.config = Config()

    def setup(self, stage=None):
        # json_list = self.config.dataset_config['train_json']
        train_datasets = self._tngo_dataset(self.config.dataset_config['train_json'], mode='train')
        # TNGoDataset(json_list, mode='train')
        train_set_size = int(len(train_datasets) * 0.9)
        val_set_size = len(train_datasets) - train_set_size

        # split the dataset
        seed = torch.Generator().manual_seed(42)
        train_datasets, val_datasets = random_split(train_datasets, [train_set_size, val_set_size], generator=seed)
        train_datasets.mode = 'train'
        val_datasets.mode = 'test'
        test_datasets = self._tngo_dataset(self.config.dataset_config['test_json'], mode='test')
        self.train_dataset = train_datasets
        self.val_dataset = val_datasets
        self.test_dataset = test_datasets
        
    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, self.batch_size, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return data.DataLoader(self.val_dataset, self.batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return data.DataLoader(self.test_dataset, self.batch_size, collate_fn=self.collate_fn)
    
    def _tngo_dataset(self, data_json, mode) -> TNGODataset:
        return TNGODataset(data_json = data_json, mode=mode)

if __name__ == '__main__':
    dm = SvtrDataModule()
    print('done')
