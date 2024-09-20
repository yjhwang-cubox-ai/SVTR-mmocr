import lightning as L
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader, random_split
from lightning.pytorch.demos.boring_classes import RandomDataset
from svtr.dataset.dataset import TNGODataset
from svtr.utils.config import Config

from svtr.model.tps_preprocessor import STN
from svtr.model.svtr_encoder import SVTREncoder
from svtr.model.svtr_decoder import SVTRDecoder
from svtr.model.loss import CTCModuleLoss
from svtr.utils.dictionary import Dictionary

class SVTR(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.dictionary = Dictionary(dict_file=self.config.global_config['character_dict_path'], with_padding=True, with_unknown=True)
        self.preprocessor = STN(in_channels=3)
        self.encoder = SVTREncoder()
        self.decoder = SVTRDecoder(in_channels=192, dictionary=self.dictionary)
        self.criterion = CTCModuleLoss(dictionary=self.dictionary)
    def forward(self, x):
        x = self.preprocessor(x)
        x = self.encoder(x)
        if self.training:
            x = self.decoder.forward_train(x)
        else:
            x = self.decoder.forward_test(x)
        return x