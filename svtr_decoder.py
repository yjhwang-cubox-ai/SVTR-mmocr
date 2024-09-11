import torch
import torch.nn as nn
from typing import Dict, List, Optional, Sequence, Union

class SVTRDecoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 dictionary: Union[Dict, Dict] = None,
                 module_loss: Optional[Dict] = None,
                 postprocessor: Optional[Dict] = None,
                 max_seq_len: int = 25,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None) -> None:
        
        super().__init__()

        self.in_channels = in_channels
        self.module_loss = module_loss
        self.postprocessor = postprocessor
        self.dictionary = dictionary
        self.max_seq_len = max_seq_len
        self._init_cfg = init_cfg

        self.decoder = nn.Linear(
            in_features=in_channels, out_features=self.dictionary.num_classes())
        self.softmax = nn.Softmax(dim=-1)
    
    def forward_train(self, out_enc):
        assert out_enc.size(2) == 1
        x = out_enc.squeeze(2)
        x = x.permute(0, 2, 1)
        predicts = self.decoder(x)
        return predicts
    
    def forward_test(self, out_enc):
        return self.softmax(self.forward_train(out_enc))