import torch.nn as nn
from svtr.utils.utils import list_from_file
from typing import List, Sequence

class Dictionary(nn.Module):
    def __init__(self,
                 dict_file: str,
                 with_start: bool = False,
                 with_end: bool = False,
                 same_start_end: bool = False,
                 with_padding: bool = True,
                 with_unknown: bool = True,
                 start_token: str = "<SOS>",
                 end_token: str = "<EOS>",
                 start_end_token: str = "<SOS/EOS>",
                 padding_token: str = "<PAD>",
                 unknown_token: str = "<UNK>") -> None:
        super().__init__()
        self.with_start = with_start
        self.with_end = with_end
        self.same_start_end = same_start_end
        self.with_padding = with_padding
        self.with_unknown = with_unknown
        self.start_end_token = start_end_token
        self.start_token = start_token
        self.end_token = end_token
        self.padding_token = padding_token
        self.unknown_token = unknown_token

        assert isinstance(dict_file, str)
        self._dict = []
        for line_num, line in enumerate(list_from_file(dict_file)):
            line = line.strip('\r\n')
            if len(line) > 1:
                raise ValueError('Expect each line has 0 or 1 character, '
                                 f'got {len(line)} characters '
                                 f'at line {line_num + 1}')
            if line != '':
                self._dict.append(line)
        
        self._char2idx = {char: idx for idx, char in enumerate(self._dict)}

        self._update_dict()
        assert len(set(self._dict)) == len(self._dict), \
            'Dictionary contains duplicated characters.'
    
    def num_classes(self):
        return len(self._dict)
    
    def dict(self) -> list:
        return self._dict
    
    def char2idx(self, char: str, strict: bool = True) -> int:
        char_idx = self._char2idx.get(char, None)
        if char_idx is None:
            if self.with_unknown:
                return self.unknown_idx
            elif not strict:
                return None
            else:
                raise Exception(f'Chararcter: {char} not in dict,'
                                ' please check gt_label and use'
                                ' custom dict file,'
                                ' or set "with_unknown=True"')
        return char_idx
    
    def str2idx(self, string: str) -> List:
        idx = list()
        for s in string:
            char_idx = self.char2idx(s)
            if char_idx is not None:
                idx.append(char_idx)
        return idx
    
    def idx2str(self, index: Sequence[int]) -> str:
        assert isinstance(index, (list, tuple))
        string = ''
        for i in index:
            assert i < len(self._dict), f'Index: {i} out of range! Index ' \
                                        f'must be less than {len(self._dict)}'
            string += self._dict[i]
        return string

    def _update_dict(self):
        # SOS/EOS
        self.start_idx = None
        self.end_idx = None
        if self.with_start and self.with_end and self.same_start_end:
            self._dict.append(self.start_end_token)
            self.start_idx = len(self._dict) - 1
            self.end_idx = self.start_idx
        else:
            if self.with_start:
                self._dict.append(self.start_token)
                self.start_idx = len(self._dict) - 1
            if self.with_end:
                self._dict.append(self.end_token)
                self.end_idx = len(self._dict) - 1
        # Padding
        self.padding_idx = None
        if self.with_padding:
            self._dict.append(self.padding_token)
            self.padding_idx = len(self._dict) - 1
        
        # Unknown
        self.unknown_idx = None
        if self.with_unknown and self.unknown_token is not None:
            self._dict.append(self.unknown_token)
            self.unknown_idx = len(self._dict) - 1
        
        # Update char2idx
        self._char2idx = {}
        for idx, char in enumerate(self._dict):
            self._char2idx[char] = idx