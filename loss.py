import torch
import torch.nn as nn

class CTCModuleLoss(nn.Module):
    def __init__(self,
                 dictionary,
                 max_seq_len: int = 40,
                 letter_case: str = 'unchanged',
                 flatten: bool = True,
                 reduction: str = 'mean',
                 zero_infinity: bool = False,
                 pad_with: str = 'auto',
                 **kwargs) -> None:
        super().__init__()
        assert isinstance(flatten, bool)
        assert isinstance(reduction, str)
        assert isinstance(zero_infinity, bool)
        
        self.dictionary = dictionary
        assert letter_case in ['unchanged', 'upper', 'lower']
        self.letter_case = letter_case
        self.flatten = flatten
        self.ctc_loss = nn.CTCLoss(
            blank=self.dictionary.padding_idx,
            reduction=reduction,
            zero_infinity=zero_infinity)
    
    def forward(self, outputs: torch.Tensor, texts: list):
        outputs = torch.log_softmax(outputs, dim=2)
        bsz, seq_len = outputs.size(0), outputs.size(1)
        outputs_for_loss = outputs.permute(1, 0, 2).contiguous()
        texts_indexes = self.get_targets(texts)
        targets = [
            indexes[:seq_len]
            for indexes in texts_indexes
        ]
        target_lengths = torch.IntTensor([len(t) for t in targets])
        target_lengths = torch.clamp(target_lengths, max=seq_len).long()
        input_lengths = torch.full(
            size=(bsz, ), fill_value=seq_len, dtype=torch.long)
        if self.flatten:
            targets = torch.cat(targets)
        else:
            padded_targets = torch.full(
                size=(bsz, seq_len),
                fill_value=self.dictionary.padding_idx,
                dtype=torch.long)
            for idx, valid_len in enumerate(target_lengths):
                padded_targets[idx, :valid_len] = targets[idx][:valid_len]
            targets = padded_targets
        loss_ctc = self.ctc_loss(outputs_for_loss, targets, input_lengths,
                                 target_lengths)
        return loss_ctc

    def get_targets(self, texts:list):
        indexes_list = []
        for text in texts:
            if self.letter_case in ['upper', 'lower']:
                text = getattr(text, self.letter_case)()
            indexes = self.dictionary.str2idx(text)
            indexes = torch.IntTensor(indexes)
            indexes_list.append(indexes)
        return indexes_list