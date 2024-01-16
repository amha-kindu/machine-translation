import torch
from torch.utils.data import Dataset

from preprocessor import PreprocessingPipeline


class BilingualDataset(Dataset):
    def __init__(self, dataset: list, src_preprocessor: PreprocessingPipeline, tgt_preprocessor: PreprocessingPipeline, src_lang: str, tgt_lang: str, seq_len: int) -> None:
        super().__init__()
        self.seq_len  = seq_len
        self.dataset = dataset
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_preprocessor = src_preprocessor
        self.tgt_preprocessor = tgt_preprocessor
        
        self.sos_token = torch.tensor([self.src_preprocessor.tokenizer.token_to_id("[SOS]")], dtype=torch.int64)  # (1,)
        self.eos_token = torch.tensor([self.src_preprocessor.tokenizer.token_to_id("[EOS]")], dtype=torch.int64)  # (1,)
        self.pad_token = torch.tensor([self.src_preprocessor.tokenizer.token_to_id("[PAD]")], dtype=torch.int64)  # (1,)
        
    def __len__(self):
        return len(self.dataset)
    
    @staticmethod
    def lookback_mask(size: int) -> torch.Tensor:
        # Lower triangular matrix
        # [
        #   [1, 0, ... , 0],
        #   [1, 1, ... , 0],
        #   [1, 1, ... , 0],
        #   [1, 1, ... , 1]
        # ] seq_len x seq_len
        return torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int) == 0
    
    def __getitem__(self, index) -> dict:
        src_tgt_pair = self.dataset[index]
        src_text = src_tgt_pair[self.src_lang]
        tgt_text = src_tgt_pair[self.tgt_lang]
        
        src_token_ids = self.src_preprocessor.preprocess(src_text)
        tgt_token_ids = self.tgt_preprocessor.preprocess(tgt_text)
        
        src_padding = self.seq_len - len(src_token_ids) - 2
        tgt_padding = self.seq_len - len(tgt_token_ids) - 1
        
        assert src_padding >= 0 or tgt_padding < 0, "Sentence length exceeds max sequence length"
        
        # (seq_len,)
        encoder_input = torch.concat([
            self.sos_token,                                                                    # (1,)
            torch.tensor(src_token_ids, dtype=torch.int64),                     # (len(src_token_ids),)
            self.eos_token,                                                                    # (1,)
            torch.tensor([self.pad_token] * src_padding, dtype=torch.int64)     # (src_padding,)
        ])     
        
        # (seq_len,)
        decoder_input = torch.concat([
            self.sos_token,                                                                    # (1,)
            torch.tensor(tgt_token_ids, dtype=torch.int64),                     # (len(tgt_token_ids),)
            torch.tensor([self.pad_token] * tgt_padding, dtype=torch.int64)     # (tgt_padding,)
        ])                    
        
        # (seq_len,)
        label = torch.concat([
            torch.tensor(tgt_token_ids, dtype=torch.int64),                     # (tgt_padding,)
            self.eos_token,                                                                    # (1,)
            torch.tensor([self.pad_token] * tgt_padding, dtype=torch.int64)     # (tgt_padding,)
        ])  
        
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len
        
        
        return {
            # (seq_len,)
            "encoder_input": encoder_input, 
            
            # (seq_len,)                                    
            "decoder_input": decoder_input,    
                                             
            # (seq_len,) != (1,) --> (seq_len,) --> (1, 1, seq_len)
            "encoder_mask": (encoder_input != self.pad_token)
                            .unsqueeze(0).unsqueeze(0).int(),
                            
            # (seq_len,) != (1,) --> (seq_len,) --> (1, 1, seq_len) --> (1, 1, seq_len) & (1, seq_len, seq_len) --> (1, seq_len, seq_len)
            "decoder_mask": (decoder_input != self.pad_token)
                            .unsqueeze(0).unsqueeze(0).int() 
                            & self.lookback_mask(self.seq_len),  
            # (seq_len,)         
            "label": label,
            
            "src_text": src_text,
            "tgt_text": tgt_text
        }