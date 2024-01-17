import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader

class BilingualDataset(Dataset):
    def __init__(self, dataset: list[dict], config: dict, src_tokenizer: Tokenizer, tgt_tokenizer: Tokenizer) -> None:
        super().__init__()
        self.dataset = dataset
        self.config = config

        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        
        self.sos_token = torch.tensor([self.src_tokenizer.token_to_id("[SOS]")], dtype=torch.int64)  # (1,)
        self.eos_token = torch.tensor([self.src_tokenizer.token_to_id("[EOS]")], dtype=torch.int64)  # (1,)
        self.pad_token = torch.tensor([self.src_tokenizer.token_to_id("[PAD]")], dtype=torch.int64)  # (1,)
        
    def __len__(self):
        return len(self.dataset)
    
    def batch_iterator(self, batch_size: int):
        return DataLoader(self, batch_size, shuffle=True)
    
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
        src_text = src_tgt_pair[self.config["src_lang"]]
        tgt_text = src_tgt_pair[self.config["tgt_lang"]]
        
        src_token_ids = self.src_tokenizer.encode(src_text).ids
        tgt_token_ids = self.tgt_tokenizer.encode(tgt_text).ids
        
        src_padding = self.config["seq_len"] - len(src_token_ids) - 2
        tgt_padding = self.config["seq_len"] - len(tgt_token_ids) - 1
        
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
        
        assert encoder_input.size(0) == self.config["seq_len"]
        assert decoder_input.size(0) == self.config["seq_len"]
        assert label.size(0) == self.config["seq_len"]
        
        
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
                            & self.lookback_mask(self.config["seq_len"]),  
            # (seq_len,)         
            "label": label,
            
            "src_text": src_text,
            "tgt_text": tgt_text
        }