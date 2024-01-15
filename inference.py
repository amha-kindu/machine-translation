import torch
import random
from enum import Enum
from tokenizers import Tokenizer
from dataset import BilingualDataset
from model import MtTransformerModel

class SamplingStrategy(Enum):
    GREEDY=0,
    TOP_K_RANDOM=1,
    NUCLEUS=2

class MtInferenceEngine:
    
    def __init__(self, model: MtTransformerModel, src_tokenizer: Tokenizer, tgt_tokenizer: Tokenizer, device: torch.device, top_k: int= 5, nucleus_threshold=10) -> None:
        self.model = model
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.device = device
        self.top_k = top_k
        self.nucleus_threshold = nucleus_threshold
        self.sos_token = torch.tensor([src_tokenizer.token_to_id("[SOS]")], dtype=torch.int64)  # (1,)
        self.eos_token = torch.tensor([src_tokenizer.token_to_id("[EOS]")], dtype=torch.int64)  # (1,)
        self.pad_token = torch.tensor([src_tokenizer.token_to_id("[PAD]")], dtype=torch.int64)  # (1,)
        self.model.eval()
     
    def translate(self, source_text: str, max_len: int, strategy=SamplingStrategy.GREEDY):
        data = BilingualDataset([{"en": source_text, "am": ""}], self.src_tokenizer, self.tgt_tokenizer, "en", "am", max_len)
        encoder_input = data["encoder_input"].to(self.device)
        encoder_mask = data["encoder_mask"].to(self.device)
        decoder_mask = data["decoder_mask"].to(self.device)
        
        self.translate_raw(encoder_input, encoder_mask, decoder_mask, max_len, strategy)
    
    @torch.no_grad()
    def translate_raw(self, encoder_input: torch.Tensor, encoder_mask: torch.Tensor, decoder_mask: torch.Tensor, max_len: int, strategy=SamplingStrategy.GREEDY):        
        # Reuse the encoder output to generate subsequent tokens using the decoder
        encoder_output = self.model.encode(encoder_input, encoder_mask)
        
        # Initialize the decoder input with the start of sentence token
        decoder_input = torch.concat([
            self.sos_token,                                                       # (1,)
            torch.tensor([self.pad_token] * (max_len - 1), dtype=torch.int64)     # (max_seq_len - 1,)
        ]).unsqueeze(0)    
        
        count = 1
                
        # While length of the generated text is less than max_len and the 
        # last generated token is different from eos_token
        # keep generating the next token
        while count < max_len and decoder_input[:, -1] != self.eos_token:
            # Calculate the output of the decoder
            decoder_output = self.model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            
            # Retrieve the current probability distribution of the tokens in the vocabulary
            token_probab = self.model.project(decoder_output)                       # (1, seq_len, tgt_vocab_size)
            
            # Retrieve the last predicted token in the sequence
            next_token_prob = token_probab[:, -1]                                   # (1, tgt_vocab_size)
            
            """
                Retrieve the next token based on its probability and the strategy selected
            """
            if strategy == SamplingStrategy.GREEDY:
                # Retrieve the token with the highest probability
                _, next_token = torch.max(next_token_prob, dim=1)                   # value=tensor([max_prob]), indice=tensor([token_id])
            elif strategy == SamplingStrategy.TOP_K_RANDOM:
                _, topk_indices = torch.topk(token_probab, k=self.top_k)
                next_token = random.choice(topk_indices.squeeze().tolist())
            else:
                raise Exception("Unknown sampling strategy passed")
            
            # Modify to the decoder input for the next iteration
            decoder_input[0, count] = next_token.item()
            count += 1
        
        # Remove the batch dimension 
        decoder_input = decoder_input.squeeze(0)                                    # torch.tensor([...]) with shape tensor.Size([max_len])
    
        return self.tgt_tokenizer.decode(decoder_input.detach().cpu().tolist())