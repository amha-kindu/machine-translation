import math
import torch
import torch.nn as nn

from config import DEVICE


class TextEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding: nn.Embedding = nn.Embedding(vocab_size, d_model)
        
    """
        Args:
            x (torch.Tensor): (batches, seq_len, 1)

        Returns:
            torch.Tensor: (batches, seq_len, d_model)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding.forward(x) * math.sqrt(self.d_model)
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # To randomly zero-out a given tensor based on the given probability to combat overfitting
        self.dropout = nn.Dropout(dropout)
        
        '''
            Create the positional encoding using the following formula:
                PE(pos, 2i) = sin(pos / (10000 ^ (2i/d_model)))
                PE(pos, 2i + 1) = cos(pos / (10000 ^ (2i/d_model)))
        '''
        # Create a matrix of shape (max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)
        
        # Create a vector of shape (max_seq_len, 1)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        
        pe = pe.unsqueeze(0) # (1, max_seq_len, d_model)
        
        self.register_buffer('pe', pe)
        
    """
        Args:
            x (torch.Tensor): (batches, seq_len, d_model) where  0 < seq_len < self.max_seq_len

        Returns:
            torch.Tensor: (batches, seq_len, d_model)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        assert x.shape[1] <= self.max_seq_len, f"Input sequence length exceeds the position encoder's max sequence length  `{self.max_seq_len}`"
        return self.dropout(x + self.pe[:, :x.shape[1], :].requires_grad_(False))
    
    
class LayerNormalization(nn.Module):
    def __init__(self, epsilon: float = 1e-06) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1).to(DEVICE)) # This is multiplied
        self.bias = nn.Parameter(torch.zeros(1).to(DEVICE)) # This is added
        
    """
        Args:
            x (torch.Tensor): (batches, seq_len, d_model) where  0 < seq_len < self.max_seq_len

        Returns:
            torch.Tensor: (batches, seq_len, d_model)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True).to(DEVICE)
        std = x.std(dim=-1, keepdim=True).to(DEVICE)
        return self.alpha * (x - mean) / (std + self.epsilon) + self.bias
    
        
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, dff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, dff).to(DEVICE) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(dff, d_model).to(DEVICE) # W2 and B2
        
    """
        Args:
            x (torch.Tensor): (batches, seq_len, d_model) where  0 < seq_len < self.max_seq_len

        Returns:
            torch.Tensor: (batches, seq_len, d_model)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batches, seq_len, d_model) -> (batches, seq_len, dff) -> (batches, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
    
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        assert d_model % heads == 0, "d_model is not divisible by heads"
        
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        
        self.d_k = d_model // heads
        
        self.W_q = nn.Linear(d_model, d_model).to(DEVICE)
        self.W_k = nn.Linear(d_model, d_model).to(DEVICE)
        self.W_v = nn.Linear(d_model, d_model).to(DEVICE)

        self.W_o = nn.Linear(d_model, d_model).to(DEVICE)
        self.dropout = nn.Dropout(dropout)
    
    """
        Args:
            query (torch.Tensor): (batches, heads, seq_len, d_k) where  0 < seq_len < self.max_seq_len
            key (torch.Tensor): (batches, heads, seq_len, d_k) where  0 < seq_len < self.max_seq_len
            value (torch.Tensor): (batches, heads, seq_len, d_k) where  0 < seq_len < self.max_seq_len
            
            dropout (nn.Dropout): -
            mask (torch.Tensor): -

        Returns:
            torch.Tensor: (batches, heads, seq_len, d_k)
    """
    @staticmethod
    def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, dropout: nn.Dropout=None, mask: torch.Tensor=None) -> tuple[torch.Tensor, torch.Tensor]:
        d_k = query.shape[-1]
        
        # (batches, heads, seq_len, d_k) @ (batches, heads, d_k, seq_len) --> (batches, heads, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        

        # Here we apply the lookback mask so that the output at a certain position(which is a token)
        # can only depend on the tokens on the previous positions. We also apply the ignore masks 
        # so that attention score for the padding special token [PAD] is zero.
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e09)
            
        # (batches, heads, seq_len, seq_len) which applies softmax to the last dimension
        # so that the sum of the probabilities along this dimension equals 1
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        # (batches, heads, seq_len, seq_len) @ (batches, heads, seq_len, d_k) --> (batches, heads, seq_len, d_k)
        return (attention_scores @ value), attention_scores
    
    # q must be of shape (batches, seq_len, self.d_model) where  0 < seq_len < self.max_seq_len
    # k must be of shape (batches, seq_len, self.d_model) where  0 < seq_len < self.max_seq_len
    # v must be of shape (batches, seq_len, self.d_model) where  0 < seq_len < self.max_seq_len
    """
        Args:
            query (torch.Tensor): (batches, seq_len, d_model) where  0 < seq_len < self.max_seq_len
            key (torch.Tensor): (batches, seq_len, d_model) where  0 < seq_len < self.max_seq_len
            value (torch.Tensor): (batches, seq_len, d_model) where  0 < seq_len < self.max_seq_len
            
            mask (torch.Tensor): -

        Returns:
            torch.Tensor: (batches, seq_len, d_model)
    """
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        query: torch.Tensor = self.W_q(q) # (batches, seq_len, d_model) @ (d_model, d_model) --> (batches, seq_len, d_model)
        key: torch.Tensor = self.W_k(k)   # (batches, seq_len, d_model) @ (d_model, d_model) --> (batches, seq_len, d_model)
        value: torch.Tensor = self.W_v(v) # (batches, seq_len, d_model) @ (d_model, d_model) --> (batches, seq_len, d_model)
        
        # (batches, seq_len, d_model) --> (batches, seq_len, heads, d_k) --> (batches, heads, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.heads, self.d_k).transpose(1, 2)
        
        # Here has shape x = (batches, heads, seq_len, d_k)
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, self.dropout, mask)
        
        # (batches, heads, seq_len, d_k) --> (batches, seq_len, heads, d_k)
        x = x.transpose(1, 2)
        
        # (batches, seq_len, heads, d_k) --> (batches, seq_len, d_model)
        x = x.contiguous().view(x.shape[0], -1, self.heads * self.d_k)
        
        # (batches, seq_len, d_model) --> (batches, seq_len, d_model)
        return self.W_o(x)
    
    
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
        
    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        # return x + self.dropout(sublayer(self.norm(x)))
        return self.dropout(self.norm(x + sublayer(x)))
    
    
class Encoder(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    
    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        return self.residual_connections[1](x, self.feed_forward_block)
    
    
class EncoderStack(nn.Module):
    def __init__(self, encoder_blocks: nn.ModuleList) -> None:
        super().__init__()
        self.encoder_blocks = encoder_blocks
        self.norm = LayerNormalization()
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for block in self.encoder_blocks:
            x = block(x, mask)
        return self.norm(x)
    
    
class Decoder(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.dropout = nn.Dropout(dropout)
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
        
    # Since this transformer model is for translation we have a src_mask(from the encoder) and tgt_mask(from the decoder) which are two different languages
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
       
        
class DecoderStack(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
    
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batches, seq_len, d_model) --> (batches, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1)
    
    
class MtTransformerModel(nn.Module):
    def __init__(self, encoder: EncoderStack, decoder: DecoderStack, src_embed: TextEmbedding, tgt_embed: TextEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x: torch.Tensor):
        return self.projection_layer(x)

    @staticmethod
    def build(
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_seq_len: int,
        tgt_seq_len: int,
        d_model: int = 512,
        n_blocks: int = 6,
        heads: int = 8,
        dropout: float = 0.1,
        dff: int = 2048
    ):
        # Create the embedding layers
        src_embed = TextEmbedding(d_model, src_vocab_size)
        tgt_embed = TextEmbedding(d_model, tgt_vocab_size)
        
        # Create the positional encoding layers
        src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
        tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
        
        # Create n_blocks number of encoders
        encoders = []
        for _ in range(n_blocks):
            self_attention_block = MultiHeadAttentionBlock(d_model, heads, dropout)
            feed_forward_block = FeedForwardBlock(d_model, dff, dropout)
            
            encoders.append(
                Encoder(self_attention_block, feed_forward_block, dropout)
            )
            
        # Create n_blocks number of decoders
        decoders = []
        for _ in range(n_blocks):
            self_attention_block = MultiHeadAttentionBlock(d_model, heads, dropout)
            cross_attention_block = MultiHeadAttentionBlock(d_model, heads, dropout)
            feed_forward_block = FeedForwardBlock(d_model, dff, dropout)
            
            decoders.append(
                Decoder(self_attention_block, cross_attention_block, feed_forward_block, dropout)
            )
            
        # Create the encoder and the decoder
        encoder = EncoderStack(encoders)
        decoder = DecoderStack(decoders)
        
        # Create the projection layer
        projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
        
        # Create the transformer
        transformer = MtTransformerModel(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
        
        # Initialize the parameters
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p.data)
                
        return transformer