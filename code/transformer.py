import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor,
                                     V: torch.Tensor,
                                     mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = self.dropout(F.softmax(scores, dim=-1))
        return torch.matmul(attn, V)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = Q.shape[0]
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        out = self.scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.W_o(out)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out, src_mask)))
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int,
                 d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout)
                                     for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int,
                 d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout)
                                     for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 d_model: int = 512, num_heads: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, d_ff: int = 2048,
                 max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.encoder = TransformerEncoder(num_encoder_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = TransformerDecoder(num_decoder_layers, d_model, num_heads, d_ff, dropout)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        return (src != pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        T = tgt.size(1)
        pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(torch.ones(T, T, device=tgt.device)).bool()
        return pad_mask & causal_mask

    def encode(self, src: torch.Tensor,
               src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder(self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model)), src_mask)

    def decode(self, tgt: torch.Tensor, enc_out: torch.Tensor,
               src_mask: Optional[torch.Tensor] = None,
               tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.decoder(self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model)),
                            enc_out, src_mask, tgt_mask)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_pad_idx: int = 0, tgt_pad_idx: int = 0) -> torch.Tensor:
        src_mask = self.make_src_mask(src, src_pad_idx)
        tgt_mask = self.make_tgt_mask(tgt, tgt_pad_idx)
        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(tgt, enc_out, src_mask, tgt_mask)
        return self.fc_out(dec_out)

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, max_len: int,
                      sos_idx: int, eos_idx: int,
                      src_pad_idx: int = 0) -> torch.Tensor:
        self.eval()
        src_mask = self.make_src_mask(src, src_pad_idx)
        enc_out = self.encode(src, src_mask)
        B = src.size(0)
        ys = torch.full((B, 1), sos_idx, dtype=torch.long, device=src.device)
        for _ in range(max_len - 1):
            tgt_mask = self.make_tgt_mask(ys)
            out = self.decode(ys, enc_out, src_mask, tgt_mask)
            next_token = self.fc_out(out[:, -1]).argmax(-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
            if (next_token == eos_idx).all():
                break
        return ys


def transformer_base(src_vocab: int, tgt_vocab: int) -> Transformer:
    return Transformer(src_vocab, tgt_vocab, d_model=512, num_heads=8,
                       num_encoder_layers=6, num_decoder_layers=6, d_ff=2048)

def transformer_small(src_vocab: int, tgt_vocab: int) -> Transformer:
    return Transformer(src_vocab, tgt_vocab, d_model=256, num_heads=4,
                       num_encoder_layers=3, num_decoder_layers=3, d_ff=1024)

def transformer_large(src_vocab: int, tgt_vocab: int) -> Transformer:
    return Transformer(src_vocab, tgt_vocab, d_model=1024, num_heads=16,
                       num_encoder_layers=6, num_decoder_layers=6, d_ff=4096)
