import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class BahdanauAttention(nn.Module):
    def __init__(self, enc_hidden: int, dec_hidden: int, attn_dim: int = 256):
        super().__init__()
        self.W_enc = nn.Linear(enc_hidden, attn_dim, bias=False)
        self.W_dec = nn.Linear(dec_hidden, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, enc_out: torch.Tensor,
                dec_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        energy = self.v(torch.tanh(
            self.W_enc(enc_out) + self.W_dec(dec_hidden).unsqueeze(1)
        )).squeeze(-1)
        attn_weights = F.softmax(energy, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), enc_out).squeeze(1)
        return context, attn_weights


class LuongAttention(nn.Module):
    def __init__(self, hidden_size: int, method: str = 'general'):
        super().__init__()
        assert method in ('dot', 'general', 'concat')
        self.method = method
        if method == 'general':
            self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        elif method == 'concat':
            self.W = nn.Linear(hidden_size * 2, hidden_size, bias=False)
            self.v = nn.Linear(hidden_size, 1, bias=False)

    def score(self, dec_h: torch.Tensor, enc_out: torch.Tensor) -> torch.Tensor:
        if self.method == 'dot':
            return torch.bmm(enc_out, dec_h.unsqueeze(-1)).squeeze(-1)
        elif self.method == 'general':
            return torch.bmm(self.W(enc_out), dec_h.unsqueeze(-1)).squeeze(-1)
        else:
            dec_exp = dec_h.unsqueeze(1).expand_as(enc_out)
            return self.v(torch.tanh(self.W(torch.cat([enc_out, dec_exp], dim=-1)))).squeeze(-1)

    def forward(self, dec_h: torch.Tensor,
                enc_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_weights = F.softmax(self.score(dec_h, enc_out), dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), enc_out).squeeze(1)
        return context, attn_weights


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int,
                 num_layers: int = 2, dropout: float = 0.3,
                 bidirectional: bool = True, pad_idx: int = 0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(embed_dim, hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
                          bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        if bidirectional:
            self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.dropout(self.embedding(src))
        enc_out, h_n = self.rnn(emb)
        if self.bidirectional:
            h_n = torch.tanh(self.fc_hidden(
                torch.cat([h_n[-2], h_n[-1]], dim=-1)
            )).unsqueeze(0).repeat(self.num_layers, 1, 1)
        return enc_out, h_n


class DecoderWithAttention(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int,
                 enc_hidden: int, num_layers: int = 2, dropout: float = 0.3,
                 attention_type: str = 'bahdanau'):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

        if attention_type == 'bahdanau':
            self.attention = BahdanauAttention(enc_hidden, hidden_size)
        else:
            self.attention = LuongAttention(hidden_size, method='general')

        self.rnn = nn.GRU(embed_dim + enc_hidden, hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc_out = nn.Linear(hidden_size + enc_hidden + embed_dim, vocab_size)

    def forward(self, tgt_token: torch.Tensor, hidden: torch.Tensor,
                enc_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.dropout(self.embedding(tgt_token.unsqueeze(1)))
        context, attn_w = self.attention(hidden[-1], enc_out)
        rnn_input = torch.cat([emb, context.unsqueeze(1)], dim=-1)
        out, hidden = self.rnn(rnn_input, hidden)
        pred_input = torch.cat([out.squeeze(1), context, emb.squeeze(1)], dim=-1)
        logit = self.fc_out(pred_input)
        return logit, hidden, attn_w


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab: int, tgt_vocab: int,
                 embed_dim: int = 256, hidden_size: int = 512,
                 enc_layers: int = 2, dec_layers: int = 2,
                 dropout: float = 0.3, attention_type: str = 'bahdanau',
                 src_pad_idx: int = 0, tgt_pad_idx: int = 0):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.tgt_vocab = tgt_vocab

        enc_out_size = hidden_size * 2
        self.encoder = Encoder(src_vocab, embed_dim, hidden_size,
                               enc_layers, dropout, bidirectional=True)
        self.decoder = DecoderWithAttention(tgt_vocab, embed_dim, hidden_size,
                                            enc_out_size, dec_layers, dropout, attention_type)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        B, T_tgt = tgt.shape
        enc_out, hidden = self.encoder(src)
        outputs = torch.zeros(B, T_tgt, self.tgt_vocab, device=src.device)
        token = tgt[:, 0]

        for t in range(1, T_tgt):
            logit, hidden, _ = self.decoder(token, hidden, enc_out)
            outputs[:, t] = logit
            use_teacher = torch.rand(1).item() < teacher_forcing_ratio
            token = tgt[:, t] if use_teacher else logit.argmax(-1)

        return outputs

    @torch.no_grad()
    def translate(self, src: torch.Tensor, max_len: int,
                  sos_idx: int, eos_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        enc_out, hidden = self.encoder(src)
        B = src.size(0)
        token = torch.full((B,), sos_idx, dtype=torch.long, device=src.device)
        outputs, attn_weights = [], []

        for _ in range(max_len):
            logit, hidden, attn_w = self.decoder(token, hidden, enc_out)
            token = logit.argmax(-1)
            outputs.append(token)
            attn_weights.append(attn_w)
            if (token == eos_idx).all():
                break

        return torch.stack(outputs, dim=1), torch.stack(attn_weights, dim=1)


class Seq2SeqLSTM(nn.Module):
    def __init__(self, src_vocab: int, tgt_vocab: int,
                 embed_dim: int = 256, hidden_size: int = 512,
                 num_layers: int = 2, dropout: float = 0.3,
                 src_pad_idx: int = 0, tgt_pad_idx: int = 0):
        super().__init__()
        self.tgt_vocab = tgt_vocab
        self.src_embed = nn.Embedding(src_vocab, embed_dim, padding_idx=src_pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab, embed_dim, padding_idx=tgt_pad_idx)
        self.encoder = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers,
                               batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
                               bidirectional=False)
        self.decoder = nn.LSTM(embed_dim + hidden_size, hidden_size, num_layers=num_layers,
                               batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.attention = BahdanauAttention(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size * 2, tgt_vocab)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        enc_out, (h, c) = self.encoder(self.dropout(self.src_embed(src)))
        B, T_tgt = tgt.shape
        outputs = torch.zeros(B, T_tgt, self.tgt_vocab, device=src.device)
        token = tgt[:, 0]

        for t in range(1, T_tgt):
            emb = self.dropout(self.tgt_embed(token)).unsqueeze(1)
            context, _ = self.attention(enc_out, h[-1])
            dec_in = torch.cat([emb, context.unsqueeze(1)], dim=-1)
            out, (h, c) = self.decoder(dec_in, (h, c))
            logit = self.fc_out(torch.cat([out.squeeze(1), context], dim=-1))
            outputs[:, t] = logit
            use_teacher = torch.rand(1).item() < teacher_forcing_ratio
            token = tgt[:, t] if use_teacher else logit.argmax(-1)

        return outputs


def seq2seq_small(src_vocab: int, tgt_vocab: int) -> Seq2Seq:
    return Seq2Seq(src_vocab, tgt_vocab, embed_dim=128, hidden_size=256,
                   enc_layers=1, dec_layers=1)

def seq2seq_base(src_vocab: int, tgt_vocab: int) -> Seq2Seq:
    return Seq2Seq(src_vocab, tgt_vocab, embed_dim=256, hidden_size=512,
                   enc_layers=2, dec_layers=2, attention_type='bahdanau')

def seq2seq_luong(src_vocab: int, tgt_vocab: int) -> Seq2Seq:
    return Seq2Seq(src_vocab, tgt_vocab, embed_dim=256, hidden_size=512,
                   enc_layers=2, dec_layers=2, attention_type='luong')

def seq2seq_lstm(src_vocab: int, tgt_vocab: int) -> Seq2SeqLSTM:
    return Seq2SeqLSTM(src_vocab, tgt_vocab, embed_dim=256, hidden_size=512, num_layers=2)
