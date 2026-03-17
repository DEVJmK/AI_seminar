import torch
import torch.nn as nn
from typing import Optional, Tuple


class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_ir = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_iz = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_in = nn.Linear(input_size, hidden_size, bias=True)
        self.W_hn = nn.Linear(hidden_size, hidden_size, bias=True)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            else:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor,
                h: Optional[torch.Tensor] = None) -> torch.Tensor:
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_size, device=x.device, dtype=x.dtype)

        r = torch.sigmoid(self.W_ir(x) + self.W_hr(h))
        z = torch.sigmoid(self.W_iz(x) + self.W_hz(h))
        n = torch.tanh(self.W_in(x) + r * self.W_hn(h))
        h_next = (1 - z) * n + z * h
        return h_next


class GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 batch_first: bool = True, dropout: float = 0.0,
                 bidirectional: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout) if dropout > 0 and num_layers > 1 else None

        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            for direction in range(self.num_directions):
                in_size = input_size if layer == 0 else hidden_size * self.num_directions
                self.cells.append(GRUCell(in_size, hidden_size))

    def _get_cell(self, layer: int, direction: int) -> GRUCell:
        return self.cells[layer * self.num_directions + direction]

    def forward(self, x: torch.Tensor,
                h0: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.batch_first:
            x = x.transpose(0, 1)

        T, B, _ = x.shape

        if h0 is None:
            h0 = torch.zeros(self.num_layers * self.num_directions, B,
                             self.hidden_size, device=x.device, dtype=x.dtype)

        h_n_list = []
        layer_input = x

        for layer in range(self.num_layers):
            fwd_outputs, bwd_outputs = [], []
            h_fwd = h0[layer * self.num_directions]

            for t in range(T):
                h_fwd = self._get_cell(layer, 0)(layer_input[t], h_fwd)
                fwd_outputs.append(h_fwd)

            h_n_list.append(h_fwd)

            if self.bidirectional:
                h_bwd = h0[layer * self.num_directions + 1]
                for t in reversed(range(T)):
                    h_bwd = self._get_cell(layer, 1)(layer_input[t], h_bwd)
                    bwd_outputs.insert(0, h_bwd)
                h_n_list.append(h_bwd)

            fwd_seq = torch.stack(fwd_outputs, dim=0)
            if self.bidirectional:
                bwd_seq = torch.stack(bwd_outputs, dim=0)
                layer_output = torch.cat([fwd_seq, bwd_seq], dim=-1)
            else:
                layer_output = fwd_seq

            if self.dropout is not None and layer < self.num_layers - 1:
                layer_output = self.dropout(layer_output)

            layer_input = layer_output

        output = layer_input
        if self.batch_first:
            output = output.transpose(0, 1)

        h_n = torch.stack(h_n_list, dim=0)
        return output, h_n


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int,
                 num_classes: int, num_layers: int = 2, dropout: float = 0.3,
                 bidirectional: bool = False, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = GRU(embed_dim, hidden_size, num_layers=num_layers,
                       dropout=dropout, bidirectional=bidirectional, batch_first=True)
        directions = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * directions, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.dropout(self.embedding(x))
        _, h_n = self.gru(emb)
        if self.gru.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h_last = h_n[-1]
        return self.fc(self.dropout(h_last))


class GRULanguageModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int,
                 num_layers: int = 2, dropout: float = 0.5, tie_weights: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = GRU(embed_dim, hidden_size, num_layers=num_layers,
                       dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        if tie_weights and embed_dim == hidden_size:
            self.fc.weight = self.embedding.weight

    def forward(self, x: torch.Tensor,
                hidden: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.dropout(self.embedding(x))
        out, hidden = self.gru(emb, hidden)
        logits = self.fc(self.dropout(out))
        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.gru.num_layers, batch_size,
                           self.gru.hidden_size, device=device)
