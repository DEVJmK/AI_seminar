import torch
import torch.nn as nn
from typing import Optional, Tuple


class LSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_ih = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.W_hh = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                param.data[self.hidden_size:2 * self.hidden_size].fill_(1.0)

    def forward(self, x: torch.Tensor,
                state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.size(0)
        if state is None:
            h = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
            c = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            h, c = state

        gates = self.W_ih(x) + self.W_hh(h)
        i, f, g, o = gates.chunk(4, dim=-1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 bias: bool = True, batch_first: bool = True,
                 dropout: float = 0.0, bidirectional: bool = False):
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
                layer_input = input_size if layer == 0 else hidden_size * self.num_directions
                self.cells.append(LSTMCell(layer_input, hidden_size))

    def _get_cell(self, layer: int, direction: int) -> LSTMCell:
        return self.cells[layer * self.num_directions + direction]

    def forward(self, x: torch.Tensor,
                hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.batch_first:
            x = x.transpose(0, 1)

        T, B, _ = x.shape

        if hx is None:
            h0 = torch.zeros(self.num_layers * self.num_directions, B,
                             self.hidden_size, device=x.device, dtype=x.dtype)
            c0 = torch.zeros_like(h0)
        else:
            h0, c0 = hx

        h_n_list, c_n_list = [], []
        layer_input = x

        for layer in range(self.num_layers):
            fwd_outputs, bwd_outputs = [], []
            h_fwd = h0[layer * self.num_directions]
            c_fwd = c0[layer * self.num_directions]

            for t in range(T):
                h_fwd, c_fwd = self._get_cell(layer, 0)(layer_input[t], (h_fwd, c_fwd))
                fwd_outputs.append(h_fwd)

            h_n_list.append(h_fwd)
            c_n_list.append(c_fwd)

            if self.bidirectional:
                h_bwd = h0[layer * self.num_directions + 1]
                c_bwd = c0[layer * self.num_directions + 1]
                for t in reversed(range(T)):
                    h_bwd, c_bwd = self._get_cell(layer, 1)(layer_input[t], (h_bwd, c_bwd))
                    bwd_outputs.insert(0, h_bwd)
                h_n_list.append(h_bwd)
                c_n_list.append(c_bwd)

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
        c_n = torch.stack(c_n_list, dim=0)
        return output, (h_n, c_n)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int,
                 num_classes: int, num_layers: int = 2, dropout: float = 0.3,
                 bidirectional: bool = False, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = LSTM(embed_dim, hidden_size, num_layers=num_layers,
                         dropout=dropout, bidirectional=bidirectional, batch_first=True)
        directions = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * directions, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.dropout(self.embedding(x))
        _, (h_n, _) = self.lstm(emb)
        if self.lstm.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h_last = h_n[-1]
        return self.fc(self.dropout(h_last))


class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int,
                 num_layers: int = 2, dropout: float = 0.5, tie_weights: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = LSTM(embed_dim, hidden_size, num_layers=num_layers,
                         dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        if tie_weights and embed_dim == hidden_size:
            self.fc.weight = self.embedding.weight

    def forward(self, x: torch.Tensor,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        emb = self.dropout(self.embedding(x))
        out, hidden = self.lstm(emb, hidden)
        logits = self.fc(self.dropout(out))
        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n = self.lstm.num_layers
        h = self.lstm.hidden_size
        return (torch.zeros(n, batch_size, h, device=device),
                torch.zeros(n, batch_size, h, device=device))
