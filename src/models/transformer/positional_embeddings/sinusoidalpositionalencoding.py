# Based off code from https://github.com/tatp22/multidim-positional-encoding
import math
import torch
import numpy as np
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)

        self.register_buffer("pe", pe)
        self.alpha = nn.Parameter(torch.zeros(d_model))  # learnable per-dim scale

    def forward(self, x):
        return x + self.alpha * self.pe[:, :x.size(1)]