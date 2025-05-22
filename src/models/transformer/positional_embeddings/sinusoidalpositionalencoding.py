# Based off code from https://github.com/tatp22/multidim-positional-encoding
import math
import torch
import numpy as np
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, init_std=1e-5):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))          # (1, max_len, D)

        # learnable per-dimension scale, tiny non-zero init
        self.alpha = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.alpha, mean=0.0, std=init_std)  # ≈ identity

    def forward(self, x):                                    # x: (B, S, D)
        return x + self.alpha * self.pe[:, :x.size(1)]