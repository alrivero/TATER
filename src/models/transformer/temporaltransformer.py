import torch
from torch import nn
from torch.nn.functional import pad
from .positional_embeddings.sinusoidalpositionalencoding import SinusoidalPositionalEncoding

class TemporalTransformer(nn.Module):
    """
    Transformer with learnable <START> and <STOP> tokens.
    No <EVAL> token is used; the second return value is always None.
    """
    def __init__(self, conf):
        super().__init__()

        # ── positional encoding ───────────────────────────────────────
        if conf.positional_embedding == "Sinusoidal":
            self.positional_embedding = SinusoidalPositionalEncoding(
                conf.attention.hidden_size
            )
        else:
            self.positional_embedding = None

        # ── transformer encoder blocks ────────────────────────────────
        self.num_layers  = conf.num_layers
        self.hidden_size = conf.attention.hidden_size

        self.attention_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model          = conf.attention.hidden_size,
                dim_feedforward  = conf.attention.hidden_size_2,
                nhead            = conf.attention.num_attention_heads,
                dropout          = 0.0,
                layer_norm_eps   = conf.attention.layer_norm_eps,
                batch_first      = True,
                norm_first      = True,
                activation       = nn.LeakyReLU(0.01, inplace=False),
            )
            for _ in range(self.num_layers)
        ])

        # ── learnable special tokens ──────────────────────────────────
        self.start_token = nn.Parameter(torch.randn(self.hidden_size))
        self.stop_token  = nn.Parameter(torch.randn(self.hidden_size))

        # ── output projection for sequence tokens (optional use) ─────
        self.res_out = nn.Linear(self.hidden_size, self.hidden_size)

        # ── optional dropout after backbone ───────────────────────────
        if conf.final_dropout.enable:
            self.final_dropout = nn.Dropout(p=conf.final_dropout.prob)
        else:
            self.final_dropout = None

        # ── masking token for MLM‑style corruption ───────────────────
        self.mask_token = nn.Parameter(torch.randn(self.hidden_size))

        self._init_identity()

    def _init_identity(self, epsilon: float = 1e-6):
        # ── 1) Make your final projection head ≈ identity ─────────────────
        #   (so res_out(x) ≈ x + O(epsilon) but bias is trainable)
        nn.init.eye_(self.res_out.weight)
        self.res_out.bias.data.normal_(mean=0.0, std=epsilon)

        for blk in self.attention_blocks:
            # ── 2) Turn off every dropout in TransformerEncoderLayer ────────
            for attr in ("dropout1", "dropout2", "activation_dropout"):
                if hasattr(blk, attr):
                    getattr(blk, attr).p = 0.0

            # ── 3) Self-attention: tiny random weights & biases ───────────
            blk.self_attn.in_proj_weight.data.normal_(mean=0.0, std=epsilon)
            blk.self_attn.in_proj_bias .data.normal_(mean=0.0, std=epsilon)
            blk.self_attn.out_proj.weight.data.normal_(mean=0.0, std=epsilon)
            blk.self_attn.out_proj.bias  .data.normal_(mean=0.0, std=epsilon)

            # ── 4) Feed-forward MLP: tiny random weights & biases ─────────
            blk.linear1.weight.data.normal_(mean=0.0, std=epsilon)
            blk.linear1.bias  .data.normal_(mean=0.0, std=epsilon)
            blk.linear2.weight.data.normal_(mean=0.0, std=epsilon)
            blk.linear2.bias  .data.normal_(mean=0.0, std=epsilon)

            # ── 5) LayerNorms → exact identity but trainable if you want to unfreeze later
            blk.norm1.weight.data.fill_(1.0)
            blk.norm1.bias  .data.zero_()
            blk.norm2.weight.data.fill_(1.0)
            blk.norm2.bias  .data.zero_()

    # ------------------------------------------------------------------
    def forward(self,
                x: torch.Tensor,              # (B, S, D)
                attention_mask: torch.Tensor, # (B, S)  True ⇒ masked
                series_len=None,
                token_mask=None):
        B, S, D = x.shape

        # 1) Positional encoding
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)

        # 2) Token-level corruption
        if token_mask is not None:
            x[token_mask] = self.mask_token

        # 3) Add <START> and <STOP>
        start_tok = self.start_token.unsqueeze(0).expand(B, 1, D)
        stop_tok  = self.stop_token .unsqueeze(0).expand(B, 1, D)
        x = torch.cat([start_tok, x, stop_tok], dim=1)  # (B, S+2, D)

        # 4) Pad the padding-mask
        attention_mask = pad(attention_mask, (1,1), value=False)  # (B, S+2)

        # 5) Backbone
        for blk in self.attention_blocks:
            x = blk(x, src_key_padding_mask=attention_mask)

        # 6) Slice out the original tokens
        core = x[:, 1 : S+1, :]  # (B, S, D)

        # # 7) Optional dropout + projection
        # if self.final_dropout is not None:
        #     core = self.final_dropout(core)
        # core = self.res_out(core)

        return core, None