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

    def _init_identity(self):
        # For each TransformerEncoderLayer, zero out its
        # self-attention + feed-forward so that only the
        # residual remains, and set LayerNorm → y = x
        for blk in self.attention_blocks:
            # zero-out the QKV + output proj weights & biases
            blk.self_attn.in_proj_weight.data.zero_()
            blk.self_attn.in_proj_bias.data.zero_()
            blk.self_attn.out_proj.weight.data.zero_()
            blk.self_attn.out_proj.bias.data.zero_()
            # zero-out the feed-forward MLP
            blk.linear1.weight.data.zero_()
            blk.linear1.bias.data.zero_()
            blk.linear2.weight.data.zero_()
            blk.linear2.bias.data.zero_()
            # layer norms as identity: y = 1·x + 0
            blk.norm1.weight.data.fill_(1.0)
            blk.norm1.bias.data.zero_()
            blk.norm2.weight.data.fill_(1.0)
            blk.norm2.bias.data.zero_()

            # replace LayerNorms with Identity → no mean/var shift
            blk.norm1 = nn.Identity()
            blk.norm2 = nn.Identity()

    # ------------------------------------------------------------------
    def forward(self,
                x: torch.Tensor,              # (B, S, D)
                attention_mask: torch.Tensor, # (B, S)  True ⇒ masked
                series_len    = None,
                token_mask    = None):
        """
        Returns
        -------
        seq_out : (B, S, D)  – per‑token embeddings (special tokens removed)
        None    : placeholder so call‑sites remain compatible
        """
        B, S, D = x.shape

        # — positional encoding —
        if False: # self.positional_embedding is not None:
            x = self.positional_embedding(x, series_len=series_len)

        # — optional random token masking —
        if token_mask is not None:
            x[token_mask] = self.mask_token

        # # — prepend <START> and append <STOP> —
        # start_tok = self.start_token.expand(B, 1, -1)
        # stop_tok  = self.stop_token .expand(B, 1, -1)
        # x = torch.cat((start_tok, x, stop_tok), dim=-2)  # (B, S+2, D)

        # pad attention mask for the two new tokens
        # attention_mask = pad(attention_mask, (1, 1), value=False)

        # — transformer backbone —
        for blk in self.attention_blocks:
            x = blk(x, src_key_padding_mask=attention_mask)

        # # — optional dropout (only on ordinary tokens) —
        # if self.final_dropout is not None:
        #     x[:, 1:S+1, :] = self.final_dropout(x[:, 1:S+1, :])

        # remove special tokens from output
        seq_out = x # [:, 1:S+1, :]  # drop <START> and <STOP>
        return seq_out, None