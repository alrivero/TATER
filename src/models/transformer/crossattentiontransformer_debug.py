# ────────────────────────────────────────────────────────────────────
#  helper to tag & store
# ────────────────────────────────────────────────────────────────────
from typing import List, Tuple, Optional
from typing import Optional, Callable
import torch
from torch import nn, Tensor
from torch.nn.functional import pad
from .positional_embeddings.sinusoidalpositionalencoding import SinusoidalPositionalEncoding

def _store_attention(backbone, tag: str, attn: Tensor):
    buf: Optional[List[Tuple[str, Tensor]]] = getattr(backbone, "_attn_buf", None)
    if buf is not None:
        buf.append((tag, attn.detach().cpu()))    # keep graph & GPU free

# ────────────────────────────────────────────────────────────────────
#  original cross-layer (only change: **unused to swallow extras)
# ────────────────────────────────────────────────────────────────────
class TransformerEncoderLayerWithCrossAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: Callable[[Tensor], Tensor] = nn.ReLU(),
                 layer_norm_eps: float = 1e-5,
                 batch_first: bool = True,
                 norm_first: bool = False,
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 **unused):          # ← new, accepts any extra kwarg
        super().__init__()
        kw = {"device": device, "dtype": dtype}
        self.self_attn  = nn.MultiheadAttention(d_model, nhead, dropout,
                                                batch_first=batch_first, bias=bias, **kw)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout,
                                                batch_first=batch_first, bias=bias, **kw)
        self.linear1 = nn.Linear(d_model, dim_feedforward, **kw)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **kw)
        self.dropout  = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **kw)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **kw)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **kw)
        self.activation = activation
        self.norm_first = norm_first

    # the three forward_* methods remain byte-for-byte identical …

# ────────────────────────────────────────────────────────────────────
#  backbone with _register_hooks
# ────────────────────────────────────────────────────────────────────
class CrossAttentionTransformer(nn.Module):
    def __init__(self, conf):
        super().__init__()
        # ---------------- same constructor body as your version ----------------
        # (only two tiny edits: use a helper to register hooks afterwards,
        #  and assert d_model_C = d1+d2 to avoid silent shape bugs)
        self.encode_layers = conf.num_layers
        self.cross_layers  = conf.cross_layers
        self.concat_layers = conf.concat_layers
        d1, d2, dC = (conf.attention.d_model_1,
                      conf.attention.d_model_2,
                      conf.attention.d_model_C)
        assert dC == d1 + d2, "d_model_C must equal d_model_1 + d_model_2"

        df1, df2, dfC = (conf.attention.dim_feedforward_1,
                         conf.attention.dim_feedforward_2,
                         conf.attention.dim_feedforward_C)
        nhead   = conf.attention.num_attention_heads
        dropout = conf.attention.dropout
        act     = nn.LeakyReLU(inplace=False)

        self.positional_embedding_1 = SinusoidalPositionalEncoding(d1) \
            if conf.positional_embedding == "Sinusoidal" else None
        self.positional_embedding_2 = SinusoidalPositionalEncoding(d2) \
            if conf.positional_embedding == "Sinusoidal" else None

        # stage-1 encoders (stock layer)
        self.encoder_1 = nn.ModuleList([
            nn.TransformerEncoderLayer(d1, nhead, df1, dropout,
                                       batch_first=True, activation=act)
            for _ in range(self.encode_layers)
        ])
        self.encoder_2 = nn.ModuleList([
            nn.TransformerEncoderLayer(d2, nhead, df2, dropout,
                                       batch_first=True, activation=act)
            for _ in range(self.encode_layers)
        ])

        # stage-2 cross (custom layer)
        self.encoder_1_cross_layers = nn.ModuleList([
            TransformerEncoderLayerWithCrossAttention(d1, nhead, df1, dropout,
                                                      batch_first=True,
                                                      activation=act)
            for _ in range(self.cross_layers)
        ])
        self.encoder_2_cross_layers = nn.ModuleList([
            TransformerEncoderLayerWithCrossAttention(d2, nhead, df2, dropout,
                                                      batch_first=True,
                                                      activation=act)
            for _ in range(self.cross_layers)
        ])

        # stage-3 concat encoder (stock layer)
        self.concat_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(dC, nhead, dfC, dropout,
                                       batch_first=True, activation=act)
            for _ in range(self.concat_layers)
        ])

        # tokens / masks  (unchanged)
        self.eval_token_1 = nn.Parameter(torch.rand(1,1,d1))
        self.eval_token_2 = nn.Parameter(torch.rand(1,1,d2))
        self.mask_token_1 = nn.Parameter(torch.rand(d1))
        self.mask_token_2 = nn.Parameter(torch.rand(d2))
        self.encoder_1_mask_rate = conf.encoder_1_mask_rate
        self.encoder_2_mask_rate = conf.encoder_2_mask_rate
        self.final_dropout = (nn.Dropout(conf.final_dropout.prob)
                              if conf.final_dropout.enable else None)

        # -------- register hooks once everything exists -------- #
        self._register_attention_hooks()

    # ------------------------------------------------------------------
    def _register_attention_hooks(self):
        """Attach forward-hooks to every MultiheadAttention we don’t own."""
        # stage-1 encoders
        for i, layer in enumerate(self.encoder_1):
            layer.self_attn.register_forward_hook(
                self._make_hook(f"enc1_L{i}_self"))
        for i, layer in enumerate(self.encoder_2):
            layer.self_attn.register_forward_hook(
                self._make_hook(f"enc2_L{i}_self"))
        # stage-3 concat
        for i, layer in enumerate(self.concat_layers):
            layer.self_attn.register_forward_hook(
                self._make_hook(f"concat_L{i}_self"))

    def _make_hook(self, tag):
        def hook(module, inp, out):
            # out is (attn_out, attn_weights) because need_weights=True by default
            if isinstance(out, tuple) and len(out) == 2:
                _store_attention(self, tag, out[1])
        return hook

    # ---------------------------- forward ---------------------------- #
    def forward(self, x1:Tensor, x2:Tensor, *,
                attention_mask=None, series_len=None, token_mask=None,
                return_attn: bool=False):

        self._attn_buf: Optional[List[Tuple[str, Tensor]]] = [] if return_attn else None
        B = x1.size(0)
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.bool)

        # positional encodings
        if self.positional_embedding_1 is not None:
            x1 = self.positional_embedding_1(x1, series_len=series_len)
            x2 = self.positional_embedding_2(x2, series_len=series_len)

        # random masking
        if token_mask is not None:
            m1 = (torch.rand_like(token_mask.float()) <= self.encoder_1_mask_rate) & token_mask
            m2 = (torch.rand_like(token_mask.float()) <= self.encoder_2_mask_rate) & token_mask
            x1[m1] = self.mask_token_1
            x2[m2] = self.mask_token_2

        # evaluation tokens
        x1 = torch.cat([x1, self.eval_token_1.expand(B, -1, -1)], 1)
        x2 = torch.cat([x2, self.eval_token_2.expand(B, -1, -1)], 1)
        if attention_mask is not None:
            attention_mask = pad(attention_mask, (0,1), value=False)

        # ── stage-1 (hooks grab their attn automatically) ──
        for layer in self.encoder_1:
            x1 = layer(x1, src_key_padding_mask=attention_mask)
        for layer in self.encoder_2:
            x2 = layer(x2, src_key_padding_mask=attention_mask)

        x1_residual = x1

        # ── stage-2 (we call custom layer; we tag manually) ──
        for i in range(self.cross_layers):
            tag = f"cross_L{i}"
            x1 = self.encoder_1_cross_layers[i].forward_self_attention(
                    x1, src_key_padding_mask=attention_mask,
                    attn_store=self._attn_buf, scope=f"{tag}_x1")
            x2 = self.encoder_2_cross_layers[i].forward_self_attention(
                    x2, src_key_padding_mask=attention_mask,
                    attn_store=self._attn_buf, scope=f"{tag}_x2")

            x1 = self.encoder_1_cross_layers[i].forward_cross_attention(
                    x1, memory=x2, memory_key_padding_mask=attention_mask,
                    attn_store=self._attn_buf, scope=tag, direction="x1→x2")
            x2 = self.encoder_2_cross_layers[i].forward_cross_attention(
                    x2, memory=x1, memory_key_padding_mask=attention_mask,
                    attn_store=self._attn_buf, scope=tag, direction="x2→x1")

            x1 = self.encoder_1_cross_layers[i].forward_feedforward(x1)
            x2 = self.encoder_2_cross_layers[i].forward_feedforward(x2)

        # ── stage-3 (hooks grab attn) ──
        x = torch.cat([x1, x2], -1)
        for layer in self.concat_layers:
            x = layer(x, src_key_padding_mask=attention_mask)

        if self.final_dropout is not None:
            x = self.final_dropout(x)
            x1_residual = self.final_dropout(x1_residual)

        outputs = (x[:, :-1], x[:, -1:], x1_residual[:, :-1], x1_residual[:, -1:])
        attn_out = self._attn_buf
        self._attn_buf = None                    # clear ref for next forward
        return (*outputs, attn_out) if return_attn else outputs

    # inject_weights_into_encoder_1  (unchanged – parameter names intact)