# ──────────────────────────────────────────────────────────────────────
#  helper
# ──────────────────────────────────────────────────────────────────────
from typing import Optional, Callable, List, Tuple
import torch
from torch import nn, Tensor
from torch.nn.functional import pad
from .positional_embeddings.sinusoidalpositionalencoding import SinusoidalPositionalEncoding

def _append(store: Optional[List[Tuple[str, Tensor]]], tag: str, attn: Tensor):
    if store is not None:                          # detach + move to CPU
        store.append((tag, attn.detach().cpu()))

# ──────────────────────────────────────────────────────────────────────
#  encoder layer that logs self-attention  (stage-1 & stage-3)
# ──────────────────────────────────────────────────────────────────────
class TransformerEncoderLayerWithLogging(nn.TransformerEncoderLayer):
    """Exactly the same parameters as the stock layer + two logging kwargs."""
    def forward(self,
                src: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                *,
                attn_store: Optional[List[Tuple[str, Tensor]]] = None,
                scope: str = "") -> Tensor:
        need = attn_store is not None
        if self.norm_first:
            x = self.norm1(src)
            attn_out, w = self.self_attn(x, x, x,
                                         attn_mask=src_mask,
                                         key_padding_mask=src_key_padding_mask,
                                         need_weights=need,
                                         average_attn_weights=False)
            _append(attn_store, f"{scope}_self", w)
            src = src + self.dropout1(attn_out)
            src = src + self.dropout2(self.linear2(
                       self.dropout(self.activation(self.linear1(self.norm2(src))))))
            return src
        # post-norm branch
        attn_out, w = self.self_attn(src, src, src,
                                     attn_mask=src_mask,
                                     key_padding_mask=src_key_padding_mask,
                                     need_weights=need,
                                     average_attn_weights=False)
        _append(attn_store, f"{scope}_self", w)
        src = self.norm1(src + self.dropout1(attn_out))
        src = self.norm2(src + self.dropout2(self.linear2(
                   self.dropout(self.activation(self.linear1(src))))))
        return src

# ──────────────────────────────────────────────────────────────────────
#  cross encoder layer  (unchanged parameters, extra logging kwargs)
# ──────────────────────────────────────────────────────────────────────
class TransformerEncoderLayerWithCrossAttention(nn.Module):
    # … constructor identical to the one you supplied …
    # (not repeated here; only the two forward_* methods are patched)

    def forward_self_attention(self, src, *, src_mask=None, src_key_padding_mask=None,
                               attn_store=None, scope=""):
        need = attn_store is not None
        if self.norm_first:
            y = self.norm1(src)
            out, w = self.self_attn(y, y, y,
                                    attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask,
                                    need_weights=need,
                                    average_attn_weights=False)
            _append(attn_store, f"{scope}_self", w)
            return src + self.dropout1(out)
        out, w = self.self_attn(src, src, src,
                                attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask,
                                need_weights=need,
                                average_attn_weights=False)
        _append(attn_store, f"{scope}_self", w)
        return self.norm1(src + self.dropout1(out))

    def forward_cross_attention(self, src, memory, *, memory_mask=None,
                                memory_key_padding_mask=None,
                                attn_store=None, scope="", direction="x1→x2"):
        need = attn_store is not None
        tag = f"{scope}_{direction}"
        if self.norm_first:
            y = self.norm2(src)
            out, w = self.cross_attn(y, memory, memory,
                                     attn_mask=memory_mask,
                                     key_padding_mask=memory_key_padding_mask,
                                     need_weights=need,
                                     average_attn_weights=False)
            _append(attn_store, tag, w)
            return src + self.dropout2(out)
        out, w = self.cross_attn(src, memory, memory,
                                 attn_mask=memory_mask,
                                 key_padding_mask=memory_key_padding_mask,
                                 need_weights=need,
                                 average_attn_weights=False)
        _append(attn_store, tag, w)
        return self.norm2(src + self.dropout2(out))

    # forward_feedforward stays unchanged

# ──────────────────────────────────────────────────────────────────────
#  backbone
# ──────────────────────────────────────────────────────────────────────
class CrossAttentionTransformer(nn.Module):
    def __init__(self, conf):
        super().__init__()
        # ---------- original constructor code, but use Logging layer ----------
        self.encode_layers = conf.num_layers
        self.cross_layers  = conf.cross_layers
        self.concat_layers = conf.concat_layers
        d1, d2, dC = (conf.attention.d_model_1,
                      conf.attention.d_model_2,
                      conf.attention.d_model_C)
        assert dC == d1 + d2, "d_model_C must equal d_model_1 + d_model_2"
        df1 = conf.attention.dim_feedforward_1
        df2 = conf.attention.dim_feedforward_2
        dfC = conf.attention.dim_feedforward_C
        nhead = conf.attention.num_attention_heads
        dropout = conf.attention.dropout
        act = nn.LeakyReLU(inplace=False)

        # positional encodings
        if conf.positional_embedding == "Sinusoidal":
            self.positional_embedding_1 = SinusoidalPositionalEncoding(d1)
            self.positional_embedding_2 = SinusoidalPositionalEncoding(d2)
        else:
            self.positional_embedding_1 = None
            self.positional_embedding_2 = None

        # stage-1 encoders
        self.encoder_1 = nn.ModuleList([
            TransformerEncoderLayerWithLogging(
                d_model=d1, nhead=nhead, dim_feedforward=df1,
                dropout=dropout, batch_first=True, activation=act)
            for _ in range(self.encode_layers)
        ])
        self.encoder_2 = nn.ModuleList([
            TransformerEncoderLayerWithLogging(
                d_model=d2, nhead=nhead, dim_feedforward=df2,
                dropout=dropout, batch_first=True, activation=act)
            for _ in range(self.encode_layers)
        ])

        # stage-2 cross
        self.encoder_1_cross_layers = nn.ModuleList([
            TransformerEncoderLayerWithCrossAttention(
                d_model=d1, nhead=nhead, dim_feedforward=df1,
                dropout=dropout, batch_first=True, activation=act)
            for _ in range(self.cross_layers)
        ])
        self.encoder_2_cross_layers = nn.ModuleList([
            TransformerEncoderLayerWithCrossAttention(
                d_model=d2, nhead=nhead, dim_feedforward=df2,
                dropout=dropout, batch_first=True, activation=act)
            for _ in range(self.cross_layers)
        ])

        # stage-3 concat
        self.concat_layers = nn.ModuleList([
            TransformerEncoderLayerWithLogging(
                d_model=dC, nhead=nhead, dim_feedforward=dfC,
                dropout=dropout, batch_first=True, activation=act)
            for _ in range(self.concat_layers)
        ])

        # everything else is byte-for-byte identical to your original ctor
        self.eval_token_1 = nn.Parameter(torch.rand(1, 1, d1))
        self.eval_token_2 = nn.Parameter(torch.rand(1, 1, d2))
        self.mask_token_1 = nn.Parameter(torch.rand(d1))
        self.mask_token_2 = nn.Parameter(torch.rand(d2))
        self.encoder_1_mask_rate = conf.encoder_1_mask_rate
        self.encoder_2_mask_rate = conf.encoder_2_mask_rate
        self.final_dropout = (nn.Dropout(conf.final_dropout.prob)
                              if conf.final_dropout.enable else None)

    # ------------------------------- forward ------------------------------- #
    def forward(self, x1:Tensor, x2:Tensor, *, attention_mask=None,
                series_len=None, token_mask=None, return_attn:bool=False):
        attn: Optional[List[Tuple[str, Tensor]]] = [] if return_attn else None
        B = x1.size(0)
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.bool)

        # positional encodings
        if self.positional_embedding_1 is not None:
            x1 = self.positional_embedding_1(x1, series_len=series_len)
            x2 = self.positional_embedding_2(x2, series_len=series_len)

        # random token masking
        if token_mask is not None:
            m1 = (torch.rand_like(token_mask.float()) <= self.encoder_1_mask_rate) & token_mask
            m2 = (torch.rand_like(token_mask.float()) <= self.encoder_2_mask_rate) & token_mask
            x1[m1] = self.mask_token_1
            x2[m2] = self.mask_token_2

        # add evaluation tokens
        x1 = torch.cat([x1, self.eval_token_1.expand(B, -1, -1)], 1)
        x2 = torch.cat([x2, self.eval_token_2.expand(B, -1, -1)], 1)
        if attention_mask is not None:
            attention_mask = pad(attention_mask, (0, 1), value=False)

        # ─── stage 1 ─────────────────────────────────────────────────────── #
        for i, layer in enumerate(self.encoder_1):
            x1 = layer(x1, src_key_padding_mask=attention_mask,
                       attn_store=attn, scope=f"enc1_L{i}")
        for i, layer in enumerate(self.encoder_2):
            x2 = layer(x2, src_key_padding_mask=attention_mask,
                       attn_store=attn, scope=f"enc2_L{i}")

        x1_residual = x1                                         # before cross

        # ─── stage 2 ─────────────────────────────────────────────────────── #
        for i in range(self.cross_layers):
            tag = f"cross_L{i}"
            x1 = self.encoder_1_cross_layers[i].forward_self_attention(
                     x1, src_key_padding_mask=attention_mask,
                     attn_store=attn, scope=f"{tag}_x1")
            x2 = self.encoder_2_cross_layers[i].forward_self_attention(
                     x2, src_key_padding_mask=attention_mask,
                     attn_store=attn, scope=f"{tag}_x2")

            x1 = self.encoder_1_cross_layers[i].forward_cross_attention(
                     x1, memory=x2, memory_key_padding_mask=attention_mask,
                     attn_store=attn, scope=tag, direction="x1→x2")
            x2 = self.encoder_2_cross_layers[i].forward_cross_attention(
                     x2, memory=x1, memory_key_padding_mask=attention_mask,
                     attn_store=attn, scope=tag, direction="x2→x1")

            x1 = self.encoder_1_cross_layers[i].forward_feedforward(x1)
            x2 = self.encoder_2_cross_layers[i].forward_feedforward(x2)

        # ─── stage 3 ─────────────────────────────────────────────────────── #
        x = torch.cat([x1, x2], -1)
        for i, layer in enumerate(self.concat_layers):
            x = layer(x, src_key_padding_mask=attention_mask,
                      attn_store=attn, scope=f"concat_L{i}")

        if self.final_dropout is not None:
            x = self.final_dropout(x)
            x1_residual = self.final_dropout(x1_residual)

        out = (x[:, :-1], x[:, -1:], x1_residual[:, :-1], x1_residual[:, -1:])
        return (*out, attn) if return_attn else out

    # inject_weights_into_encoder_1 is *unchanged* – parameter names match.