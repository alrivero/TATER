import torch
from torch import nn, Tensor
from torch.nn.functional import pad
from typing import Optional, Callable, List, Tuple, Any
from .positional_embeddings.sinusoidalpositionalencoding import SinusoidalPositionalEncoding

# --------------------------- helper -------------------------------- #
def _append(store: Optional[List[Tuple[str, Tensor]]],
            tag: str,
            attn: Tensor):
    """Detach + send to CPU so we don’t keep the graph or GPU memory."""
    if store is not None:
        store.append((tag, attn.detach().cpu()))

# -------------------- self-attention encoder layer ----------------- #
class SelfAttnEncoderLayer(nn.Module):
    """
    Same API as nn.TransformerEncoderLayer but can push attention maps
    into a shared list.
    """
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: Callable[[Tensor], Tensor] = nn.ReLU(),
                 batch_first: bool = True,
                 norm_first: bool = False,
                 layer_norm_eps: float = 1e-5,
                 bias: bool = True,
                 device=None,
                 dtype=None):
        super().__init__()
        kw = dict(device=device, dtype=dtype)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout,
                                               batch_first=batch_first, bias=bias, **kw)
        self.linear1   = nn.Linear(d_model, dim_feedforward, **kw)
        self.linear2   = nn.Linear(dim_feedforward, d_model, **kw)
        self.dropout   = nn.Dropout(dropout)
        self.dropout1  = nn.Dropout(dropout)
        self.dropout2  = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **kw)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **kw)
        self.activation = activation
        self.norm_first = norm_first

    def forward(self,
                src: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                *,
                attn_store: Optional[List],
                scope: str) -> Tensor:
        collect = attn_store is not None
        if self.norm_first:
            y = self.norm1(src)
            attn_out, w = self.self_attn(y, y, y,
                                         attn_mask=src_mask,
                                         key_padding_mask=src_key_padding_mask,
                                         need_weights=collect,
                                         average_attn_weights=False)
            _append(attn_store, f"{scope}_self", w)
            src = src + self.dropout1(attn_out)
            src = src + self.dropout2(self.activation(self.linear1(self.norm2(src))))
            return src
        # post-norm
        attn_out, w = self.self_attn(src, src, src,
                                     attn_mask=src_mask,
                                     key_padding_mask=src_key_padding_mask,
                                     need_weights=collect,
                                     average_attn_weights=False)
        _append(attn_store, f"{scope}_self", w)
        src = self.norm1(src + self.dropout1(attn_out))
        src = self.norm2(src + self.dropout2(self.activation(self.linear1(src))))
        return src

# ------------- cross-layer (unchanged except tagging) -------------- #
class TransformerEncoderLayerWithCrossAttention(nn.Module):
    """
    Adds `scope` + `attn_store` so we can tag every map.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        activation = kwargs.pop("activation", nn.ReLU())
        factory_kwargs = {"device": kwargs.get("device", None),
                          "dtype": kwargs.get("dtype", None)}
        d_model = args[0]; nhead = args[1]; dropout = kwargs.get("dropout", 0.1)
        dim_ff = kwargs.get("dim_feedforward", 2048)
        batch_first = kwargs.get("batch_first", True)
        bias = kwargs.get("bias", True)
        norm_first = kwargs.get("norm_first", False)
        eps = kwargs.get("layer_norm_eps", 1e-5)

        self.self_attn  = nn.MultiheadAttention(d_model, nhead, dropout,
                                                batch_first=batch_first, bias=bias,
                                                **factory_kwargs)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout,
                                                batch_first=batch_first, bias=bias,
                                                **factory_kwargs)
        self.linear1  = nn.Linear(d_model, dim_ff, **factory_kwargs)
        self.linear2  = nn.Linear(dim_ff,  d_model, **factory_kwargs)
        self.dropout  = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=eps, **factory_kwargs)
        self.activation = activation
        self.norm_first = norm_first

    # ─────────── wrappers with tagging ─────────── #
    def _self_attn(self, tag, q, k, v, need, mask, pad, store):
        out, w = self.self_attn(q, k, v,
                                attn_mask=mask,
                                key_padding_mask=pad,
                                need_weights=need,
                                average_attn_weights=False)
        _append(store, tag, w)
        return out

    def _cross_attn(self, tag, q, k, v, need, mask, pad, store):
        out, w = self.cross_attn(q, k, v,
                                 attn_mask=mask,
                                 key_padding_mask=pad,
                                 need_weights=need,
                                 average_attn_weights=False)
        _append(store, tag, w)
        return out

    def forward_self_attention(self, src, *, src_mask=None, src_key_padding_mask=None,
                               attn_store=None, scope=""):
        need = attn_store is not None
        if self.norm_first:
            y = self.norm1(src)
            attn_out = self._self_attn(f"{scope}_self", y, y, y,
                                       need, src_mask, src_key_padding_mask, attn_store)
            return src + self.dropout1(attn_out)
        attn_out = self._self_attn(f"{scope}_self", src, src, src,
                                   need, src_mask, src_key_padding_mask, attn_store)
        return self.norm1(src + self.dropout1(attn_out))

    def forward_cross_attention(self, src, memory, *, memory_mask=None,
                                memory_key_padding_mask=None,
                                attn_store=None, scope="", direction="x1→x2"):
        need = attn_store is not None
        tag = f"{scope}_{direction}"
        if self.norm_first:
            y = self.norm2(src)
            attn_out = self._cross_attn(tag, y, memory, memory,
                                        need, memory_mask, memory_key_padding_mask,
                                        attn_store)
            return src + self.dropout2(attn_out)
        attn_out = self._cross_attn(tag, src, memory, memory,
                                    need, memory_mask, memory_key_padding_mask,
                                    attn_store)
        return self.norm2(src + self.dropout2(attn_out))

    def forward_feedforward(self, src):
        if self.norm_first:
            y  = self.norm3(src)
            ff = self.linear2(self.dropout(self.activation(self.linear1(y))))
            return src + self.dropout3(ff)
        ff = self.linear2(self.dropout(self.activation(self.linear1(src))))
        return self.norm3(src + self.dropout3(ff))

# ------------------ the full transformer backbone ------------------ #
class CrossAttentionTransformer(nn.Module):
    def __init__(self, conf):
        super().__init__()
        # params
        self.encode_layers  = conf.num_layers
        self.cross_layers   = conf.cross_layers
        self.concat_layers  = conf.concat_layers
        d1 = conf.attention.d_model_1
        d2 = conf.attention.d_model_2
        dC = conf.attention.d_model_C
        assert dC == d1 + d2, "d_model_C must equal d_model_1 + d_model_2"
        hf = conf.attention.num_attention_heads
        dp = conf.attention.dropout
        act = nn.LeakyReLU(inplace=False)

        # pos-emb
        if conf.positional_embedding == "Sinusoidal":
            PE = SinusoidalPositionalEncoding
            self.pos1, self.pos2 = PE(d1), PE(d2)
        else:
            self.pos1 = self.pos2 = None

        # stage 1 encoders (now SelfAttnEncoderLayer)
        self.enc1 = nn.ModuleList([SelfAttnEncoderLayer(d1, hf,
                                                        conf.attention.dim_feedforward_1,
                                                        dp, act)
                                   for _ in range(self.encode_layers)])
        self.enc2 = nn.ModuleList([SelfAttnEncoderLayer(d2, hf,
                                                        conf.attention.dim_feedforward_2,
                                                        dp, act)
                                   for _ in range(self.encode_layers)])

        # stage 2 cross
        self.cross1 = nn.ModuleList([TransformerEncoderLayerWithCrossAttention(
                                        d1, hf, conf.attention.dim_feedforward_1,
                                        dp, activation=act)
                                     for _ in range(self.cross_layers)])
        self.cross2 = nn.ModuleList([TransformerEncoderLayerWithCrossAttention(
                                        d2, hf, conf.attention.dim_feedforward_2,
                                        dp, activation=act)
                                     for _ in range(self.cross_layers)])

        # stage 3 concat
        self.concat = nn.ModuleList([SelfAttnEncoderLayer(dC, hf,
                                                          conf.attention.dim_feedforward_C,
                                                          dp, act)
                                     for _ in range(self.concat_layers)])

        # tokens & masks
        self.eval_tok1 = nn.Parameter(torch.randn(1,1,d1))
        self.eval_tok2 = nn.Parameter(torch.randn(1,1,d2))
        self.mask_tok1 = nn.Parameter(torch.randn(d1))
        self.mask_tok2 = nn.Parameter(torch.randn(d2))
        self.mask_rate1 = conf.encoder_1_mask_rate
        self.mask_rate2 = conf.encoder_2_mask_rate
        self.final_dropout = (nn.Dropout(conf.final_dropout.prob)
                              if conf.final_dropout.enable else None)

    # --------------------------- forward --------------------------- #
    def forward(self, x1:Tensor, x2:Tensor,
                attention_mask:Optional[Tensor]=None,
                series_len:Optional[int]=None,
                token_mask:Optional[Tensor]=None,
                return_attn:bool=False):

        attn: Optional[List[Tuple[str,Tensor]]] = [] if return_attn else None
        B = x1.size(0)
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.bool)

        # pos-emb
        if self.pos1 is not None:
            x1 = self.pos1(x1, series_len=series_len)
            x2 = self.pos2(x2, series_len=series_len)

        # random masking
        if token_mask is not None:
            m1 = (torch.rand_like(token_mask.float()) <= self.mask_rate1) & token_mask
            m2 = (torch.rand_like(token_mask.float()) <= self.mask_rate2) & token_mask
            x1[m1] = self.mask_tok1
            x2[m2] = self.mask_tok2

        # eval tokens
        x1 = torch.cat([x1, self.eval_tok1.expand(B,-1,-1)], 1)
        x2 = torch.cat([x2, self.eval_tok2.expand(B,-1,-1)], 1)
        if attention_mask is not None:
            attention_mask = pad(attention_mask, (0,1), value=False)

        # ---------------- stage 1 ---------------- #
        for l, layer in enumerate(self.enc1):
            x1 = layer(x1, src_key_padding_mask=attention_mask,
                       attn_store=attn, scope=f"enc1_L{l}")
        for l, layer in enumerate(self.enc2):
            x2 = layer(x2, src_key_padding_mask=attention_mask,
                       attn_store=attn, scope=f"enc2_L{l}")

        x1_residual = x1  # before any cross info

        # ---------------- stage 2 ---------------- #
        for i in range(self.cross_layers):
            tag = f"cross_L{i}"
            x1 = self.cross1[i].forward_self_attention(
                    x1, src_key_padding_mask=attention_mask,
                    attn_store=attn, scope=f"{tag}_x1")
            x2 = self.cross2[i].forward_self_attention(
                    x2, src_key_padding_mask=attention_mask,
                    attn_store=attn, scope=f"{tag}_x2")

            x1 = self.cross1[i].forward_cross_attention(
                    x1, memory=x2, memory_key_padding_mask=attention_mask,
                    attn_store=attn, scope=tag, direction="x1→x2")
            x2 = self.cross2[i].forward_cross_attention(
                    x2, memory=x1, memory_key_padding_mask=attention_mask,
                    attn_store=attn, scope=tag, direction="x2→x1")

            x1 = self.cross1[i].forward_feedforward(x1)
            x2 = self.cross2[i].forward_feedforward(x2)

        # ---------------- stage 3 ---------------- #
        x = torch.cat([x1, x2], -1)
        for l, layer in enumerate(self.concat):
            x = layer(x, src_key_padding_mask=attention_mask,
                      attn_store=attn, scope=f"concat_L{l}")

        if self.final_dropout is not None:
            x = self.final_dropout(x)
            x1_residual = self.final_dropout(x1_residual)

        out = (x[:, :-1], x[:, -1:], x1_residual[:, :-1], x1_residual[:, -1:])
        return (*out, attn) if return_attn else out