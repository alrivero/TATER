import torch
from torch import nn, Tensor
from torch.nn.functional import pad
import torch.nn.functional as F
from typing import Optional, Callable, List, Tuple, Union
from .positional_embeddings.sinusoidalpositionalencoding import SinusoidalPositionalEncoding

# ---------------------------------------------------------------- #
# 1) Drop-in subclass of TransformerEncoderLayer that adds
#    return_attn → (output, attn_maps) via Python fallback
# ---------------------------------------------------------------- #
class TransformerEncoderLayerWithMaps(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._attn_buffer: Optional[List[Tensor]] = None

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False
    ) -> Tensor:
        out, weights = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=(self._attn_buffer is not None),
            is_causal=is_causal
        )
        if self._attn_buffer is not None:
            self._attn_buffer.append(weights.detach().cpu())
        return self.dropout1(out)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        return_attn: bool = False
    ) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        if not return_attn:
            return super().forward(src, src_mask, src_key_padding_mask, is_causal)

        self._attn_buffer = []

        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype,
        )
        mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        x = src
        if self.norm_first:
            y = self.norm1(x)
            x = x + self._sa_block(y, mask, src_key_padding_mask, is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, mask, src_key_padding_mask, is_causal))
            x = self.norm2(x + self._ff_block(x))

        maps = self._attn_buffer
        self._attn_buffer = None
        return x, maps


# ---------------------------------------------------------------- #
# 2) Custom cross-attention encoder layer (unchanged)
# ---------------------------------------------------------------- #
class TransformerEncoderLayerWithCrossAttention(nn.Module):
    def __init__(
        self,
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
        **unused,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn  = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=batch_first, bias=bias, **factory_kwargs
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=batch_first, bias=bias, **factory_kwargs
        )
        self.linear1    = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout    = nn.Dropout(dropout)
        self.linear2    = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        self.norm1      = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2      = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3      = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1   = nn.Dropout(dropout)
        self.dropout2   = nn.Dropout(dropout)
        self.dropout3   = nn.Dropout(dropout)
        self.activation = activation
        self.norm_first = norm_first

    def forward_self_attention(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        *,
        attn_store: Optional[List[Tuple[str, Tensor]]] = None,
        scope: str = ""
    ) -> Tensor:
        need = attn_store is not None
        q = self.norm1(src) if self.norm_first else src
        out, w = self.self_attn(
            q, q, q,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=need,
            average_attn_weights=False
        )
        if attn_store is not None:
            attn_store.append((f"{scope}_self", w.detach().cpu()))
        return src + self.dropout1(out) if self.norm_first else self.norm1(src + self.dropout1(out))

    def forward_cross_attention(
        self,
        src: Tensor,
        memory: Tensor,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        *,
        attn_store: Optional[List[Tuple[str, Tensor]]] = None,
        scope: str = "",
        direction: str = "x1→x2"
    ) -> Tensor:
        need = attn_store is not None
        tag  = f"{scope}_{direction}"
        q = self.norm2(src) if self.norm_first else src
        out, w = self.cross_attn(
            q, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=need,
            average_attn_weights=False
        )
        if attn_store is not None:
            attn_store.append((tag, w.detach().cpu()))
        return src + self.dropout2(out) if self.norm_first else self.norm2(src + self.dropout2(out))

    def forward_feedforward(self, src: Tensor) -> Tensor:
        if self.norm_first:
            y  = self.norm3(src)
            ff = self.linear2(self.dropout(self.activation(self.linear1(y))))
            return src + self.dropout3(ff)
        ff = self.linear2(self.dropout(self.activation(self.linear1(src))))
        return self.norm3(src + self.dropout3(ff))


# ---------------------------------------------------------------- #
# 3) CrossAttentionTransformer integrating both subclasses
# ---------------------------------------------------------------- #
class CrossAttentionTransformer(nn.Module):
    def __init__(self, conf):
        super().__init__()

        self.encode_layers = conf.num_layers
        self.cross_layers  = conf.cross_layers
        self.concat_layers = conf.concat_layers
        d1 = conf.attention.d_model_1
        d2 = conf.attention.d_model_2
        dC = conf.attention.d_model_C
        assert dC == d1 + d2, "d_model_C must equal d_model_1 + d_model_2"
        df1, df2, dfC = (
            conf.attention.dim_feedforward_1,
            conf.attention.dim_feedforward_2,
            conf.attention.dim_feedforward_C
        )
        nhead   = conf.attention.num_attention_heads
        dropout = conf.attention.dropout
        act     = nn.LeakyReLU(inplace=False)

        # Positional embeddings
        if conf.positional_embedding == "Sinusoidal":
            self.positional_embedding_1 = SinusoidalPositionalEncoding(d1)
            self.positional_embedding_2 = SinusoidalPositionalEncoding(d2)
        else:
            self.positional_embedding_1 = None
            self.positional_embedding_2 = None

        # Stage 1: use our TransformerEncoderLayerWithMaps
        self.encoder_1 = nn.ModuleList([
            TransformerEncoderLayerWithMaps(
                d1, nhead,
                dim_feedforward=df1,
                dropout=dropout,
                batch_first=True,
                activation=act
            )
            for _ in range(self.encode_layers)
        ])
        self.encoder_2 = nn.ModuleList([
            TransformerEncoderLayerWithMaps(
                d2, nhead,
                dim_feedforward=df2,
                dropout=dropout,
                batch_first=True,
                activation=act
            )
            for _ in range(self.encode_layers)
        ])

        # Stage 2: custom cross-attention
        self.encoder_1_cross_layers = nn.ModuleList([
            TransformerEncoderLayerWithCrossAttention(
                d1, nhead,
                dim_feedforward=df1,
                dropout=dropout,
                activation=act,
                batch_first=True
            )
            for _ in range(self.cross_layers)
        ])
        self.encoder_2_cross_layers = nn.ModuleList([
            TransformerEncoderLayerWithCrossAttention(
                d2, nhead,
                dim_feedforward=df2,
                dropout=dropout,
                activation=act,
                batch_first=True
            )
            for _ in range(self.cross_layers)
        ])

        # Stage 3: again TransformerEncoderLayerWithMaps
        self.concat_layers = nn.ModuleList([
            TransformerEncoderLayerWithMaps(
                dC, nhead,
                dim_feedforward=dfC,
                dropout=dropout,
                batch_first=True,
                activation=act
            )
            for _ in range(self.concat_layers)
        ])

        # Eval & mask tokens
        self.eval_token_1        = nn.Parameter(torch.rand(1,1,d1))
        self.eval_token_2        = nn.Parameter(torch.rand(1,1,d2))
        self.mask_token_1        = nn.Parameter(torch.rand(d1))
        self.mask_token_2        = nn.Parameter(torch.rand(d2))
        self.encoder_1_mask_rate = conf.encoder_1_mask_rate
        self.encoder_2_mask_rate = conf.encoder_2_mask_rate
        self.final_dropout       = nn.Dropout(conf.final_dropout.prob) if conf.final_dropout.enable else None

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        series_len: Optional[int] = None,
        token_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False
    ):
        self._attn_buf = [] if return_attn else None
        B = x1.size(0)

        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.bool)

        # positional embeddings
        if self.positional_embedding_1 is not None:
            x1 = self.positional_embedding_1(x1, series_len=series_len)
            x2 = self.positional_embedding_2(x2, series_len=series_len)

        # token masking
        if token_mask is not None:
            m1 = (torch.rand_like(token_mask.float()) <= self.encoder_1_mask_rate) & token_mask
            m2 = (torch.rand_like(token_mask.float()) <= self.encoder_2_mask_rate) & token_mask
            x1[m1] = self.mask_token_1
            x2[m2] = self.mask_token_2

        # append eval tokens
        x1 = torch.cat([x1, self.eval_token_1.expand(B, -1, -1)], dim=1)
        x2 = torch.cat([x2, self.eval_token_2.expand(B, -1, -1)], dim=1)
        if attention_mask is not None:
            attention_mask = pad(attention_mask, (0, 1), value=False)

        # Stage 1
        for i, layer in enumerate(self.encoder_1):
            if return_attn:
                x1, maps = layer(
                    x1,
                    src_mask=None,
                    src_key_padding_mask=attention_mask,
                    is_causal=False,
                    return_attn=True
                )
                for w in maps:
                    self._attn_buf.append((f"encoder_1.{i}.self_attn", w))
            else:
                x1 = layer(
                    x1,
                    src_mask=None,
                    src_key_padding_mask=attention_mask,
                    is_causal=False
                )

        for i, layer in enumerate(self.encoder_2):
            if return_attn:
                x2, maps = layer(
                    x2,
                    src_mask=None,
                    src_key_padding_mask=attention_mask,
                    is_causal=False,
                    return_attn=True
                )
                for w in maps:
                    self._attn_buf.append((f"encoder_2.{i}.self_attn", w))
            else:
                x2 = layer(
                    x2,
                    src_mask=None,
                    src_key_padding_mask=attention_mask,
                    is_causal=False
                )

        x1_residual = x1

        # Stage 2
        for i in range(self.cross_layers):
            tag = f"cross_L{i}"
            x1 = self.encoder_1_cross_layers[i].forward_self_attention(
                x1, src_key_padding_mask=attention_mask,
                attn_store=self._attn_buf, scope=f"{tag}_x1"
            )
            x2 = self.encoder_2_cross_layers[i].forward_self_attention(
                x2, src_key_padding_mask=attention_mask,
                attn_store=self._attn_buf, scope=f"{tag}_x2"
            )
            x1 = self.encoder_1_cross_layers[i].forward_cross_attention(
                x1, memory=x2, memory_key_padding_mask=attention_mask,
                attn_store=self._attn_buf, scope=tag, direction="x1→x2"
            )
            x2 = self.encoder_2_cross_layers[i].forward_cross_attention(
                x2, memory=x1, memory_key_padding_mask=attention_mask,
                attn_store=self._attn_buf, scope=tag, direction="x2→x1"
            )
            x1 = self.encoder_1_cross_layers[i].forward_feedforward(x1)
            x2 = self.encoder_2_cross_layers[i].forward_feedforward(x2)

        # Stage 3
        x = torch.cat([x1, x2], dim=-1)
        for i, layer in enumerate(self.concat_layers):
            if return_attn:
                x, maps = layer(
                    x,
                    src_mask=None,
                    src_key_padding_mask=attention_mask,
                    is_causal=False,
                    return_attn=True
                )
                for w in maps:
                    self._attn_buf.append((f"concat_layers.{i}.self_attn", w))
            else:
                x = layer(
                    x,
                    src_mask=None,
                    src_key_padding_mask=attention_mask,
                    is_causal=False
                )

        if self.final_dropout is not None:
            x = self.final_dropout(x)
            x1_residual = self.final_dropout(x1_residual)

        outputs = (x[:, :-1], x[:, -1:], x1_residual[:, :-1], x1_residual[:, -1:])
        attn_out = self._attn_buf
        self._attn_buf = None
        return (*outputs, attn_out) if return_attn else outputs

    def inject_weights_into_encoder_1(self, weight_dict: dict):
        num_layers = len(self.encoder_1)
        for i in range(num_layers):
            self.encoder_1[i].load_state_dict({
                "self_attn.in_proj_weight":  weight_dict[f"{i}.self_attn.in_proj_weight"],
                "self_attn.in_proj_bias":    weight_dict[f"{i}.self_attn.in_proj_bias"],
                "self_attn.out_proj.weight": weight_dict[f"{i}.self_attn.out_proj.weight"],
                "self_attn.out_proj.bias":   weight_dict[f"{i}.self_attn.out_proj.bias"],
                "linear1.weight":            weight_dict[f"{i}.linear1.weight"],
                "linear1.bias":              weight_dict[f"{i}.linear1.bias"],
                "linear2.weight":            weight_dict[f"{i}.linear2.weight"],
                "linear2.bias":              weight_dict[f"{i}.linear2.bias"],
                "norm1.weight":              weight_dict[f"{i}.norm1.weight"],
                "norm1.bias":                weight_dict[f"{i}.norm1.bias"],
                "norm2.weight":              weight_dict[f"{i}.norm2.weight"],
                "norm2.bias":                weight_dict[f"{i}.norm2.bias"],
            })