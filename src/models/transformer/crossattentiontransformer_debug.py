import torch
from torch import nn, Tensor
from torch.nn.functional import pad
from typing import Optional, Callable, List, Tuple
from .positional_embeddings.sinusoidalpositionalencoding import SinusoidalPositionalEncoding

def _store_attention(backbone, tag: str, attn: Tensor):
    """
    Safely append (tag, attn.cpu()) to backbone._attn_buf if it exists as a list.
    """
    buf = getattr(backbone, "_attn_buf", None)
    if not isinstance(buf, list):
        return               # nothing to do when return_attn=False
    buf.append((tag, attn.detach().cpu()))

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
        **unused,                # swallow any extra kwargs
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            **factory_kwargs,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            **factory_kwargs,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation
        self.norm_first = norm_first

    def forward_self_attention(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        *,
        attn_store: Optional[List[Tuple[str,Tensor]]] = None,
        scope: str = ""
    ) -> Tensor:
        need = attn_store is not None
        if self.norm_first:
            y = self.norm1(src)
            out, w = self.self_attn(
                query=y, key=y, value=y,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=need,
                average_attn_weights=False,
            )
            _store_attention(self, f"{scope}_self", w)
            return src + self.dropout1(out)
        out, w = self.self_attn(
            query=src, key=src, value=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=need,
            average_attn_weights=False,
        )
        _store_attention(self, f"{scope}_self", w)
        return self.norm1(src + self.dropout1(out))

    def forward_cross_attention(
        self,
        src: Tensor,
        memory: Tensor,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        *,
        attn_store: Optional[List[Tuple[str,Tensor]]] = None,
        scope: str = "",
        direction: str = "x1→x2"
    ) -> Tensor:
        need = attn_store is not None
        tag = f"{scope}_{direction}"
        if self.norm_first:
            y = self.norm2(src)
            out, w = self.cross_attn(
                query=y, key=memory, value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                need_weights=need,
                average_attn_weights=False,
            )
            _store_attention(self, tag, w)
            return src + self.dropout2(out)
        out, w = self.cross_attn(
            query=src, key=memory, value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=need,
            average_attn_weights=False,
        )
        _store_attention(self, tag, w)
        return self.norm2(src + self.dropout2(out))

    def forward_feedforward(self, src: Tensor) -> Tensor:
        if self.norm_first:
            y = self.norm3(src)
            ff = self.linear2(self.dropout(self.activation(self.linear1(y))))
            return src + self.dropout3(ff)
        ff = self.linear2(self.dropout(self.activation(self.linear1(src))))
        return self.norm3(src + self.dropout3(ff))


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
        df1 = conf.attention.dim_feedforward_1
        df2 = conf.attention.dim_feedforward_2
        dfC = conf.attention.dim_feedforward_C
        nhead   = conf.attention.num_attention_heads
        dropout = conf.attention.dropout
        act     = nn.LeakyReLU(inplace=False)

        if conf.positional_embedding == "Sinusoidal":
            self.positional_embedding_1 = SinusoidalPositionalEncoding(d1)
            self.positional_embedding_2 = SinusoidalPositionalEncoding(d2)
        else:
            self.positional_embedding_1 = None
            self.positional_embedding_2 = None

        # Stage 1: stock encoder layers
        self.encoder_1 = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d1, nhead=nhead,
                dim_feedforward=df1, dropout=dropout,
                batch_first=True, activation=act
            ) for _ in range(self.encode_layers)
        ])
        self.encoder_2 = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d2, nhead=nhead,
                dim_feedforward=df2, dropout=dropout,
                batch_first=True, activation=act
            ) for _ in range(self.encode_layers)
        ])

        # Stage 2: custom cross layers
        self.encoder_1_cross_layers = nn.ModuleList([
            TransformerEncoderLayerWithCrossAttention(
                d_model=d1, nhead=nhead,
                dim_feedforward=df1, dropout=dropout,
                batch_first=True, activation=act
            ) for _ in range(self.cross_layers)
        ])
        self.encoder_2_cross_layers = nn.ModuleList([
            TransformerEncoderLayerWithCrossAttention(
                d_model=d2, nhead=nhead,
                dim_feedforward=df2, dropout=dropout,
                batch_first=True, activation=act
            ) for _ in range(self.cross_layers)
        ])

        # Stage 3: stock concat layers
        self.concat_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dC, nhead=nhead,
                dim_feedforward=dfC, dropout=dropout,
                batch_first=True, activation=act
            ) for _ in range(self.concat_layers)
        ])

        # tokens & masks
        self.eval_token_1 = nn.Parameter(torch.rand(1, 1, d1))
        self.eval_token_2 = nn.Parameter(torch.rand(1, 1, d2))
        self.mask_token_1 = nn.Parameter(torch.rand(d1))
        self.mask_token_2 = nn.Parameter(torch.rand(d2))
        self.encoder_1_mask_rate = conf.encoder_1_mask_rate
        self.encoder_2_mask_rate = conf.encoder_2_mask_rate
        self.final_dropout = (nn.Dropout(conf.final_dropout.prob)
                              if conf.final_dropout.enable else None)

        # register forward‐hooks on .self_attn of stage-1 and stage-3
        self._register_attention_hooks()

    def _register_attention_hooks(self):
        for i, layer in enumerate(self.encoder_1):
            layer.self_attn.register_forward_hook(
                self._make_hook(f"enc1_L{i}_self"))
        for i, layer in enumerate(self.encoder_2):
            layer.self_attn.register_forward_hook(
                self._make_hook(f"enc2_L{i}_self"))
        for i, layer in enumerate(self.concat_layers):
            layer.self_attn.register_forward_hook(
                self._make_hook(f"concat_L{i}_self"))

    def _make_hook(self, tag: str):
        def hook(module, inp, out):
            # out == (attn_output, attn_weights)
            if isinstance(out, tuple) and len(out) == 2:
                _store_attention(self, tag, out[1])
        return hook

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        series_len: Optional[int] = None,
        token_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False
    ):
        # prepare buffer
        self._attn_buf: Optional[List[Tuple[str, Tensor]]] = [] if return_attn else None

        B = x1.size(0)
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.bool)

        if self.positional_embedding_1 is not None:
            x1 = self.positional_embedding_1(x1, series_len=series_len)
            x2 = self.positional_embedding_2(x2, series_len=series_len)

        if token_mask is not None:
            m1 = (torch.rand_like(token_mask.float()) <= self.encoder_1_mask_rate) & token_mask
            m2 = (torch.rand_like(token_mask.float()) <= self.encoder_2_mask_rate) & token_mask
            x1[m1] = self.mask_token_1
            x2[m2] = self.mask_token_2

        x1 = torch.cat([x1, self.eval_token_1.expand(B, -1, -1)], dim=1)
        x2 = torch.cat([x2, self.eval_token_2.expand(B, -1, -1)], dim=1)
        if attention_mask is not None:
            attention_mask = pad(attention_mask, (0, 1), mode="constant", value=False)

        # ── Stage 1 ──
        for layer in self.encoder_1:
            x1 = layer(x1, src_key_padding_mask=attention_mask)
        for layer in self.encoder_2:
            x2 = layer(x2, src_key_padding_mask=attention_mask)

        x1_residual = x1

        # ── Stage 2 ──
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

        # ── Stage 3 ──
        x = torch.cat([x1, x2], dim=-1)
        for layer in self.concat_layers:
            x = layer(x, src_key_padding_mask=attention_mask)

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
                "self_attn.in_proj_weight": weight_dict[f"{i}.self_attn.in_proj_weight"],
                "self_attn.in_proj_bias":   weight_dict[f"{i}.self_attn.in_proj_bias"],
                "self_attn.out_proj.weight":weight_dict[f"{i}.self_attn.out_proj.weight"],
                "self_attn.out_proj.bias":  weight_dict[f"{i}.self_attn.out_proj.bias"],
                "linear1.weight":           weight_dict[f"{i}.linear1.weight"],
                "linear1.bias":             weight_dict[f"{i}.linear1.bias"],
                "linear2.weight":           weight_dict[f"{i}.linear2.weight"],
                "linear2.bias":             weight_dict[f"{i}.linear2.bias"],
                "norm1.weight":             weight_dict[f"{i}.norm1.weight"],
                "norm1.bias":               weight_dict[f"{i}.norm1.bias"],
                "norm2.weight":             weight_dict[f"{i}.norm2.weight"],
                "norm2.bias":               weight_dict[f"{i}.norm2.bias"],
            })