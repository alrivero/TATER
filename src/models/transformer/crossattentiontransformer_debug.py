import torch
from torch import nn, Tensor
from torch.nn.functional import pad
from typing import Optional, Callable, List, Tuple
from .positional_embeddings.sinusoidalpositionalencoding import SinusoidalPositionalEncoding

class TransformerEncoderLayerWithCrossAttention(nn.Module):
    """
    Exactly your original cross-attention layer, but forward_self_attention &
    forward_cross_attention now directly append to attn_store list.
    """
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
        **unused,  # swallow extras
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn  = nn.MultiheadAttention(d_model, nhead, dropout,
                                                batch_first=batch_first, bias=bias,
                                                **factory_kwargs)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout,
                                                batch_first=batch_first, bias=bias,
                                                **factory_kwargs)
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        self.norm1    = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2    = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3    = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
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
        attn_store: Optional[List[Tuple[str, Tensor]]] = None,
        scope: str = ""
    ) -> Tensor:
        need = attn_store is not None
        q, k, v = (self.norm1(src),) * 3 if self.norm_first else (src, src, src)
        out, w = self.self_attn(
            query=q, key=k, value=v,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=need,
            average_attn_weights=False,
        )
        if attn_store is not None:
            attn_store.append((f"{scope}_self", w.detach().cpu()))
        if self.norm_first:
            return src + self.dropout1(out)
        return self.norm1(src + self.dropout1(out))

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
        tag = f"{scope}_{direction}"
        q = self.norm2(src) if self.norm_first else src
        out, w = self.cross_attn(
            query=q, key=memory, value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=need,
            average_attn_weights=False,
        )
        if attn_store is not None:
            attn_store.append((tag, w.detach().cpu()))
        if self.norm_first:
            return src + self.dropout2(out)
        return self.norm2(src + self.dropout2(out))

    def forward_feedforward(self, src: Tensor) -> Tensor:
        if self.norm_first:
            y  = self.norm3(src)
            ff = self.linear2(self.dropout(self.activation(self.linear1(y))))
            return src + self.dropout3(ff)
        ff = self.linear2(self.dropout(self.activation(self.linear1(src))))
        return self.norm3(src + self.dropout3(ff))


class CrossAttentionTransformer(nn.Module):
    """
    Same public API; forward adds return_attn flag to collect every map.
    """
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

        # Stage 1: separate streams
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

        # Stage 2: cross attention
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

        # Stage 3: concat
        self.concat_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dC, nhead=nhead,
                dim_feedforward=dfC, dropout=dropout,
                batch_first=True, activation=act
            ) for _ in range(self.concat_layers)
        ])

        # Tokens & masks
        self.eval_token_1 = nn.Parameter(torch.rand(1, 1, d1))
        self.eval_token_2 = nn.Parameter(torch.rand(1, 1, d2))
        self.mask_token_1 = nn.Parameter(torch.rand(d1))
        self.mask_token_2 = nn.Parameter(torch.rand(d2))
        self.encoder_1_mask_rate = conf.encoder_1_mask_rate
        self.encoder_2_mask_rate = conf.encoder_2_mask_rate
        self.final_dropout = (nn.Dropout(conf.final_dropout.prob)
                              if conf.final_dropout.enable else None)

        # Hooks for stage-1 & stage-3 self-attn
        self._register_attention_hooks()

    def _register_attention_hooks(self):
        for i, layer in enumerate(self.encoder_1):
            layer.self_attn.register_forward_hook(
                self._hook_factory(f"enc1_L{i}_self"))
        for i, layer in enumerate(self.encoder_2):
            layer.self_attn.register_forward_hook(
                self._hook_factory(f"enc2_L{i}_self"))
        for i, layer in enumerate(self.concat_layers):
            layer.self_attn.register_forward_hook(
                self._hook_factory(f"concat_L{i}_self"))

    def _hook_factory(self, tag: str):
        def hook(module, inp, out):
            # out == (attn_out, attn_weights)
            if isinstance(out, tuple) and len(out) == 2 and out[1] is not None:
                # use the shared buffer
                buf = getattr(self, "_attn_buf", None)
                if isinstance(buf, list):
                    buf.append((tag, out[1].detach().cpu()))
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
        # Shared list of (tag, Tensor)
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
            attention_mask = pad(attention_mask, (0, 1), value=False)

        # Stage 1
        for layer in self.encoder_1:
            x1 = layer(x1, src_key_padding_mask=attention_mask)
        for layer in self.encoder_2:
            x2 = layer(x2, src_key_padding_mask=attention_mask)

        x1_residual = x1

        # Stage 2
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

        # Stage 3
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