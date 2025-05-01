import torch
from torch import nn, Tensor
from torch.nn.functional import pad
from typing import Optional, Callable, List, Any
from .positional_embeddings.sinusoidalpositionalencoding import SinusoidalPositionalEncoding
# ----------------------------------------------------------------------------- #
#                               Helper utilities                                #
# ----------------------------------------------------------------------------- #
def _maybe_append(store: Optional[List[Any]], item: Any):
    if store is not None:
        store.append(item)

# ----------------------------------------------------------------------------- #
#             Transformer encoder layer with optional attention dump            #
# ----------------------------------------------------------------------------- #
class TransformerEncoderLayerWithCrossAttention(nn.Module):
    """
    Identical to the original layer, but every forward_* method now accepts
    `attn_store` – a Python list.  If the list is not None, the raw (per-head)
    attention weights produced by the underlying `nn.MultiheadAttention`
    are appended to it.  Nothing else changes.
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
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn  = nn.MultiheadAttention(d_model, nhead, dropout,
                                                batch_first=batch_first, bias=bias,
                                                **factory_kwargs)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout,
                                                batch_first=batch_first, bias=bias,
                                                **factory_kwargs)
        self.linear1  = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.linear2  = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        self.dropout  = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.activation = activation
        self.norm_first = norm_first

    # --------  attention blocks (optionally save weights)  -------- #
    def forward_self_attention(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        attn_store: Optional[List[Tensor]] = None,   # <-- NEW
    ) -> Tensor:
        need = attn_store is not None
        if self.norm_first:
            y = self.norm1(src)
            attn_out, w = self.self_attn(
                y, y, y,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=need,
                average_attn_weights=False)
            _maybe_append(attn_store, w)
            return src + self.dropout1(attn_out)
        else:
            attn_out, w = self.self_attn(
                src, src, src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=need,
                average_attn_weights=False)
            _maybe_append(attn_store, w)
            return self.norm1(src + self.dropout1(attn_out))

    def forward_cross_attention(
        self,
        src: Tensor,
        memory: Tensor,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        attn_store: Optional[List[Tensor]] = None,   # <-- NEW
    ) -> Tensor:
        need = attn_store is not None
        if self.norm_first:
            y = self.norm2(src)
            attn_out, w = self.cross_attn(
                y, memory, memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                need_weights=need,
                average_attn_weights=False)
            _maybe_append(attn_store, w)
            return src + self.dropout2(attn_out)
        else:
            attn_out, w = self.cross_attn(
                src, memory, memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                need_weights=need,
                average_attn_weights=False)
            _maybe_append(attn_store, w)
            return self.norm2(src + self.dropout2(attn_out))

    # unchanged
    def forward_feedforward(self, src: Tensor) -> Tensor:
        if self.norm_first:
            y  = self.norm3(src)
            ff = self.linear2(self.dropout(self.activation(self.linear1(y))))
            return src + self.dropout3(ff)
        ff = self.linear2(self.dropout(self.activation(self.linear1(src))))
        return self.norm3(src + self.dropout3(ff))

# ----------------------------------------------------------------------------- #
#                        Cross-stream transformer backbone                      #
# ----------------------------------------------------------------------------- #
class CrossAttentionTransformer(nn.Module):
    """
    Added `return_attn=False` to `forward`.
    When True, the call returns an additional list containing *all* attention
    maps collected in depth-first order.
    """
    def __init__(self, conf):
        super().__init__()
        # --------------  (original constructor unchanged)  -------------------- #
        self.encode_layers  = conf.num_layers
        self.cross_layers   = conf.cross_layers
        self.concat_layers  = conf.concat_layers
        d_model_1 = conf.attention.d_model_1
        d_model_2 = conf.attention.d_model_2
        d_model_C = conf.attention.d_model_C
        dim_feedforward_1 = conf.attention.dim_feedforward_1
        dim_feedforward_2 = conf.attention.dim_feedforward_2
        dim_feedforward_C = conf.attention.dim_feedforward_C
        nhead   = conf.attention.num_attention_heads
        dropout = conf.attention.dropout
        activation_fn = nn.LeakyReLU(inplace=False)

        # positional encodings
        if conf.positional_embedding == "Sinusoidal":
            self.positional_embedding_1 = SinusoidalPositionalEncoding(d_model_1)
            self.positional_embedding_2 = SinusoidalPositionalEncoding(d_model_2)
        else:
            self.positional_embedding_1 = None
            self.positional_embedding_2 = None

        # plain encoders
        self.encoder_1 = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model_1, nhead,
                                       dim_feedforward_1, dropout,
                                       batch_first=True, activation=activation_fn)
            for _ in range(self.encode_layers)
        ])
        self.encoder_2 = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model_2, nhead,
                                       dim_feedforward_2, dropout,
                                       batch_first=True, activation=activation_fn)
            for _ in range(self.encode_layers)
        ])

        # cross-attention encoders (our custom layer)
        self.encoder_1_cross_layers = nn.ModuleList([
            TransformerEncoderLayerWithCrossAttention(d_model_1, nhead,
                                                      dim_feedforward_1, dropout,
                                                      batch_first=True,
                                                      activation=activation_fn)
            for _ in range(self.cross_layers)
        ])
        self.encoder_2_cross_layers = nn.ModuleList([
            TransformerEncoderLayerWithCrossAttention(d_model_2, nhead,
                                                      dim_feedforward_2, dropout,
                                                      batch_first=True,
                                                      activation=activation_fn)
            for _ in range(self.cross_layers)
        ])

        # post-concat encoder
        self.concat_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model_C, nhead,
                                       dim_feedforward_C, dropout,
                                       batch_first=True, activation=activation_fn)
            for _ in range(self.concat_layers)
        ])

        # eval / mask tokens
        self.eval_token_1 = nn.Parameter(torch.randn(1, 1, d_model_1))
        self.eval_token_2 = nn.Parameter(torch.randn(1, 1, d_model_2))
        self.mask_token_1 = nn.Parameter(torch.randn(d_model_1))
        self.mask_token_2 = nn.Parameter(torch.randn(d_model_2))
        self.encoder_1_mask_rate = conf.encoder_1_mask_rate
        self.encoder_2_mask_rate = conf.encoder_2_mask_rate

        self.final_dropout = (nn.Dropout(conf.final_dropout.prob)
                              if conf.final_dropout.enable else None)

    # ------------------------------------------------------------------ #
    #                               forward                              #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        series_len:     Optional[int]         = None,
        token_mask:     Optional[torch.Tensor] = None,
        return_attn:    bool = False,                     # <-- NEW
    ):
        attn_maps: Optional[List[Tensor]] = [] if return_attn else None

        B = x1.size(0)
        if attention_mask is not None:
            attention_mask = attention_mask.to(torch.bool)

        # positional encodings
        if self.positional_embedding_1 is not None:
            x1 = self.positional_embedding_1(x1, series_len=series_len)
            x2 = self.positional_embedding_2(x2, series_len=series_len)

        # random masking
        if token_mask is not None:
            x1_mask = ((torch.rand_like(token_mask, dtype=torch.float32)
                        <= self.encoder_1_mask_rate) & token_mask)
            x2_mask = ((torch.rand_like(token_mask, dtype=torch.float32)
                        <= self.encoder_2_mask_rate) & token_mask)
            x1[x1_mask] = self.mask_token_1
            x2[x2_mask] = self.mask_token_2

        # evaluation tokens
        x1 = torch.cat([x1, self.eval_token_1.expand(B, -1, -1)], dim=1)
        x2 = torch.cat([x2, self.eval_token_2.expand(B, -1, -1)], dim=1)
        if attention_mask is not None:
            attention_mask = pad(attention_mask, (0, 1), value=False)

        # ------------------------------------------------------------------ #
        #                    STAGE 1 – separate encoders                      #
        # ------------------------------------------------------------------ #
        for layer in self.encoder_1:
            x1 = layer(x1, src_key_padding_mask=attention_mask)
        for layer in self.encoder_2:
            x2 = layer(x2, src_key_padding_mask=attention_mask)

        # keep residual of x1 before cross-attention
        x1_residual = x1

        # ------------------------------------------------------------------ #
        #                    STAGE 2 – cross-attention                        #
        # ------------------------------------------------------------------ #
        for i in range(self.cross_layers):
            # self-attention (both streams)
            x1 = self.encoder_1_cross_layers[i].forward_self_attention(
                x1, src_key_padding_mask=attention_mask,
                attn_store=attn_maps)
            x2 = self.encoder_2_cross_layers[i].forward_self_attention(
                x2, src_key_padding_mask=attention_mask,
                attn_store=attn_maps)

            # cross-attention (1↔2)
            x1 = self.encoder_1_cross_layers[i].forward_cross_attention(
                x1, memory=x2, memory_key_padding_mask=attention_mask,
                attn_store=attn_maps)
            x2 = self.encoder_2_cross_layers[i].forward_cross_attention(
                x2, memory=x1, memory_key_padding_mask=attention_mask,
                attn_store=attn_maps)

            # FFN
            x1 = self.encoder_1_cross_layers[i].forward_feedforward(x1)
            x2 = self.encoder_2_cross_layers[i].forward_feedforward(x2)

        # ------------------------------------------------------------------ #
        #                    STAGE 3 – concatenate & refine                   #
        # ------------------------------------------------------------------ #
        x = torch.cat([x1, x2], dim=-1)
        for layer in self.concat_layers:
            x = layer(x, src_key_padding_mask=attention_mask)

        # optional final dropout
        if self.final_dropout is not None:
            x  = self.final_dropout(x)
            x1_residual = self.final_dropout(x1_residual)

        outputs = (x[:, :-1],        # everything but eval token
                   x[:, -1:],        # eval token embedding(s)
                   x1_residual[:, :-1],
                   x1_residual[:, -1:])

        if return_attn:
            return (*outputs, attn_maps)
        return outputs

    # (inject_weights_into_encoder_1 stays unchanged)