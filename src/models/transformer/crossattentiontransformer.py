import torch
from torch import nn, Tensor
from torch.nn.functional import pad
from typing import Optional, Callable
from .positional_embeddings.sinusoidalpositionalencoding import SinusoidalPositionalEncoding

class TransformerEncoderLayerWithCrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable[[Tensor], Tensor] = nn.ReLU(),
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
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
    ) -> Tensor:
        src2 = self.self_attn(
            query=src,
            key=src,
            value=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]
        return src + self.dropout1(src2) if self.norm_first else self.norm1(src + self.dropout1(src2))

    def forward_cross_attention(
        self,
        src: Tensor,
        memory: Tensor,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        src2 = self.cross_attn(
            query=src,
            key=memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        return src + self.dropout2(src2) if self.norm_first else self.norm2(src + self.dropout2(src2))

    def forward_feedforward(self, src: Tensor) -> Tensor:
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        return src + self.dropout3(src2) if self.norm_first else self.norm3(src + self.dropout3(src2))


class CrossAttentionTransformer(nn.Module):
    def __init__(self, conf):
        super().__init__()

        # Unpack configuration
        self.num_layers = conf.num_layers
        self.cross_attention_indices = conf.cross_attention_indices or []
        d_model_1 = conf.attention.d_model_1
        d_model_2 = conf.attention.d_model_2
        d_model_C = conf.attention.d_model_C
        dim_feedforward_1 = conf.attention.dim_feedforward_1
        dim_feedforward_2 = conf.attention.dim_feedforward_2
        dim_feedforward_C = conf.attention.dim_feedforward_C


        nhead = conf.attention.num_attention_heads
        dropout = conf.attention.dropout
        activation_fn = nn.LeakyReLU(negative_slope=conf.attention.activation_slope, inplace=False)

        # Define positional embeddings
        if conf.positional_embedding == "Sinusoidal":
            self.positional_embedding_1 = SinusoidalPositionalEncoding(d_model_1)
            self.positional_embedding_2 = SinusoidalPositionalEncoding(d_model_2)
        else:
            self.positional_embedding_1 = None
            self.positional_embedding_2 = None

        # Define custom transformer layers for both streams
        self.encoder_1 = nn.ModuleList([
            TransformerEncoderLayerWithCrossAttention(
                d_model=d_model_1,
                nhead=nhead,
                dim_feedforward=dim_feedforward_1,
                dropout=dropout,
                activation=activation_fn
            )
            for _ in range(self.num_layers)
        ])
        self.encoder_2 = nn.ModuleList([
            TransformerEncoderLayerWithCrossAttention(
                d_model=d_model_2,
                nhead=nhead,
                dim_feedforward=dim_feedforward_2,
                dropout=dropout,
                activation=activation_fn
            )
            for _ in range(self.num_layers)
        ])

        self.concat_layer = nn.TransformerEncoderLayer(
            d_model=d_model_C,
            nhead=nhead,
            dim_feedforward=dim_feedforward_C,
            nhead=conf.attention.num_attention_heads,
            dropout=dropout,
            activation=activation_fn
        )

        # Define projections and evaluation tokens
        self.projection_1_to_2 = nn.Linear(d_model_1, d_model_2)
        self.projection_2_to_1 = nn.Linear(d_model_2, d_model_1)
        self.eval_token_1 = nn.Parameter(torch.rand(1, 1, d_model_1))
        self.eval_token_2 = nn.Parameter(torch.rand(1, 1, d_model_2))

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        series_len: Optional[int] = None,
    ):
        B = x1.size(0)

        # Apply positional embeddings
        if self.positional_embedding_1 is not None:
            x1 = self.positional_embedding_1(x1, series_len=series_len)
        if self.positional_embedding_2 is not None:
            x2 = self.positional_embedding_2(x2, series_len=series_len)

        # Add evaluation tokens
        x1 = torch.cat((x1, self.eval_token_1.expand(B, -1, -1)), dim=1)
        x2 = torch.cat((x2, self.eval_token_2.expand(B, -1, -1)), dim=1)

        if attention_mask is not None:
            attention_mask = pad(attention_mask, (0, 1), mode="constant", value=False)
        if attention_mask is not None:
            attention_mask = pad(attention_mask, (0, 1), mode="constant", value=False)

        # Pass through transformer layers with custom cross-attention
        for i in range(self.num_layers):
            layer_1 = self.encoder_1[i]
            layer_2 = self.encoder_2[i]

            x1 = layer_1.forward_self_attention(x1, src_key_padding_mask=attention_mask)
            x2 = layer_2.forward_self_attention(x2, src_key_padding_mask=attention_mask)

            if i in self.cross_attention_indices:
                memory_1 = self.projection_2_to_1(x2)
                memory_2 = self.projection_1_to_2(x1)
                x1 = layer_1.forward_cross_attention(
                    x1, memory=memory_1, memory_key_padding_mask=attention_mask
                )
                x2 = layer_2.forward_cross_attention(
                    x2, memory=memory_2, memory_key_padding_mask=attention_mask
                )

            x1 = layer_1.forward_feedforward(x1)
            x2 = layer_2.forward_feedforward(x2)

        x = torch.cat([x1, x2], dim=-1)
        x = self.concat_layer(x, src_key_padding_mask=attention_mask)

        if self.final_dropout is not None:
            x[:, :-1, :] = self.final_dropout(x[:, :-1, :])

        return x[:, :-1, :], x[:, -1:, :]
