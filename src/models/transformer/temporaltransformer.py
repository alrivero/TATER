import torch
from torch import nn
from random import randint
from torch.nn.functional import pad
from .positional_embeddings.sinusoidalpositionalencoding import SinusoidalPositionalEncoding

# Resnet Blocks
class TemporalTransformer(nn.Module):
    """
    A transformer used for regression/classification of data using a "eval" class token
    """
    def __init__(self, conf):
        super().__init__()

        # Define what positional encoding we'll be using
        if conf.positional_embedding == "Sinusoidal":
            self.positional_embedding = SinusoidalPositionalEncoding(conf.attention.hidden_size)
        else:
            self.positional_embedding = None

        # Define our transformer blocks we'll be using
        self.num_layers = conf.num_layers
        self.hidden_size = conf.attention.hidden_size
        self.num_attention_heads = conf.attention.num_attention_heads

        # Let's try using the transformer encoder layers from PyTorch
        self.attention_blocks = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model=conf.attention.hidden_size,
            dim_feedforward=conf.attention.hidden_size_2,
            nhead=conf.attention.num_attention_heads,
            dropout=conf.attention.attention_probs_dropout_prob,
            layer_norm_eps=conf.attention.layer_norm_eps,
            batch_first=True,
            activation=nn.LeakyReLU(negative_slope=0.01, inplace=False)  # LeakyReLU activation
        ) for _ in range(self.num_layers)])

        # Define a class token used for evaluating a sequence
        self.eval_token = torch.rand(conf.attention.hidden_size)
        self.eval_token = torch.nn.Parameter(self.eval_token)
        
        # Final layer used to compute residual and eval toke
        self.res_out = nn.Linear(conf.attention.hidden_size, conf.attention.hidden_size)
        self.eval_out = nn.Linear(conf.attention.hidden_size, conf.attention.hidden_size)
        
        # Decide if we're using dropout in our final layer
        if conf.final_dropout.enable:
            self.final_dropout = torch.nn.Dropout(p=conf.final_dropout.prob)
        else:
            self.final_dropout = None

    def forward(self, x, attention_mask, series_len=None):
        B, S, D = x.shape

        # Apply our positional encoding
        if self.positional_embedding is not None:
            x = self.positional_embedding(
                x,
                series_len=series_len
            )

        # Insert our eval token at the end of each sequence by modifying x and attention masks
        x = torch.cat((x, self.eval_token.expand(B, 1, -1)), dim=-2)
        attention_mask = pad(attention_mask, (0, 1), mode="constant", value=True)

        for i in range(self.num_layers):
            x = self.attention_blocks[i](x, src_key_padding_mask=attention_mask)

        if self.final_dropout is not None:
            x[:, :-1, :] = self.final_dropout(x[:, :-1, :])

        return x[:, :-1, :], x[:, -1:, :]