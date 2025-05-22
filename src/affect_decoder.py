from torch import nn

class AffectDecoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.out_layer = nn.Linear(config.arch.TATER.Affect_Decoder.linear_size, 2)
        # self.out_activation = nn.Tanh()
    
    def forward(self, x):
        return self.out_layer(x)