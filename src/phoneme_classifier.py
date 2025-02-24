import torch
from torch import nn
import torch.nn.functional as F
from .models.transformer.temporaltransformer import TemporalTransformer

class PhonemeClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.type = config.arch.Phoneme_Classifier.type  # Correct config access
        if self.type == "Linear":
            if config.arch.Phoneme_Classifier.use_latent:
                self.linear_out = torch.nn.Linear(640, 10)
            else:
                self.linear_out = torch.nn.Linear(56, 10)
        elif self.type == "Transformer":
            self.transformer = TemporalTransformer(config.arch.Phoneme_Classifier.Transformer)

        self.classifier = nn.Linear(config.arch.Phoneme_Classifier.Transformer.attention.hidden_size, 10)
    
    def forward(self, x, att_mask=None, series_len=None):
        if self.type == "Linear":
            return self.linear_out(x)
        elif self.type == "Transformer":
            _, cls = self.transformer(x, att_mask, series_len)
            logits = self.classifier(cls.squeeze(1))
            return logits
        
