
import torch
import torch.nn.functional as F

from tater_encoder import TATEREncoder

class CARAEncoder(TATEREncoder):
    def __init__(self, config, n_exp=50, n_shape=300):
        super().__init__(config, n_exp, n_shape)
        
        del self.

    def forward(self, img_batch, og_series_len, audio_batch=None, token_mask=None, video_mask=None, audio_mask=None):
        outputs = {}
        if isinstance(img_batch, list):
            img_cat = torch.cat(img_batch)
        else:
            img_cat = img_batch
        
        if self.use_latent_exp:
            exp_encodings_all = self.expression_encoder.encoder(img_cat)
            exp_encodings_all = F.adaptive_avg_pool2d(exp_encodings_all[-1], (1, 1)).squeeze(-1).squeeze(-1)
        else:
            exp_encodings_all = self.expression_encoder(img_cat)
            exp_encodings_all = torch.cat(list(exp_encodings_all.values()), dim=-1)
            exp_encodings_all = F.pad(exp_encodings_all, (0, 1), "constant", 0) # Slighlty too short
        
        if self.use_exp_linear_downsample:
            exp_encodings_all = self.exp_layer_down(exp_encodings_all)
        
        if video_mask is not None:
            series_start = 0
            for i in range(len(video_mask)):
                if not video_mask[i]:
                    exp_encodings_all[series_start:series_start + og_series_len[i]] = 0.0
                series_start += og_series_len[i]
        
        exp_encodings_down, attention_mask, series_len = self.pad_and_create_mask(
            exp_encodings_all,
            og_series_len,
            downsample=self.interp_down_residual,
            sample_rate=self.downsample_rate
        )

        if self.exp_use_audio:
            if audio_mask is not None:
                for i in range(len(audio_mask)):
                    if not audio_mask[i]:
                        audio_batch[i] = torch.zeros_like(audio_batch[i])

            audio_batch = torch.cat(audio_batch)
            if self.use_exp_linear_downsample:
                audio_batch = self.exp_layer_audio_down(audio_batch)

            audio_encodings_down, _, series_len = self.pad_and_create_mask(
                audio_batch,
                og_series_len,
                downsample=self.interp_down_residual,
                sample_rate=self.downsample_rate
            )

            exp_residual_out, exp_class, _, _ = self.exp_transformer(exp_encodings_down, audio_encodings_down, attention_mask, series_len, token_mask)
        else:
            exp_residual_out, exp_class = self.exp_transformer(exp_encodings_down, attention_mask, series_len, token_mask)
        
        outputs["exp_residual_out"] = exp_residual_out
        outputs["exp_class"] = exp_class
    
        return outputs




