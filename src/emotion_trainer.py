import copy
import cv2
import math
import random
import torch.utils.data
from torch import nn
import torch.nn.functional as F
from src.FLAME.FLAME import FLAME
from src.renderer.renderer import Renderer
from src.smirk_generator import SmirkGenerator
import numpy as np
import src.utils.utils as utils
import src.utils.masking as masking_utils
from src.utils.utils import batch_draw_keypoints, make_grid_from_opencv_images
from torchvision.utils import make_grid

from src.models.transformer.temporaltransformer import TemporalTransformer
from .phoneme_classifier import PhonemeClassifier
from .scheduler.risefallscheduler import CustomWarmupCosineScheduler

class EmotionTrainer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


        self.accumulate_steps = config.train.accumulate_steps if hasattr(config.train, 'accumulate_steps') else 1
        self.global_step = 0  # to track global steps

        self.transformer = TemporalTransformer(config.arch.TATER.Expression.Transformer)
        self.out_layer = nn.Linear(56, 7)

        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def configure_optimizers(self, num_steps, use_default_annealing=False):
        # Adjust total steps for gradient accumulation
        effective_total_steps = num_steps // self.config.train.accumulate_steps
        encoder_scale = .25

        # Define max_lr, min_lr, and warmup steps directly from config
        max_lr = self.config.train.max_lr  # Use max_lr as defined in config
        min_lr = self.config.train.min_lr  # Scaled min_lr as required
        warmup_steps = self.config.train.iterations_until_max_lr  # e.g., 10000

        # Initialize the encoder optimizer
        self.encoder_optimizer = torch.optim.Adam(self.transformer.parameters(), lr=max_lr)

        # Set up CustomCosineScheduler for encoder if OneCycleLR is not preferred
        self.encoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.encoder_optimizer, T_max=effective_total_steps,
                                                                                  eta_min=0.01 * encoder_scale * self.config.train.lr)

    def scheduler_step(self):
        self.encoder_scheduler.step()

    def train(self):
        self.transformer.train()
    
    def eval(self):
        self.transformer.eval()

    def optimizers_zero_grad(self):
        self.transformer.zero_grad()

    def optimizers_step(self, step_encoder=True, step_fuse_generator=True):
        self.encoder_optimizer.step()
    
    def pad_and_create_mask(self, combined_tensor, og_series_len, downsample=False, sample_rate=1):
        def downsample_tensor(tensor, sample_rate):
            """Applies uniform downsampling along the first dimension of the tensor."""
            return tensor[::sample_rate]

        tensor_list = []
        lengths = []
        start = 0
        for s in og_series_len:
            if downsample:
                tensor_list.append(downsample_tensor(combined_tensor[start:start+s], sample_rate))
            else:
                tensor_list.append(combined_tensor[start:start+s])
            
            lengths.append(len(tensor_list[-1]))
            start += s

        # Get the new lengths after downsampling (if applied) or original lengths
        max_len = max(lengths)  # Find the maximum length

        # Initialize the padded tensor with zeros
        num_tensors = len(lengths)
        padded_tensor = torch.zeros((num_tensors, max_len, combined_tensor.shape[1]), dtype=combined_tensor.dtype).cuda()
        
        # Initialize the attention mask (T/F) with False
        attention_mask = torch.zeros((num_tensors, max_len), dtype=torch.bool).cuda()
        
        start_idx = 0
        for i, length in enumerate(lengths):
            # Fill the padded tensor with the corresponding slice from the combined_tensor
            padded_tensor[i, :length] = tensor_list[i]
            
            # Set the attention mask to True for the actual tokens
            attention_mask[i, :length] = 1
            
            # Move to the next slice in the combined tensor
            start_idx += length
        
        # Invert the mask for use with nn.TransformerEncoderLayer (True for padding positions)
        key_padding_mask = ~attention_mask  # Now True for padding, False for valid tokens
        
        return padded_tensor, key_padding_mask, lengths

    def step(self, params, labels, series_len, batch_idx, epoch, phase='train'):
        if phase == 'train':
            self.train()
            torch.set_grad_enabled(True)
        else:
            self.eval()
            torch.set_grad_enabled(False)

        exp_params, attention_mask, series_len = self.pad_and_create_mask(
                torch.cat([params[key] for key in ["expression_params", 'eyelid_params', 'jaw_params']]),
                series_len,
                downsample=self.interp_down_residual,
                sample_rate=self.downsample_rate
            )
        _, exp_class = self.exp_transformer(exp_params, attention_mask, series_len)
        exp_logits = self.out_layer(exp_class)

        loss = self.cross_entropy_loss(exp_logits, labels)
        if phase == 'train':
            self.optimizers_zero_grad()  # Zero the gradients
            loss.backward()  # Accumulate gradients
            
            if (batch_idx + 1) % self.accumulate_steps == 0:
                self.optimizers_step(step_encoder=True, step_fuse_generator=True)  # Apply accumulated gradients
                self.scheduler_step()  # Step the scheduler
                self.global_step += 1  # Increment global step
            
            iteration = batch_idx if epoch == 0 else -1
            self.tater.update_residual_scale(iteration)

        losses = {"cross entropy": loss}
        self.logging(batch_idx, losses, phase)

        return losses

    def save_model(self, state_dict, save_path):
        new_state_dict = {}
        for key in list(state_dict.keys()):
            if key.startswith('transformer'):
                new_state_dict[key] = state_dict[key]

        torch.save(new_state_dict, save_path)
