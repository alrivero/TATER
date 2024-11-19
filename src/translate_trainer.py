import copy
import cv2
import math
import random
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn
from src.FLAME.FLAME import FLAME
from src.renderer.renderer import Renderer
from src.smirk_encoder import SmirkEncoder
from src.smirk_generator import SmirkGenerator
from src.base_trainer import BaseTrainer 
import numpy as np
import src.utils.utils as utils
import src.utils.masking as masking_utils
from src.utils.utils import batch_draw_keypoints, make_grid_from_opencv_images
from torchvision.utils import make_grid

from .smirk_trainer import SmirkTrainer
from .tater_encoder import TATEREncoder
from .phoneme_classifier import PhonemeClassifier
from .scheduler.risefallscheduler import CustomWarmupCosineScheduler

class TranslateTrainer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.smirk_generator = SmirkGenerator(in_channels=6, out_channels=3, init_features=32, res_blocks=5)
            
        self.flame = FLAME(n_exp=self.config.arch.num_expression, n_shape=self.config.arch.num_shape)
        self.renderer = Renderer(render_full_head=False)

        from src.losses.VGGPerceptualLoss import VGGPerceptualLoss
        self.vgg_loss = VGGPerceptualLoss()
        self.vgg_loss.eval()
        for param in self.vgg_loss.parameters():
            param.requires_grad_(False)
            
        # --------- setup flame masks for sampling --------- #
        self.face_probabilities = masking_utils.load_probabilities_per_FLAME_triangle()

    def configure_optimizers(self, num_steps):
        # Adjust total steps for gradient accumulation
        effective_total_steps = num_steps // self.config.train.accumulate_steps

        if hasattr(self, 'fuse_generator_optimizer'):
            for g in self.smirk_generator_optimizer.param_groups:
                g['lr'] = self.config.train.lr
        else:
            self.smirk_generator_optimizer = torch.optim.Adam(self.smirk_generator.parameters(), lr= self.config.train.lr, betas=(0.5, 0.999))

        
        self.smirk_generator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.smirk_generator_optimizer, T_max=effective_total_steps,
                                                                                eta_min=0.01 * self.config.train.lr)

    def logging(self, batch_idx, losses, phase):
        # ---------------- logging ---------------- #
        if self.config.train.log_losses_every > 0 and batch_idx % self.config.train.log_losses_every == 0:
            # print losses in one line
            loss_str = ''
            for k, v in losses.items():
                loss_str += f'{k}: {v:.6f} '
            if self.config.arch.enable_fuse_generator and self.config.train.optimize_generator:
                loss_str += f'Generator LR: {self.smirk_generator_scheduler.get_last_lr()[0]:.6f} '
            print(loss_str)

    def scheduler_step(self):
        self.smirk_generator_scheduler.step()

    def train(self):
        self.smirk_generator.train()
    
    def eval(self):
        self.smirk_generator.eval()

    def optimizers_zero_grad(self):
        self.smirk_generator_optimizer.zero_grad()

    def optimizers_step(self, step_encoder=True, step_fuse_generator=True):
        self.smirk_generator_optimizer.step()

    def step1(self, batch, encoder_output):
        to_concat = ['flag_landmarks_fan', 'img', 'img_mica', 'landmarks_fan', 'landmarks_mp', 'mask']
        for key in to_concat:
            batch[key] = torch.concat(batch[key])

        flame_output = self.flame.forward(encoder_output)
        renderer_output = self.renderer.forward(flame_output['vertices'], encoder_output['cam'],
                                                landmarks_fan=flame_output['landmarks_fan'], landmarks_mp=flame_output['landmarks_mp'])
        rendered_img = renderer_output['rendered_img']
        flame_output.update(renderer_output)
 
        losses = {}
        img = batch['img']
        masks = batch['mask']
        rendered_mask = 1 - (rendered_img == 0).all(dim=1, keepdim=True).float()
        tmask_ratio = self.config.train.mask_ratio
        
        npoints, _ = masking_utils.mesh_based_mask_uniform_faces(flame_output['transformed_vertices'], 
                                                                    flame_faces=self.flame.faces_tensor,
                                                                    face_probabilities=self.face_probabilities,
                                                                    mask_ratio=tmask_ratio)
        extra_points = masking_utils.transfer_pixels(img, npoints, npoints)
        masked_img = masking_utils.masking(img, masks, extra_points, self.config.train.mask_dilation_radius, rendered_mask=rendered_mask)

        reconstructed_img = self.smirk_generator(torch.cat([rendered_img, masked_img], dim=1))

        reconstruction_loss = F.l1_loss(reconstructed_img, img, reduction='none')

        loss_img = reconstruction_loss.mean(dim=1, keepdim=True)
        losses['reconstruction_loss'] = reconstruction_loss.mean()
        losses['perceptual_vgg_loss'] = self.vgg_loss(reconstructed_img, img)

        fuse_generator_losses = losses['perceptual_vgg_loss'] * self.config.train.loss_weights['perceptual_vgg_loss'] + \
                                losses['reconstruction_loss'] * self.config.train.loss_weights['reconstruction_loss']

        loss_first_path = (
            (fuse_generator_losses if self.config.arch.enable_fuse_generator else 0)
        )

        for key, value in losses.items():
            losses[key] = value.item() if isinstance(value, torch.Tensor) else value

        outputs = {}
        outputs['rendered_img'] = rendered_img
        outputs['vertices'] = flame_output['vertices']
        outputs['img'] = img
        outputs['landmarks_fan_gt'] = batch['landmarks_fan']
        outputs['landmarks_fan'] = flame_output['landmarks_fan']
        outputs['landmarks_mp'] = flame_output['landmarks_mp']
        outputs['landmarks_mp_gt'] = batch['landmarks_mp']
        
        if self.config.arch.enable_fuse_generator:
            outputs['loss_img'] = loss_img
            outputs['reconstructed_img'] = reconstructed_img
            outputs['masked_1st_path'] = masked_img

        for key in outputs.keys():
            outputs[key] = outputs[key].detach().cpu()

        outputs['encoder_output'] = encoder_output

        return outputs, losses, loss_first_path, encoder_output

    def step(self, batch, all_params, batch_idx, epoch, phase='train'):
        outputs1, losses1, loss_first_path, encoder_output = self.step1(batch, all_params)
        
        if phase == 'train':
            self.optimizers_zero_grad()  # Zero the gradients
            loss_first_path.backward()  # Accumulate gradients
            
            if (batch_idx + 1) % self.accumulate_steps == 0:
                self.optimizers_step(step_encoder=True, step_fuse_generator=True)  # Apply accumulated gradients
                self.scheduler_step()  # Step the scheduler
                self.global_step += 1  # Increment global step

        losses = losses1
        self.logging(batch_idx, losses, phase)

        return outputs1
    
    def save_model(self, state_dict, save_path):
        # remove everything that is not smirk_encoder or smirk_generator
        new_state_dict = {}
        for key in list(state_dict.keys()):
            if key.startswith('smirk_generator'):
                new_state_dict[key] = state_dict[key]

        torch.save(new_state_dict, save_path)