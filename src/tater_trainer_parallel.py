import copy
import cv2
import math
import random
import wandb
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms.functional as F_v
from src.FLAME.FLAME import FLAME
from src.renderer.renderer import Renderer
from src.smirk_encoder import SmirkEncoder
from src.smirk_generator import SmirkGenerator
from src.tater_generator import TATERGenerator
from src.base_trainer import BaseTrainer 
import numpy as np
import os
import src.utils.utils as utils
import src.utils.masking as masking_utils
from src.utils.utils import batch_draw_keypoints, make_grid_from_opencv_images
from torchvision.utils import make_grid
from transformers import Wav2Vec2Processor

from .smirk_trainer import SmirkTrainer
from .tater_encoder import TATEREncoder
from .phoneme_classifier import PhonemeClassifier
from .scheduler.risefallscheduler import CustomWarmupCosineScheduler

from external.Visual_Speech_Recognition_for_Multiple_Languages.lipreading.model import Lipreading
from external.Visual_Speech_Recognition_for_Multiple_Languages.dataloader.transform import Compose, Normalize, CenterCrop, SpeedRate, Identity
from configparser import ConfigParser

import sys
sys.path.append("/local/PTSD_STOP/TATER/external/stylegan_v")
sys.path.append("/local/PTSD_STOP/TATER/external/stylegan_v/src_v")

from omegaconf import OmegaConf, DictConfig
from external.stylegan_v.src_v.training.networks import Discriminator
from external.stylegan_v.src_v.torch_utils.misc import copy_params_and_buffers
from external.stylegan_v.src_v.legacy import load_network_pkl
from external.stylegan_v.src_v.dnnlib.util import open_url
from external.stylegan_v.src_v.torch_utils.ops import conv2d_gradfix

class TATERTrainerParallel(SmirkTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.accumulate_steps = config.train.accumulate_steps if hasattr(config.train, 'accumulate_steps') else 1
        self.global_step = 0  # to track global steps

        if self.config.arch.enable_fuse_generator:
            if self.config.arch.enable_temporal_generator:
                self.smirk_generator = TATERGenerator(in_channels=6, out_channels=3, init_features=32, res_blocks=5)
            else:
                self.smirk_generator = SmirkGenerator(in_channels=6, out_channels=3, init_features=32, res_blocks=5)
        
        self.tater = TATEREncoder(self.config, n_exp=self.config.arch.num_expression, n_shape=self.config.arch.num_shape)
        self.smirk_encoder = self.tater  # Backwards compatibility
        self.using_phoneme_classifier = self.config.arch.Phoneme_Classifier.enable
        if self.using_phoneme_classifier:
            self.phoneme_classifier = PhonemeClassifier(config)
            self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
            self.use_latent_for_phoneme = self.config.arch.Phoneme_Classifier.use_latent
        self.use_audio = self.config.arch.TATER.Expression.use_audio

        self.mask_mouth = self.config.train.mask_mouth
        self.swap_mouth = self.config.train.swap_mouth
        
        self.flame = FLAME(n_exp=self.config.arch.num_expression, n_shape=self.config.arch.num_shape)
        self.renderer = Renderer(render_full_head=False)
        self.setup_losses()

        self.templates = utils.load_templates()
        self.l2_loss = torch.nn.MSELoss()
            
        # --------- setup flame masks for sampling --------- #
        self.face_probabilities = masking_utils.load_probabilities_per_FLAME_triangle()

        self.device = next(self.parameters()).device

        lip_config = ConfigParser()
        lip_config.read('configs/lipread_config.ini')
        self.lip_reader = Lipreading(
            lip_config,
            device=self.device
        )
        self.lip_reader.eval()
        self.lip_reader.model.eval()

        self.mp_upper_inner_lip_idxs = [66, 73, 74, 75, 76, 86, 91, 92, 93, 94, 104]
        self.mp_lower_inner_lip_idxs = [67, 78, 79, 81, 83, 96, 97, 99, 101]

        # Load processor for phoneme-based Wav2Vec2 model
        processor = Wav2Vec2Processor.from_pretrained("vitouphy/wav2vec2-xls-r-300m-phoneme")
        vocab_size = processor.tokenizer.vocab_size

        # Define phoneme-to-masking probability dictionary (legible version)
        self.phoneme_mask_probs = {
            "m": 0.60, "b": 0.60, "p": 0.60,
            "f": 0.60, "v": 0.60, "w": 0.60,
            "ch": 0.60, "jh": 0.60,
            "s": 0.60, "z": 0.60, "sh": 0.60, "zh": 0.60,
            "t": 0.60, "d": 0.60, "n": 0.60, "l": 0.60, "dx": 0.60, "r": 0.60,
            "k": 0.55, "g": 0.55, "ng": 0.55, "y": 0.55, "j": 0.55,
            "hh": 0.50, "h#": 0.50,
            "aa": 0.475, "ae": 0.475, "ah": 0.475, "aw": 0.475, "ay": 0.475,
            "eh": 0.475, "er": 0.475, "ey": 0.475, "ih": 0.475, "iy": 0.475,
            "ow": 0.475, "oy": 0.475, "uh": 0.475, "uw": 0.475,
        }
        # self.phoneme_mask_probs = {
        #     "m": 0.80, "b": 0.80, "p": 0.80,
        #     "f": 0.75, "v": 0.75, "w": 0.75,
        #     "ch": 0.70, "jh": 0.70,
        #     "s": 0.60, "z": 0.60, "sh": 0.60, "zh": 0.60,
        #     "t": 0.50, "d": 0.50, "n": 0.50, "l": 0.50, "dx": 0.50, "r": 0.50,
        #     "k": 0.40, "g": 0.40, "ng": 0.40, "y": 0.40, "j": 0.40,
        #     "hh": 0.30, "h#": 0.30,
        #     "aa": 0.25, "ae": 0.25, "ah": 0.25, "aw": 0.25, "ay": 0.25,
        #     "eh": 0.25, "er": 0.25, "ey": 0.25, "ih": 0.25, "iy": 0.25,
        #     "ow": 0.25, "oy": 0.25, "uh": 0.25, "uw": 0.25,
        # }

        # Convert phoneme names to numeric token IDs
        self.phoneme_id_to_mask_prob = {}
        for idx in range(vocab_size):
            phoneme = processor.tokenizer.convert_ids_to_tokens([idx])[0]

            # Assign probability if phoneme exists in the dictionary, else default to 0.25
            self.phoneme_id_to_mask_prob[idx] = self.phoneme_mask_probs.get(phoneme, 0.25)
        
        self.token_masking = self.config.train.token_masking
        self.masking_rate = self.config.train.masking_rate
        self.max_masked = self.config.train.max_masked
        self.min_masked = self.config.train.min_masked
        if not (self.token_masking == "Random" or self.token_masking == "Phoneme"):
            self.token_masking = None

        self.modality_dropout = self.config.train.modality_dropout
        self.video_dropout_rate = self.config.train.video_dropout_rate
        self.audio_dropout_rate = self.config.train.audio_dropout_rate

        self.use_phoneme_onehot = self.config.train.phoneme_onehot

        self.use_series_pixel_sampling = self.config.train.series_pixel_sampling

        self.D_cfg = OmegaConf.load("/local/PTSD_STOP/TATER/configs/stylegan_D.yaml")
        OmegaConf.set_struct(self.D_cfg, True)
        self.discriminator = Discriminator(
            c_dim=0,
            img_resolution=256,
            img_channels=3,
            channel_base=16384,
            cfg=self.D_cfg.discriminator
        ).to(self.device)
        self.r1_gamma = 0.2

        self.last_D_value = 0.0

    def logging(self, batch_idx, losses, phase):
        """
        Logs losses and other information during training and evaluation.
        Logs only from rank 0 to prevent redundant outputs.
        Also logs to wandb if initialized.
        """
        # Check if distributed training is being used
        is_distributed = torch.distributed.is_initialized()

        # Get the rank of the current process
        rank = torch.distributed.get_rank() if is_distributed else 0

        # Only log from rank 0
        if rank == 0 and self.config.train.log_losses_every > 0 and batch_idx % self.config.train.log_losses_every == 0:
            # Construct loss string for console logging
            # loss_str = ''
            # for k, v in losses.items():
            #     loss_str += f'{k}: {v:.6f} '
            # loss_str += f'Encoder LR: {self.encoder_scheduler.get_last_lr()[0]:.6f} '
            # if self.config.arch.enable_fuse_generator and self.config.train.optimize_generator:
            #     loss_str += f'Generator LR: {self.smirk_generator_scheduler.get_last_lr()[0]:.6f} '

            # # Print loss string
            # print(f'[{phase.upper()}] Batch {batch_idx}: {loss_str}')

            # Log losses and learning rates to wandb
            if 'wandb' in globals() and wandb.run is not None:
                wandb_log_data = {f"{phase}/{k}": v for k, v in losses.items()}
                wandb_log_data[f"{phase}/encoder_lr"] = self.encoder_scheduler.get_last_lr()[0]
                wandb_log_data[f"{phase}/gan_D_recon_loss"] = self.last_D_value
                if self.config.arch.enable_fuse_generator and self.config.train.optimize_generator:
                    wandb_log_data[f"{phase}/generator_lr"] = self.smirk_generator_scheduler.get_last_lr()[0]
                wandb_log_data[f"{phase}/batch_idx"] = batch_idx
                wandb.log(wandb_log_data)

    def configure_optimizers(self, num_steps, use_default_annealing=False):
        # Adjust total steps for gradient accumulation
        effective_total_steps = num_steps // self.config.train.accumulate_steps // self.config.train.batch_size

        # Define max_lr, min_lr, and warmup steps directly from config
        max_lr = self.config.train.max_lr  # Use max_lr as defined in config
        min_lr = self.config.train.min_lr  # Scaled min_lr as required
        gen_max_lr = self.config.train.max_lr  # Use unscaled LR values for generator
        gen_min_lr = self.config.train.min_lr  # Scaled min_lr for generator as required
        warmup_steps = self.config.train.iterations_until_max_lr  # e.g., 10000

        # Encoder Optimizer setup
        params = []
        if self.config.train.optimize_base_expression:
            params += list(self.tater.expression_encoder.parameters()) 
        if self.config.train.optimize_base_shape:
            params += list(self.tater.shape_encoder.parameters())
        if self.config.train.optimize_base_pose:
            params += list(self.tater.pose_encoder.parameters())
        if self.config.train.optimize_tater:
            if not self.config.arch.TATER.Expression.use_base_encoder:
                params += list(self.tater.exp_transformer.parameters())
                if self.config.arch.TATER.Expression.use_linear:
                    params += list(self.tater.exp_layer.parameters())
                if self.config.arch.TATER.Expression.use_linear_downsample:
                    params += list(self.tater.exp_layer_down.parameters())
                if self.config.arch.TATER.Expression.use_audio:
                    params += list(self.tater.residual_linear.parameters())
            if not self.config.arch.TATER.Shape.use_base_encoder:
                params += list(self.tater.shape_transformer.parameters())
                if self.config.arch.TATER.Shape.use_linear:
                    params += list(self.tater.shape_layer.parameters())
                if self.config.arch.TATER.Shape.use_linear_downsample:
                    params += list(self.tater.shape_layer_down.parameters())
            if not self.config.arch.TATER.Pose.use_base_encoder:
                params += list(self.tater.pose_transformer.parameters())
                if self.config.arch.TATER.Pose.use_linear:
                    params += list(self.tater.pose_layer.parameters())
                if self.config.arch.TATER.Pose.use_linear_downsample:
                    params += list(self.tater.pose_layer_down.parameters())
        if self.using_phoneme_classifier and self.config.train.optimize_phoneme_classifier:
            params += list(self.phoneme_classifier.parameters()) 

        # Initialize the encoder optimizer
        self.encoder_optimizer = torch.optim.Adam(params, lr=max_lr)

        # Set up CustomCosineScheduler for encoder if OneCycleLR is not preferred
        if self.config.train.use_one_cycle_lr and not use_default_annealing:
            self.encoder_scheduler = CustomWarmupCosineScheduler(
                self.encoder_optimizer,
                min_lr=min_lr,
                max_lr=max_lr,
                warmup_steps=warmup_steps,
                total_steps=effective_total_steps
            )
        else:
            self.encoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.encoder_optimizer, T_max=effective_total_steps, eta_min=gen_max_lr
            )

        if hasattr(self, 'fuse_generator_optimizer'):
            for g in self.smirk_generator_optimizer.param_groups:
                g['lr'] = gen_max_lr
        else:
            self.smirk_generator_optimizer = torch.optim.Adam(
                self.smirk_generator.parameters(), 
                lr=gen_max_lr, 
                betas=(0.5, 0.999)
            )
        

        if self.config.train.use_one_cycle_lr and not use_default_annealing:
            self.smirk_generator_scheduler = CustomWarmupCosineScheduler(
                self.smirk_generator_optimizer,
                min_lr=gen_min_lr,
                max_lr=gen_max_lr,
                warmup_steps=warmup_steps,
                total_steps=effective_total_steps
            )
        else:
            self.smirk_generator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.smirk_generator_optimizer, T_max=effective_total_steps, eta_min=gen_max_lr
            )

        if self.config.train.optimize_discriminator:
            self.tater_discriminator_optimizer = torch.optim.Adam(
                self.discriminator.parameters(), 
                lr=gen_max_lr, 
                betas=(0.5, 0.999)
            )
            self.tater_discriminator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.tater_discriminator_optimizer, T_max=effective_total_steps, eta_min=gen_max_lr
            )

    # def configure_optimizers(self, train_loader):
    #     n_steps = train_loader  # Number of steps per epoch
    #     encoder_scale = .25
    #     max_lr = encoder_scale * self.config.train.lr

    #     if hasattr(self, 'encoder_optimizer'):
    #         for g in self.encoder_optimizer.param_groups:
    #             g['lr'] = max_lr
    #     else:
    #         params = []
    #         if self.config.train.optimize_base_expression:
    #             params += list(self.tater.expression_encoder.parameters()) 
    #         if self.config.train.optimize_base_shape:
    #             params += list(self.tater.shape_encoder.parameters())
    #         if self.config.train.optimize_base_pose:
    #             params += list(self.tater.pose_encoder.parameters())
    #         if self.config.train.optimize_tater:
    #             if not self.config.arch.TATER.Expression.use_base_encoder:
    #                 params += list(self.tater.exp_transformer.parameters())
    #                 if self.config.arch.TATER.Expression.use_linear:
    #                     params += list(self.tater.exp_layer.parameters())
    #             if not self.config.arch.TATER.Shape.use_base_encoder:
    #                 params += list(self.tater.shape_transformer.parameters())
    #                 if self.config.arch.TATER.Shape.use_linear:
    #                     params += list(self.tater.shape_layer.parameters())
    #             if not self.config.arch.TATER.Pose.use_base_encoder:
    #                 params += list(self.tater.pose_transformer.parameters())
    #                 if self.config.arch.TATER.Pose.use_linear:
    #                     params += list(self.tater.pose_layer.parameters())
    #         if self.using_phoneme_classifier and self.config.train.optimize_phoneme_classifier:
    #             params += list(self.phoneme_classifier.parameters()) 

    #         self.encoder_optimizer = torch.optim.Adam(params, lr=max_lr)
        
    #     # Choose between OneCycleLR and CosineAnnealingLR
    #     if self.config.train.use_one_cycle_lr:
    #         # OneCycleLR with warmup and cosine annealing
    #         self.encoder_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #             self.encoder_optimizer, max_lr=max_lr, total_steps=n_steps, pct_start=15000/n_steps,
    #             anneal_strategy='cos', final_div_factor=10
    #         )
    #     else:
    #         # CosineAnnealingLR scheduler as default
    #         self.encoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #             self.encoder_optimizer, T_max=n_steps, eta_min=0.01 * max_lr
    #         )

    #     if self.config.arch.enable_fuse_generator and self.config.train.optimize_generator:
    #         if hasattr(self, 'fuse_generator_optimizer'):
    #             for g in self.smirk_generator_optimizer.param_groups:
    #                 g['lr'] = self.config.train.lr
    #         else:
    #             self.smirk_generator_optimizer = torch.optim.Adam(self.smirk_generator.parameters(), lr=self.config.train.lr, betas=(0.5, 0.999))

    #         # Apply the same choice for the generator scheduler
    #         if self.config.train.use_one_cycle_lr:
    #             self.smirk_generator_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #                 self.smirk_generator_optimizer, max_lr=self.config.train.lr, total_steps=n_steps,
    #                 pct_start=15000/n_steps, anneal_strategy='cos', final_div_factor=10
    #             )
    #         else:
    #             self.smirk_generator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #                 self.smirk_generator_optimizer, T_max=n_steps, eta_min=0.01 * self.config.train.lr
    #             )


    def scheduler_step(self):
        self.encoder_scheduler.step()
        if self.config.arch.enable_fuse_generator and self.config.train.optimize_generator:
            self.smirk_generator_scheduler.step()
        self.tater_discriminator_scheduler.step()

    def train(self):
        self.smirk_encoder.train()
        if self.config.arch.enable_fuse_generator:
            self.smirk_generator.train()
    
    def eval(self):
        self.smirk_encoder.eval()
        if self.config.arch.enable_fuse_generator:
            self.smirk_generator.eval()

    def optimizers_zero_grad(self):
        self.encoder_optimizer.zero_grad()
        if self.config.arch.enable_fuse_generator and self.config.train.optimize_generator:
            self.smirk_generator_optimizer.zero_grad()
        self.tater_discriminator_optimizer.zero_grad()

    def optimizers_step(self, step_encoder=True, step_fuse_generator=True):
        if step_encoder:
            self.encoder_optimizer.step()
        if step_fuse_generator and self.config.arch.enable_fuse_generator and self.config.train.optimize_generator:
            self.smirk_generator_optimizer.step()
        self.tater_discriminator_optimizer.step()

    def create_base_encoder(self):
        self.base_exp_encoder = copy.deepcopy(self.tater.expression_encoder)
        self.base_shape_encoder = copy.deepcopy(self.tater.shape_encoder)
        self.base_pose_encoder = copy.deepcopy(self.tater.pose_encoder)
        self.base_exp_encoder.eval()
        self.base_shape_encoder.eval()
        self.base_pose_encoder.eval()

    def base_encode(self, img):
        expression_outputs = self.base_exp_encoder(img)
        shape_outputs = self.base_shape_encoder(img)
        pose_outputs = self.base_pose_encoder(img)

        outputs = {}
        outputs.update(expression_outputs)
        outputs.update(shape_outputs)
        outputs.update(pose_outputs)

        return outputs

    def pad_and_create_attention_mask(self, phoneme_exp):
        """
        Pads sequences in phoneme_exp to the same length and creates an attention mask for a Transformer.
        The attention mask uses False for valid tokens and True for padded tokens.
        
        Args:
        phoneme_exp (list of tensors): List of tensors, where each tensor has shape (seq_len, feature_dim).
        
        Returns:
        padded_phoneme_exp (tensor): Padded tensor of shape (batch_size, max_length, feature_dim).
        attention_mask (tensor): Attention mask of shape (batch_size, max_length), where False indicates valid tokens.
        """
        # Find the maximum sequence length
        max_length = max(exp.size(0) for exp in phoneme_exp)
        
        # Initialize list to store padded sequences and masks
        padded_phoneme_exp = []
        attention_mask = []
        
        # Pad each sequence and create the attention mask
        for exp in phoneme_exp:
            seq_len = exp.size(0)
            padding_size = max_length - seq_len
            
            if padding_size > 0:
                # Pad the sequence with zeros along the first dimension (sequence length)
                padded_exp = torch.nn.functional.pad(exp, (0, 0, 0, padding_size))  # Pad only in the time dimension
            else:
                padded_exp = exp  # No padding needed
            
            # Create the attention mask (False for valid tokens, True for padding)
            mask = torch.cat([torch.zeros(seq_len, dtype=torch.bool), torch.ones(padding_size, dtype=torch.bool)])
            
            # Store the padded sequence and mask
            padded_phoneme_exp.append(padded_exp)
            attention_mask.append(mask)
        
        # Convert lists to tensors
        padded_phoneme_exp = torch.stack(padded_phoneme_exp)  # Shape: (batch_size, max_length, feature_dim)
        attention_mask = torch.stack(attention_mask).to(self.device)  # Shape: (batch_size, max_length)
        
        return padded_phoneme_exp, attention_mask
    
    def compute_mouth_bounding_box(self, landmarks, img_height, img_width, scale_w=None, scale_h=None, offset_x=None, offset_y=None):
        """
        Computes a bounding box around the mouth region with optional scaling and offsets.
        If scaling and offsets are not provided, defaults to no scaling and no offset.

        Args:
            landmarks (Tensor): Tensor of shape (B, num_landmarks, 2) with normalized coordinates [-1, 1].
            img_height (int): Height of the image.
            img_width (int): Width of the image.
            scale_w (Optional[Tensor]): Scaling factors for the width of the box (B,). Defaults to 1.0 (no scaling).
            scale_h (Optional[Tensor]): Scaling factors for the height of the box (B,). Defaults to 1.0 (no scaling).
            offset_x (Optional[Tensor]): Horizontal offsets (B,). Defaults to 0.5 (no offset).
            offset_y (Optional[Tensor]): Vertical offsets (B,). Defaults to 0.5 (no offset).

        Returns:
            Tuple[Tensor, Tensor]: Min and max coordinates for the bounding box (B, 2).
        """
        B = landmarks.shape[0]  # Batch size

        # Default to no scaling and no offset if arguments are not provided
        if scale_w is None:
            scale_w = torch.ones(B, device=landmarks.device)
        if scale_h is None:
            scale_h = torch.ones(B, device=landmarks.device)
        if offset_x is None:
            offset_x = torch.full((B,), 0.5, device=landmarks.device)
        if offset_y is None:
            offset_y = torch.full((B,), 0.5, device=landmarks.device)

        # Get the mouth landmarks
        mouth_landmarks = landmarks[:, self.mp_upper_inner_lip_idxs + self.mp_lower_inner_lip_idxs, :]

        # Map normalized coordinates [-1, 1] to pixel coordinates [0, img_width] for x and [0, img_height] for y
        mouth_landmarks[:, :, 0] = (mouth_landmarks[:, :, 0] + 1) * (img_width / 2)  # x-coordinate
        mouth_landmarks[:, :, 1] = (mouth_landmarks[:, :, 1] + 1) * (img_height / 2)  # y-coordinate

        # Compute the center of the mouth bounding box
        center = torch.mean(mouth_landmarks, dim=1)  # (B, 2)

        # Base width and height of the box
        base_width = 96
        base_height = 96

        # Apply scaling factors per frame
        crop_width = (base_width * scale_w).long()  # (B,)
        crop_height = (base_height * scale_h).long()  # (B,)

        # Compute half-width and half-height
        half_width = crop_width // 2  # (B,)
        half_height = crop_height // 2  # (B,)

        # Compute offset displacements based on the normalized offset range [0, 1]
        max_displacement = 48  # Maximum displacement in either direction
        displacement_x = (offset_x * 2 - 1) * max_displacement  # (B,)
        displacement_y = (offset_y * 2 - 1) * max_displacement  # (B,)

        # Adjust the center with the displacement
        center[:, 0] += displacement_x
        center[:, 1] += displacement_y

        # Compute min and max coordinates for the bounding box
        min_coords = center - torch.stack([half_width, half_height], dim=1)  # (B, 2)
        max_coords = center + torch.stack([half_width, half_height], dim=1)  # (B, 2)

        # Ensure the bounding box is within image bounds
        min_coords[:, 0] = torch.clamp(min_coords[:, 0], min=0, max=img_width)
        min_coords[:, 1] = torch.clamp(min_coords[:, 1], min=0, max=img_height)
        max_coords[:, 0] = torch.clamp(max_coords[:, 0], min=0, max=img_width)
        max_coords[:, 1] = torch.clamp(max_coords[:, 1], min=0, max=img_height)

        return min_coords, max_coords

    def differentiable_histogram_equalization(self, image):
        """
        Differentiable approximation of histogram equalization.
        Uses local normalization instead of OpenCV's non-differentiable function.
        
        Args:
            image (torch.Tensor): Grayscale image tensor (B, 1, H, W)
            
        Returns:
            torch.Tensor: Equalized grayscale image
        """
        # Compute local mean and std
        mean = image.mean(dim=(-1, -2), keepdim=True)
        std = image.std(dim=(-1, -2), keepdim=True)

        # Normalize (CLAHE-like contrast enhancement)
        equalized = (image - mean) / (std + 1e-6)  # Prevent div by zero

        # Scale back to [0,1] range
        equalized = (equalized - equalized.min()) / (equalized.max() - equalized.min() + 1e-6)

        return equalized

    def crop_and_transform_mouth_region(self, image, min_coords, max_coords, grayscale=True, equalize=True):
        B, C, H, W = image.shape
        # Convert coordinates to integer indices
        min_coords = min_coords.long()
        max_coords = max_coords.long()

        # Define the transformations
        crop_size = (88, 88)
        mean, std = 0.421, 0.165

        mouth_transform = Compose([
            Normalize(0.0, 1.0),
            CenterCrop(crop_size),
            Normalize(mean, std),
            Identity()
        ])

        cropped_mouths = []
        for b in range(B):
            x_min, y_min = min_coords[b, 0], min_coords[b, 1]
            x_max, y_max = max_coords[b, 0], max_coords[b, 1]

            # Crop the mouth region from the image
            cropped_mouth = image[b, :, y_min:y_max, x_min:x_max]

            if grayscale:
                # Convert to grayscale (1, H, W)
                cropped_mouth = F_v.rgb_to_grayscale(cropped_mouth).squeeze(0).unsqueeze(0)  # Keep batch dim
            if equalize:
                # Apply differentiable histogram equalization
                cropped_mouth = self.differentiable_histogram_equalization(cropped_mouth)


            # Ensure the cropped mouth has 3 dimensions (C, H, W)
            if cropped_mouth.dim() == 2:
                cropped_mouth = cropped_mouth.unsqueeze(0)

            # Resize the cropped mouth to 96x96 to match the original fixed size
            cropped_mouth = F.interpolate(cropped_mouth.unsqueeze(0), size=(96, 96), mode='bilinear', align_corners=False).squeeze(0)

            # Apply the transformations
            cropped_mouth = mouth_transform(cropped_mouth)

            cropped_mouths.append(cropped_mouth)

        # Stack the cropped mouth regions
        cropped_mouths = torch.stack(cropped_mouths, dim=0)
        return cropped_mouths

    def compute_pose_smoothness_loss(self, rotations, series_lengths):
        """
        Compute the average smoothness loss for a tensor of rotation matrices (M, 18).

        Args:
        - rotations: Tensor of shape (M, 18), where each row represents two concatenated rotation matrices.
        - series_lengths: List of integers indicating the lengths of each series.

        Returns:
        - Average smoothness loss across all series and both rotation matrices in each slice.
        """
        # Reshape the rotations to (M, 2, 3, 3) to separate the two rotation matrices per entry
        rotations = rotations.reshape(-1, 2, 3, 3)  # Shape: (M, 2, 3, 3)

        total_loss = 0.0
        total_count = 0
        start_idx = 0

        for s_len in series_lengths:
            if s_len <= 1:
                start_idx += s_len
                continue

            # Extract the series for the current slice
            series = rotations[start_idx : start_idx + s_len]  # Shape: (s_len, 2, 3, 3)

            # Compute smoothness loss for both concatenated rotation matrices
            for i in range(1, s_len):
                for j in range(2):  # Iterate over the two rotation matrices
                    diff = series[i, j] - series[i - 1, j]  # Difference between consecutive matrices
                    loss = torch.norm(diff, p='fro') ** 2
                    total_loss += loss
                    total_count += 1

            start_idx += s_len

        # Compute average smoothness loss
        average_loss = total_loss / total_count if total_count > 0 else torch.tensor(0.0)
        return average_loss

    def crop_switch_inject_mouth(self, img, landmarks, masks, img_height=224, img_width=224, crop_width=67, crop_height=57):
        """
        Crops the mouth region from one image and injects it into another image, using masks to determine which parts to swap.
        Ensures all crops are consistent in size across frames by fixing the crop resolution.

        Args:
            img (Tensor): Input image tensor of shape (B, C, H, W).
            landmarks (Tensor): Tensor of shape (B, num_landmarks, 2) with normalized coordinates [-1, 1].
            masks (Tensor): Binary mask tensor of shape (B, 1, H, W), where 0 indicates the region to swap.
            img_height (int): Height of the image.
            img_width (int): Width of the image.
            crop_width (int): Fixed crop width to ensure consistent resolution.
            crop_height (int): Fixed crop height to ensure consistent resolution.

        Returns:
            Tensor: Modified image tensor of shape (B, C, H, W) with switched mouth regions.
        """
        B, C, H, W = img.shape

        # Get the center of the mouth bounding box
        mouth_landmarks = landmarks[:, self.mp_upper_inner_lip_idxs + self.mp_lower_inner_lip_idxs, :]
        mouth_landmarks[:, :, 0] = (mouth_landmarks[:, :, 0] + 1) * (img_width / 2)  # x-coordinate
        mouth_landmarks[:, :, 1] = (mouth_landmarks[:, :, 1] + 1) * (img_height / 2)  # y-coordinate
        center = torch.mean(mouth_landmarks, dim=1)  # (B, 2)

        # Ensure the crop size is fixed and consistent for all frames
        half_crop_width = crop_width // 2
        half_crop_height = crop_height // 2

        # Compute per-frame crop bounds
        crop_min = center - torch.tensor([half_crop_width, half_crop_height], device=img.device)
        crop_max = center + torch.tensor([half_crop_width, half_crop_height], device=img.device)

        # Clamp crop bounds to image dimensions
        crop_min[:, 0] = torch.clamp(crop_min[:, 0], min=0, max=img_width - crop_width)
        crop_min[:, 1] = torch.clamp(crop_min[:, 1], min=0, max=img_height - crop_height)
        crop_max[:, 0] = crop_min[:, 0] + crop_width
        crop_max[:, 1] = crop_min[:, 1] + crop_height

        # Clone the input image to avoid modifying the original
        modified_img = img.clone()

        # Shuffle the batch indices to switch mouth crops between images
        shuffled_indices = torch.randperm(B, device=img.device)

        for b in range(B):
            # Extract crop coordinates for the current image
            x_min, y_min = crop_min[b, 0].long(), crop_min[b, 1].long()
            x_max, y_max = crop_max[b, 0].long(), crop_max[b, 1].long()

            # Extract crop coordinates for the target (switched) image
            target_idx = shuffled_indices[b]
            target_x_min, target_y_min = crop_min[target_idx, 0].long(), crop_min[target_idx, 1].long()
            target_x_max, target_y_max = crop_max[target_idx, 0].long(), crop_max[target_idx, 1].long()

            # Crop the mouth region from the target image
            mouth_crop = img[target_idx, :, target_y_min:target_y_max, target_x_min:target_x_max]  # (C, crop_height, crop_width)

            # Use the mask of the current image being swapped, ensuring proper channel handling
            mask_cropped = masks[b, :, y_min:y_max, x_min:x_max]  # Maintain channels from mask
            inverse_mask = 1 - mask_cropped

            # Inject the swapped mouth region while respecting the mask
            modified_img[b, :, y_min:y_max, x_min:x_max] = (
                mouth_crop * inverse_mask + modified_img[b, :, y_min:y_max, x_min:x_max] * mask_cropped
            )

        # # Debug: Visualize and save the first frame of the modified image with red tint on the cropped region
        # first_frame = modified_img[0].cpu().permute(1, 2, 0).numpy()  # Convert to HWC format
        # first_frame = (first_frame - first_frame.min()) / (first_frame.max() - first_frame.min())  # Normalize to [0, 1]
        # first_frame = (first_frame * 255).astype(np.uint8)  # Convert to uint8

        # # Add red tint to the crop region
        # import imageio
        # debug_frame = first_frame.copy()
        # x_min, y_min = crop_min[0, 0].long().item(), crop_min[0, 1].long().item()
        # x_max, y_max = crop_max[0, 0].long().item(), crop_max[0, 1].long().item()
        # debug_frame[y_min:y_max, x_min:x_max, 0] = 255  # Increase red channel intensity

        # imageio.imwrite("mouth_debug.png", debug_frame)

        return modified_img

    def apply_mask_mouth(self, masked_img, landmarks_mp, img_size=224):
        # Compute bounding box for the mouth, randomly sample per frame
        B, C, H, W = masked_img.shape

        # Generate random parameters for each frame
        w = torch.rand(B, device=masked_img.device) * (0.65 - 0.3) + 0.3  # Random width scale
        h = torch.rand(B, device=masked_img.device) * (0.65 - 0.2) + 0.2  # Random height scale
        x_off = torch.rand(B, device=masked_img.device) * (0.625 - 0.375) + 0.375  # Random x offset
        y_off = torch.rand(B, device=masked_img.device) * (0.625 - 0.375) + 0.375  # Random y offset

        # Compute bounding boxes for all frames
        gt_mouth_min, gt_mouth_max = self.compute_mouth_bounding_box(
            landmarks_mp, img_size, img_size, w, h, x_off, y_off
        )

        # Extract coordinates for the bounding box
        y_min, x_min = gt_mouth_min[:, 1], gt_mouth_min[:, 0]  # (B,)
        y_max, x_max = gt_mouth_max[:, 1], gt_mouth_max[:, 0]  # (B,)

        # Generate a grid for masking
        y = torch.arange(H, device=masked_img.device).view(1, 1, H, 1)
        x = torch.arange(W, device=masked_img.device).view(1, 1, 1, W)

        # Broadcast dimensions to match (B, 1, H, W)
        mask = (y >= y_min.view(B, 1, 1, 1)) & (y < y_max.view(B, 1, 1, 1)) & \
            (x >= x_min.view(B, 1, 1, 1)) & (x < x_max.view(B, 1, 1, 1))

        # Expand mask to match image channels (B, C, H, W)
        mask = mask.expand(-1, C, -1, -1)

        # Reduce to single-channel mask (B, H, W) for saving
        mask_bw = mask[:, 0, :, :].float()  # Take the first channel and convert to float for visualization

        # Zero out pixels in the masked region
        masked_img = masked_img.clone()  # Avoid modifying input tensor in-place
        masked_img[mask] = 0.0

        return masked_img

    
    def sample_discriminator_frames(self, img, series_len, k=3, max_dist=32):
        device = img.device
        imgs_sampled, idxs_sampled = [], []
        start_idx = 0

        for length in series_len:
            if length < k:
                raise ValueError(f"Subsequence length {length} < k={k}.")

            max_gap = min(max_dist, length - (k - 1))
            if max_gap < (k - 1):
                raise ValueError(f"No valid gap: length={length}, k={k}, max_dist={max_dist}.")

            # Pick first frame t1
            t1 = torch.randint(0, length - k + 1, (1,), device=device).item()

            # Gap range for tk (ensures tk <= length-1)
            gap_min, gap_max = (k - 1), min(max_gap, (length - 1) - t1)
            if gap_min > gap_max:
                raise ValueError(f"No valid gap range with t1={t1} in [0, {length - k}]")

            g = torch.randint(gap_min, gap_max + 1, (1,), device=device).item()
            tk = t1 + g

            # Middle frames
            middle_count = g - 1
            if middle_count < (k - 2):
                raise ValueError(f"Can't pick {k-2} middle frames from gap {middle_count}.")

            mid = torch.randperm(middle_count, device=device)[: k - 2] + (t1 + 1)
            idx = torch.cat([torch.tensor([t1, tk], device=device), mid]).sort().values + start_idx

            imgs_sampled.append(img.index_select(0, idx))
            idxs_sampled.append(idx)
            start_idx += length

        imgs_sampled = torch.cat(imgs_sampled, dim=0)
        imgs_sampled = F.interpolate(imgs_sampled, size=(256, 256), mode='bilinear', align_corners=False)
        idxs_sampled = torch.stack(idxs_sampled, dim=0).to(device)
        return imgs_sampled, idxs_sampled

    def step1(self, batch, encoder_output, batch_idx, series_len):
        B, C, H, W = batch['img'].shape
        flame_output = self.flame.forward(encoder_output)
        
        renderer_output = self.renderer.forward(flame_output['vertices'], encoder_output['cam'],
                                                landmarks_fan=flame_output['landmarks_fan'], landmarks_mp=flame_output['landmarks_mp'])
        rendered_img = renderer_output['rendered_img']
        flame_output.update(renderer_output)
 
        losses = {}
        img = batch['img']
        valid_landmarks = batch['flag_landmarks_fan']
        losses['landmark_loss_fan'] = 0 if torch.sum(valid_landmarks) == 0 else F.mse_loss(flame_output['landmarks_fan'][valid_landmarks,:17], batch['landmarks_fan'][valid_landmarks,:17])
        losses['landmark_loss_mp'] = F.mse_loss(flame_output['landmarks_mp'], batch['landmarks_mp'])

        if self.config.train.use_base_model_for_regularization:
            with torch.no_grad():
                base_output = self.base_encode(batch['img'])
        else:
            base_output = {key[0]: torch.zeros(B, key[1]).to(self.device) for key in zip(['expression_params', 'shape_params', 'jaw_params'], [self.config.arch.num_expression, self.config.arch.num_shape, 3])}

        losses['expression_regularization'] = torch.mean((encoder_output['expression_params'] - base_output['expression_params'])**2)
        losses['shape_regularization'] = torch.mean((encoder_output['shape_params'] - base_output['shape_params'])**2)
        losses['pose_regularization'] = torch.mean((encoder_output['pose_params'] - base_output['pose_params'])**2)
        losses['cam_regularization'] = torch.mean((encoder_output['cam'] - base_output['cam'])**2)
        losses['jaw_regularization'] = torch.mean((encoder_output['jaw_params'] - base_output['jaw_params'])**2)
        losses['eyelid_regularization'] = torch.mean((encoder_output['eyelid_params'] - base_output['eyelid_params'])**2)

        if not self.config.arch.TATER.Expression.use_base_encoder:
            exp_res = [x[:s] for x, s in zip(encoder_output['expression_residuals_down'], encoder_output['res_series_len'])]
            exp_res = torch.cat(exp_res)
            losses['exp_res_regularization'] = torch.mean(exp_res**2)
        if not self.config.arch.TATER.Shape.use_base_encoder:
            shape_res = [x[:s] for x, s in zip(encoder_output['shape_residuals_down'], encoder_output['res_series_len'])]
            shape_res = torch.cat(shape_res)
            losses['shape_res_regularization'] = torch.mean(shape_res**2)
        if not self.config.arch.TATER.Pose.use_base_encoder:
            pose_res = [x[:s] for x, s in zip(encoder_output['pose_residuals_down'], encoder_output['res_series_len'])]
            pose_res = torch.cat(pose_res)
            losses['pose_res_regularization'] = torch.mean(pose_res**2)

        if self.config.arch.enable_fuse_generator:
            masks = batch['mask']
            rendered_mask = 1 - (rendered_img == 0).all(dim=1, keepdim=True).float()
            tmask_ratio = self.config.train.mask_ratio
            
            if self.use_series_pixel_sampling:
                npoints, _ = masking_utils.mesh_based_mask_uniform_faces_series(flame_output['transformed_vertices'],
                                                                        flame_faces=self.flame.faces_tensor,
                                                                        face_probabilities=self.face_probabilities,
                                                                        series_len=series_len,
                                                                        mask_ratio=tmask_ratio)
                # print(npoints.shape)
            else:
                npoints, _ = masking_utils.mesh_based_mask_uniform_faces(flame_output['transformed_vertices'], 
                                                                        flame_faces=self.flame.faces_tensor,
                                                                        face_probabilities=self.face_probabilities,
                                                                        mask_ratio=tmask_ratio)

            # Generate a mask to select frames for augmentation (50% chance)
            augment_mask = torch.rand(len(img)) > 0.5

            # Create specific masks for swapping and masking among the augmented frames
            swap_mask = augment_mask & (torch.rand(len(img)) > 0.5)  # 50% of augmented frames for swapping
            mask_mask = augment_mask & (torch.rand(len(img)) > 0.5)  # 50% of augmented frames for masking

            # Apply swap_mouth augmentation
            if self.swap_mouth:
                img_sampled = img.clone()

                series_start = 0
                for s_len in series_len:
                    # Create series-specific mask for swapping
                    series_swap_mask = swap_mask[series_start:series_start + s_len]

                    # Only apply swapping to frames where swap_mask is True
                    if series_swap_mask.any():  # Proceed if there are True values in the mask
                        img_sampled[series_start:series_start + s_len][series_swap_mask] = self.crop_switch_inject_mouth(
                            img_sampled[series_start:series_start + s_len][series_swap_mask],
                            batch['landmarks_mp'][series_start:series_start + s_len][series_swap_mask],
                            masks
                        )

                    series_start += s_len
            else:
                img_sampled = img

            # Apply masking_utils operations
            extra_points = masking_utils.transfer_pixels(img_sampled, npoints, npoints)
            masked_img = masking_utils.masking(
                img_sampled, masks, extra_points, self.config.train.mask_dilation_radius, rendered_mask=rendered_mask, extra_noise=False
            )

            # Apply mask_mouth augmentation
            if self.mask_mouth:
                series_start = 0
                for s_len in series_len:
                    # Create series-specific mask for masking
                    series_mask_mask = mask_mask[series_start:series_start + s_len]

                    # Only apply masking to frames where mask_mask is True
                    if series_mask_mask.any():  # Proceed if there are True values in the mask
                        masked_img[series_start:series_start + s_len][series_mask_mask] = self.apply_mask_mouth(
                            masked_img[series_start:series_start + s_len][series_mask_mask],
                            batch['landmarks_mp'][series_start:series_start + s_len][series_mask_mask]
                        )

                    series_start += s_len

            if self.config.arch.enable_temporal_generator:
                reconstructed_img = self.smirk_generator(torch.cat([rendered_img, masked_img], dim=1), series_len)
            else:
                reconstructed_img = self.smirk_generator(torch.cat([rendered_img, masked_img], dim=1))

            reconstruction_loss = F.l1_loss(reconstructed_img, img, reduction='none')

            loss_img = reconstruction_loss.mean(dim=1, keepdim=True)
            losses['reconstruction_loss'] = reconstruction_loss.mean()
            losses['perceptual_vgg_loss'] = self.vgg_loss(reconstructed_img, img)

            recon_velocity_loss = 0.0
            if self.config.train.loss_weights["recon_velocity_loss"] > 0:
                total_sequences = len(series_len)  # Number of sequences (videos) in the batch
                start_idx = 0
                recon_velocity_loss = 0.0
                for i, seq_len in enumerate(series_len):
                    if seq_len <= 1:
                        continue  # No velocity loss for sequences with 1 or fewer frames

                    # Extract vertices for the current sequence
                    seq_recons = reconstructed_img[start_idx:start_idx + seq_len, :]  # (seq_len, N)

                    # Compute velocity between consecutive frames for this sequence
                    pred_velocity = seq_recons[1:] - seq_recons[:-1]  # (seq_len-1, N)

                    # Compute L2 loss on the velocity difference (encourages smooth transitions)
                    seq_velocity_loss = F.mse_loss(pred_velocity[1:], pred_velocity[:-1])
                    recon_velocity_loss += seq_velocity_loss / seq_len  # Normalize by sequence length

                    start_idx += seq_len  # Move to the next sequence
            
                # Average velocity loss across all sequences
                recon_velocity_loss /= total_sequences
                losses['recon_velocity_loss'] = recon_velocity_loss
                recon_velocity_loss *= self.config.train.loss_weights['recon_velocity_loss']


            if self.config.train.loss_weights['emotion_loss'] > 0:
                if self.config.train.optimize_generator:
                    for param in self.smirk_generator.parameters():
                        param.requires_grad_(False)
                self.smirk_generator.eval()

                if self.config.arch.enable_temporal_generator:
                    reconstructed_img_p = self.smirk_generator(torch.cat([rendered_img, masked_img], dim=1), series_len)
                else:
                    reconstructed_img_p = self.smirk_generator(torch.cat([rendered_img, masked_img], dim=1))

                if self.config.train.optimize_generator:
                    for param in self.smirk_generator.parameters():
                        param.requires_grad_(True)
                    self.smirk_generator.train()

                losses['emotion_loss'] = self.emotion_loss(reconstructed_img_p, img, metric='l2', use_mean=False).mean()
            else:
                losses['emotion_loss'] = 0
        else:
            losses['reconstruction_loss'] = 0
            losses['perceptual_vgg_loss'] = 0
            losses['emotion_loss'] = 0

        if self.config.train.loss_weights['mica_loss'] > 0:
            losses['mica_loss'] = self.mica.calculate_mica_shape_loss(encoder_output['shape_params'], batch['img_mica'])
        else:
            losses['mica_loss'] = 0

        shape_losses = losses['shape_regularization'] * self.config.train.loss_weights['shape_regularization'] + \
                                    (losses['shape_res_regularization'] * self.config.train.loss_weights['shape_res_regularization'] if not self.config.arch.TATER.Shape.use_base_encoder else 0) + \
                                    losses['mica_loss'] * self.config.train.loss_weights['mica_loss']

        expression_losses = losses['expression_regularization'] * self.config.train.loss_weights['expression_regularization'] + \
                            losses['jaw_regularization'] * self.config.train.loss_weights['jaw_regularization'] + \
                            (losses['exp_res_regularization'] * self.config.train.loss_weights['exp_res_regularization'] if not self.config.arch.TATER.Expression.use_base_encoder else 0)
        
        pose_losses = losses['pose_regularization'] * self.config.train.loss_weights['pose_regularization'] + \
                            losses['cam_regularization'] * self.config.train.loss_weights['cam_regularization'] + \
                            (losses['pose_res_regularization'] * self.config.train.loss_weights['pose_res_regularization'] if not self.config.arch.TATER.Pose.use_base_encoder else 0)
        
        landmark_losses = losses['landmark_loss_fan'] * self.config.train.loss_weights['landmark_loss'] + \
                            losses['landmark_loss_mp'] * self.config.train.loss_weights['landmark_loss'] 

        fuse_generator_losses = losses['perceptual_vgg_loss'] * self.config.train.loss_weights['perceptual_vgg_loss'] + \
                                losses['reconstruction_loss'] * self.config.train.loss_weights['reconstruction_loss'] + \
                                losses['emotion_loss'] * self.config.train.loss_weights['emotion_loss']
        
        # Mouth-specific losses
        self.mp_upper_inner_lip_idxs = [66, 73, 74, 75, 76, 86, 91, 92, 93, 94, 104]
        self.mp_lower_inner_lip_idxs = [67, 78, 79, 81, 83, 96, 97, 99, 101]

        mp_distance = flame_output['landmarks_mp'][:, self.mp_upper_inner_lip_idxs, :].mean(dim=1) - flame_output['landmarks_mp'][:, self.mp_lower_inner_lip_idxs, :].mean(dim=1)
        mp_distance = torch.norm(mp_distance, p=2, dim=-1)
        mp_distance_gt = batch['landmarks_mp'][:, self.mp_upper_inner_lip_idxs, :].mean(dim=1) - batch['landmarks_mp'][:, self.mp_lower_inner_lip_idxs, :].mean(dim=1)
        mp_distance_gt = torch.norm(mp_distance_gt, p=2, dim=-1)
        fan_distance = flame_output['landmarks_fan'][:, 61:64, :].mean(dim=1) - flame_output['landmarks_fan'][:, 65:68, :].mean(dim=1)
        fan_distance = torch.norm(fan_distance, p=2, dim=-1)
        fan_distance_gt = batch['landmarks_fan'][:, 61:64, :].mean(dim=1) - batch['landmarks_fan'][:, 65:68, :].mean(dim=1)
        fan_distance_gt = torch.norm(fan_distance_gt, p=2, dim=-1)


        mp_mouth_loss = self.l2_loss(mp_distance_gt, mp_distance)
        fan_mouth_loss = self.l2_loss(fan_distance_gt, fan_distance)
        landmark_mouth_dist_loss = (mp_mouth_loss + fan_mouth_loss)
        losses['landmark_mouth_dist_loss'] = landmark_mouth_dist_loss
        landmark_mouth_dist_loss *= self.config.train.loss_weights['landmark_mouth_dist_loss']

        gt_mouth_min, gt_mouth_max = self.compute_mouth_bounding_box(batch['landmarks_mp'], 224, 224)
        render_mouths = self.crop_and_transform_mouth_region(reconstructed_img, gt_mouth_min, gt_mouth_max)
        gt_mouths = self.crop_and_transform_mouth_region(batch['img'], gt_mouth_min, gt_mouth_max)

        # Lip reading loss
        if self.config.train.loss_weights['lipreading_loss'] > 0.0:
            self.lip_reader.eval()
            self.lip_reader.model.eval()

            lip_features_gt = self.lip_reader.model.encoder(
                gt_mouths,
                None,
                extract_resnet_feats=True
            )
            lip_features_pred = self.lip_reader.model.encoder(
                render_mouths,
                None,
                extract_resnet_feats=True
            )

            lip_features_gt = lip_features_gt.view(-1, lip_features_gt.shape[-1])
            lip_features_pred = lip_features_pred.view(-1, lip_features_pred.shape[-1])
            lipreading_loss = (lip_features_gt*lip_features_pred).sum(1)/torch.linalg.norm(lip_features_pred,dim=1)/torch.linalg.norm(lip_features_gt,dim=1)
            lipreading_loss = torch.mean(1-torch.mean(lipreading_loss))

            # print(lipreading_loss.requires_grad)

            losses['lipreading_loss'] = lipreading_loss
            lipreading_loss *= self.config.train.loss_weights['lipreading_loss']
        else:
            lipreading_loss = 0.0
        
        if self.config.train.loss_weights['sym_lipreading_loss'] > 0.0:
            self.lip_reader.eval()
            self.lip_reader.model.eval()

            mesh_mouths = self.crop_and_transform_mouth_region(rendered_img, gt_mouth_min, gt_mouth_max, equalize=False)
            teeth_mask = mesh_mouths == 0.0
            mesh_mouths[teeth_mask] = gt_mouths[teeth_mask]

            self.lip_reader.eval()
            self.lip_reader.model.eval()

            lip_features_mesh = self.lip_reader.model.encoder(
                mesh_mouths,
                None,
                extract_resnet_feats=True
            )
            lip_features_pred = self.lip_reader.model.encoder(
                render_mouths,
                None,
                extract_resnet_feats=True
            )

            lip_features_mesh = lip_features_mesh.view(-1, lip_features_mesh.shape[-1])
            lip_features_pred = lip_features_pred.view(-1, lip_features_pred.shape[-1])
            sym_lipreading_loss = (lip_features_mesh*lip_features_pred).sum(1)/torch.linalg.norm(lip_features_pred,dim=1)/torch.linalg.norm(lip_features_mesh,dim=1)
            sym_lipreading_loss = torch.mean(1-torch.mean(sym_lipreading_loss))

            # print(lipreading_loss.requires_grad)

            losses['sym_lipreading_loss'] = sym_lipreading_loss
            sym_lipreading_loss *= self.config.train.loss_weights['sym_lipreading_loss']
        else:
            sym_lipreading_loss = 0.0

        # Velocity loss calculation
        total_sequences = len(series_len)  # Number of sequences (videos) in the batch
        start_idx = 0
        velocity_loss = 0.0
        for i, seq_len in enumerate(series_len):
            if seq_len <= 1:
                continue  # No velocity loss for sequences with 1 or fewer frames

            # Extract vertices for the current sequence
            seq_vertices = flame_output['vertices'][start_idx:start_idx + seq_len, :]  # (seq_len, N)

            # Compute velocity between consecutive frames for this sequence
            pred_velocity = seq_vertices[1:] - seq_vertices[:-1]  # (seq_len-1, N)

            # Compute L2 loss on the velocity difference (encourages smooth transitions)
            seq_velocity_loss = F.mse_loss(pred_velocity[1:], pred_velocity[:-1])
            velocity_loss += seq_velocity_loss / seq_len  # Normalize by sequence length

            start_idx += seq_len  # Move to the next sequence

        # Average velocity loss across all sequences
        velocity_loss /= total_sequences
        losses['velocity_loss'] = velocity_loss
        velocity_loss *= self.config.train.loss_weights['velocity_loss']

        # Rotation smoothness loss for pose
        pose_smoothness_loss = 0.0
        if not self.config.arch.TATER.Pose.use_base_encoder:
            pose_smoothness_loss = self.compute_pose_smoothness_loss(encoder_output["pose_params_mat"], series_len)
            losses["pose_smoothness_loss"] = pose_smoothness_loss
            pose_smoothness_loss *= self.config.train.loss_weights['pose_smoothness_loss']

        if self.using_phoneme_classifier:
            phoneme_loss = torch.tensor(0.0).to(self.device)
            non_silent = 0
            video_start = 0

            if self.phoneme_classifier.type == "Linear":
                for i, video_phonemes in enumerate(batch["phoneme_timestamps"]):
                    if batch["silent"][i] or len(video_phonemes) == 0:
                        continue

                    if self.use_latent_for_phoneme:
                        all_exp = encoder_output["expression_residuals_down"][i]
                    else:
                        all_exp = torch.cat([encoder_output["expression_params"], encoder_output["jaw_params"], encoder_output["eyelid_params"]], dim=-1)

                    phoneme_idxs = [int(round((s + e) / 2)) for (_ , _, s, e) in video_phonemes]
                    phoneme_exp = all_exp[phoneme_idxs]
                    phoneme_gt = torch.tensor([g for (g, _, _, _) in video_phonemes]).to(self.device)

                    phoneme_logits = self.phoneme_classifier(phoneme_exp)
                    phoneme_loss += self.cross_entropy_loss(phoneme_logits, phoneme_gt)
                    video_start += series_len[i]
                    non_silent += 1
                
                phoneme_loss = phoneme_loss / non_silent if non_silent != 0 else phoneme_loss
                losses["phoneme_loss"] = phoneme_loss
                phoneme_loss = self.config.train.loss_weights['phoneme_loss'] * phoneme_loss
            if self.phoneme_classifier.type == "Transformer":
                for i, video_phonemes in enumerate(batch["phoneme_timestamps"]):
                    if batch["silent"][i] or len(video_phonemes) == 0:
                        continue

                    if self.use_latent_for_phoneme:
                        all_exp = encoder_output["expression_residuals_down"][i]
                    else:
                        all_exp = torch.cat([encoder_output["expression_params"], encoder_output["jaw_params"], encoder_output["eyelid_params"]], dim=-1)

                    all_exp = torch.cat(all_exp)
                    phoneme_gt = torch.tensor([g for (g, _, _, _) in video_phonemes]).to(self.device)

                    phoneme_logits = self.phoneme_classifier(phoneme_exp)
                    phoneme_loss += self.cross_entropy_loss(phoneme_logits, phoneme_gt)
                    video_start += series_len[i]
                    non_silent += 1
                
                phoneme_loss = phoneme_loss / non_silent if non_silent != 0 else phoneme_loss
                losses["phoneme_loss"] = phoneme_loss
                phoneme_loss = self.config.train.loss_weights['phoneme_loss'] * phoneme_loss

        # Discriminator/Generator GAN Losses
        gan_loss = 0
        if self.config.train.loss_weights['gan_loss'] > 0.0:
            sampled_recon, sampled_idxs_recon = self.sample_discriminator_frames(reconstructed_img, series_len)
            sampled_gt, sampled_idxs_gt = self.sample_discriminator_frames(img, series_len)
            dummy_c = torch.zeros(len(series_len), 0).to(self.device)

            G_pass = self.config.train.freeze_discriminator_in_first_path
            D_pass = self.config.train.freeze_encoder_in_first_path and self.config.train.freeze_generator_in_first_path
            D_R1 = batch_idx % 150 == 0

            loss_sign = -1 if G_pass else 1
            D_out_recon = self.discriminator(sampled_recon, dummy_c, sampled_idxs_recon)
            gan_loss = F.softplus(loss_sign * D_out_recon['image_logits']).mean()
            if 'video_logits' in D_out_recon:
                gan_loss += F.softplus(loss_sign * D_out_recon['video_logits']).mean()
            
            if G_pass:
                losses["gan_G_recon_loss"] = gan_loss
            else:
                losses["gan_D_recon_loss"] = gan_loss
                self.last_D_value = gan_loss.item()

            if D_pass:
                real_img_tmp = sampled_gt.detach().requires_grad_(D_R1)
                D_out_gt = self.discriminator(real_img_tmp, dummy_c, sampled_idxs_gt)

                if D_R1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(
                            outputs=[D_out_gt['image_logits'].sum()],
                            inputs=[real_img_tmp],
                            create_graph=True,
                            only_inputs=True
                        )[0]

                    r1_penalty = r1_grads.square().sum(dim=[1, 2, 3])  
                    r1_gan_loss = r1_penalty * (self.r1_gamma / 2)  # shape [B * F]

                    B = D_out_gt['image_logits'].shape[0]  # Discriminator output batch
                    N = real_img_tmp.shape[0]             # total images = B * F
                    frames = N // B                       # how many frames per sample?

                    r1_gan_loss = r1_gan_loss.view(B, frames).mean(dim=1)  # shape [B]
                    r1_gan_loss = r1_gan_loss.mean()                       # shape []

                    gan_loss += r1_gan_loss
                    losses["gan_R1_loss"] = r1_gan_loss
                else:
                    gt_gan_loss = F.softplus(-D_out_gt['image_logits']).mean()
                    if 'video_logits' in D_out_gt:
                        gt_gan_loss += F.softplus(-D_out_gt['video_logits']).mean()
                    gan_loss += gt_gan_loss
                    
                    losses["gan_gt_loss"] = gt_gan_loss

            # print("GAN", D_pass, G_pass, D_R1, gan_loss)
            gan_loss *= self.config.train.loss_weights['gan_loss'] 

        loss_first_path = (
            (shape_losses if self.config.train.optimize_base_shape or not self.config.arch.TATER.Shape.use_base_encoder else 0) +
            (expression_losses if self.config.train.optimize_base_expression or not self.config.arch.TATER.Expression.use_base_encoder else 0) +
            (pose_losses if self.config.train.optimize_base_pose or not self.config.arch.TATER.Pose.use_base_encoder else 0) +
            (landmark_losses) +
            (fuse_generator_losses if self.config.arch.enable_fuse_generator else 0) +
            (phoneme_loss if self.using_phoneme_classifier else 0) +
            (landmark_mouth_dist_loss) +
            (lipreading_loss) +
            (sym_lipreading_loss) +
            (pose_smoothness_loss) + 
            (velocity_loss) + 
            (recon_velocity_loss) +
            (gan_loss)
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

    def load_random_template(self):
        temp_len = 0
        while temp_len < 9:
            random_key = random.choice(list(self.templates.keys()))
            templates = self.templates[random_key]
            temp_len = len(templates)

        return templates

    def inject_expressions(self, expression_series, jaw_series, eyelid_series, series_start_idx, series_end_idx, series_indices, flame_feats, img, masks, series_len, sl_idx, audio_feats=None, batch=None):
        # Shorten the series to match the expression sequence length
        min_length = min(series_end_idx - series_start_idx, len(expression_series))
        sampled_indices = series_indices[:min_length]
        
        # Replace the expression parameters with the template expressions
        flame_feats['expression_params'][sampled_indices, :self.config.arch.num_expression] = \
            torch.Tensor(expression_series[:min_length, :self.config.arch.num_expression]).to(self.device)

        # Replace audio features if necessary
        if audio_feats is not None:
            batch['audio_feat'][sl_idx] = audio_feats[:min_length]
            flame_feats['jaw_params'][sampled_indices] = jaw_series[:min_length].to(self.device)
            flame_feats['eyelid_params'][sampled_indices] = eyelid_series[:min_length].to(self.device)

        # Keep the original shape and pose information for the shortened series
        # flame_feats['shape_params'][series_indices[:min_length]] = flame_feats['shape_params'][series_indices[:min_length]]
        # flame_feats['pose_params'][series_indices[:min_length]] = flame_feats['pose_params'][series_indices[:min_length]]
        
        # Shorten the batch-related structures to match the new series length
        series_indices_new = torch.ones(len(img)).bool()
        series_indices_new[series_start_idx + min_length:series_end_idx] = False

        img = img[series_indices_new]
        masks = masks[series_indices_new]
        for k, v in flame_feats.items():
            flame_feats[k] = flame_feats[k][series_indices_new]
        
        # Update the new series length after truncation
        series_len[sl_idx] = min_length
        series_indices = list(range(series_start_idx, series_start_idx + min_length))

        return flame_feats, img, masks, series_len, series_indices, batch

    def augment_series_new(self, img, masks, flame_feats, series_len, batch, augment_scale=0.20):                            
        num_series = len(series_len)
        series_start_idx = 0
        all_sampled = [None] * num_series  # Ensure we track indices correctly for each sequence

        # Clone necessary tensors
        all_expressions_copy = torch.clone(flame_feats['expression_params'])
        all_jaws_copy = torch.clone(flame_feats['jaw_params'])
        all_eyelids_copy = torch.clone(flame_feats['eyelid_params'])
        all_batch_audio = [torch.clone(x) for x in batch["audio_feat"]]

        # Iterate through each series
        for i in range(len(series_len)):
            series_end_idx = series_start_idx + series_len[i]
            series_indices = list(range(series_start_idx, series_end_idx))

            # Check if audio is zeroed out
            audio_zeroed_out = torch.all(batch["audio_feat"][i] == 0)

            if not audio_zeroed_out:  # Use audio-based augmentation (neighbor swapping)
                # Get the neighbor's length
                neighbor_length = series_len[(i + 1) % len(series_len)]

                # Sample `neighbor_length` number of indices from the original series_indices
                sampled_indices = torch.tensor(random.sample(series_indices, neighbor_length), device=self.device)

                # Swap expressions with the neighbor
                neighbor_start_idx = series_end_idx % len(img)
                neighbor_end_idx = neighbor_start_idx + neighbor_length
                neighbor_indices = list(range(neighbor_start_idx, neighbor_end_idx))

                neighbor_expressions = all_expressions_copy[neighbor_indices]
                neighbor_jaws = all_jaws_copy[neighbor_indices]
                neighbor_eyelids = all_eyelids_copy[neighbor_indices]
                neighbor_audio = all_batch_audio[(i + 1) % len(series_len)]

                flame_feats, img, masks, series_len, series_indices, batch = self.inject_expressions(
                    neighbor_expressions,
                    neighbor_jaws,
                    neighbor_eyelids,
                    series_start_idx,
                    series_end_idx,
                    series_indices,  # Use sampled subset instead of full sequence
                    flame_feats,
                    img,
                    masks,
                    series_len,
                    i,
                    neighbor_audio,
                    batch
                )

                # Store the sampled subset (not the neighbor's indices directly)
                all_sampled[i] = sampled_indices

            else:  # Apply non-audio-based augmentations
                augmentations = ["random_expression", "zero_expression", "permutation_noise", "template_injection"]
                available_augmentations = augmentations.copy()

                # Sample an augmentation for this series
                sampled_aug = random.choice(available_augmentations)
                available_augmentations.remove(sampled_aug)

                augmentation_type = sampled_aug
                feats_dim = flame_feats['expression_params'].size(1)

                if augmentation_type == "random_expression":
                    param_mask = torch.bernoulli(torch.ones((len(series_indices), feats_dim)) * 0.5).to(self.device)
                    new_expressions = (torch.randn((len(series_indices), feats_dim)).to(self.device)) * \
                                    (1 + 2 * torch.rand((len(series_indices), 1)).to(self.device)) * \
                                    param_mask * augment_scale + \
                                    flame_feats['expression_params'][series_indices]
                    flame_feats['expression_params'][series_indices] = torch.clamp(new_expressions, -4.0, 4.0)

                elif augmentation_type == "zero_expression":
                    flame_feats['expression_params'][series_indices] *= 0.0

                elif augmentation_type == "permutation_noise":
                    flame_feats['expression_params'][series_indices] = \
                        (0.25 + 1.25 * torch.rand((len(series_indices), 1)).to(self.device)) * \
                        flame_feats['expression_params'][series_indices][torch.randperm(len(series_indices))]

                elif augmentation_type == "template_injection":
                    expression_series = self.load_random_template()
                    min_length = min(len(series_indices), len(expression_series))
                    flame_feats['expression_params'][series_indices[:min_length], :self.config.arch.num_expression] = \
                        torch.Tensor(expression_series[:min_length, :self.config.arch.num_expression]).to(self.device)

                    flame_feats['shape_params'][series_indices[:min_length]] = \
                        flame_feats['shape_params'][series_indices[:min_length]]
                    flame_feats['pose_params'][series_indices[:min_length]] = \
                        flame_feats['pose_params'][series_indices[:min_length]]

                    series_indices_new = torch.ones(len(img)).bool()
                    series_indices_new[series_start_idx + min_length:series_end_idx] = False

                    img = img[series_indices_new]
                    masks = masks[series_indices_new]
                    for k, v in flame_feats.items():
                        flame_feats[k] = flame_feats[k][series_indices_new]

                    series_len[i] = min_length
                    series_indices = list(range(series_start_idx, series_start_idx + min_length))

                flame_feats['jaw_params'][series_indices] += \
                    torch.randn(flame_feats['jaw_params'][series_indices].size()).to(self.device) * 0.2 * augment_scale
                flame_feats['jaw_params'][series_indices][..., 0] = \
                    torch.clamp(flame_feats['jaw_params'][series_indices][..., 0], 0.0, 0.5)

                # Store the correctly updated indices in all_sampled
                all_sampled[i] = torch.tensor(series_indices, device=self.device)

            # Ensure eyelid augmentations apply regardless of the case
            if self.config.arch.use_eyelids:
                flame_feats['eyelid_params'][series_indices] += \
                    (-1 + 2 * torch.rand(size=flame_feats['eyelid_params'][series_indices].size()).to(self.device)) * \
                    0.25 * augment_scale
                flame_feats['eyelid_params'][series_indices] = \
                    torch.clamp(flame_feats['eyelid_params'][series_indices], 0.0, 1.0)

            series_start_idx += series_len[i]

        # Convert `all_sampled` to a concatenated tensor after modification
        all_sampled = torch.cat(all_sampled) if all_sampled else torch.tensor([], device=self.device)

        # Detach tensors to avoid unnecessary gradients
        for key in ['expression_params', 'pose_params', 'shape_params', 'jaw_params', 'eyelid_params']:
            flame_feats[key] = flame_feats[key].detach()

        return img, masks, flame_feats, series_len, batch, all_sampled

    def augment_series(self, img, masks, flame_feats, series_len, batch, augment_scale=0.20):                            
        # Handle series
        num_series = len(series_len)
        series_start_idx = 0
        all_sampled = [] # NEED TO ACCOUNT FOR IN NON AUDIO

        all_expressions_copy = torch.clone(flame_feats['expression_params'])
        all_jaws_copy = torch.clone(flame_feats['jaw_params'])
        all_eyelids_copy = torch.clone(flame_feats['eyelid_params'])
        all_batch_audio = [torch.clone(x) for x in batch["audio_feat"]]

        if self.use_audio:
            # Shiting all expressions to their neighbor and wrapping around
            all_expressions_copy = torch.clone(flame_feats['expression_params'])
            all_jaws_copy = torch.clone(flame_feats['jaw_params'])
            all_eyelids_copy = torch.clone(flame_feats['eyelid_params'])
            all_batch_audio = [torch.clone(x) for x in batch["audio_feat"]]

            sampled_start = sampled_start = torch.cumsum(torch.tensor([0] + series_len[:-1]), dim=0)
            sampled_end = sampled_start + min(series_len)
            for s, e in zip (sampled_start, sampled_end):
                all_sampled.append(torch.arange(s, e))
            all_sampled = torch.cat(all_sampled)

            for i in range(len(series_len)):
                og_series_len = series_len[i]
                series_end_idx = series_start_idx + og_series_len
                series_indices = list(range(series_start_idx, series_end_idx))

                neighbor_start_idx = series_end_idx % len(img)
                neighbor_end_idx = neighbor_start_idx + series_len[(i + 1) % len(series_len)]
                neighbor_indices = list(range(neighbor_start_idx, neighbor_end_idx))

                neighbor_expressions = all_expressions_copy[neighbor_indices]
                neighbor_jaws = all_jaws_copy[neighbor_indices]
                neighbor_eyelids = all_eyelids_copy[neighbor_indices]
                neighbor_audio = all_batch_audio[(i + 1) % len(series_len)]

                flame_feats, img, masks, series_len, series_indices, batch = self.inject_expressions(
                    neighbor_expressions,
                    neighbor_jaws,
                    neighbor_eyelids,
                    series_start_idx,
                    series_end_idx,
                    series_indices,
                    flame_feats,
                    img,
                    masks,
                    series_len,
                    i,
                    neighbor_audio,
                    batch
                )

                # if self.config.arch.use_eyelids:
                #     flame_feats['eyelid_params'][series_indices] += (-1 + 2 * torch.rand(size=flame_feats['eyelid_params'][series_indices].size()).to(self.device)) * 0.25 * augment_scale
                #     flame_feats['eyelid_params'][series_indices] = torch.clamp(flame_feats['eyelid_params'][series_indices], 0.0, 1.0)
                
                series_start_idx += series_len[i]
        else:
            augmentations = ["random_expression", "zero_expression", "permutation_noise", "template_injection"]
        
            sampled_augmentations = []
            available_augmentations = augmentations.copy()
            all_sampled = None

            # Sample augmentations modulo 4
            for i in range(num_series):
                if len(available_augmentations) == 0:  # If we've exhausted the set, reset it
                    available_augmentations = augmentations.copy()
                
                # Randomly sample one augmentation from the remaining ones
                sampled_aug = random.choice(available_augmentations)
                sampled_augmentations.append(sampled_aug)
                
                # Remove the sampled augmentation to ensure it's not picked again in this group of 4
                available_augmentations.remove(sampled_aug)

            for i, series_length in enumerate(series_len):
                series_end_idx = series_start_idx + series_length
                series_indices = list(range(series_start_idx, series_end_idx))

                # Determine which augmentation to apply
                augmentation_type = sampled_augmentations[i]
                feats_dim = flame_feats['expression_params'].size(1)

                # Apply augmentation for each series
                if augmentation_type == "random_expression":
                    param_mask = torch.bernoulli(torch.ones((len(series_indices), feats_dim)) * 0.5).to(self.device)
                    new_expressions = (torch.randn((len(series_indices), feats_dim)).to(self.device)) * (1 + 2 * torch.rand((len(series_indices), 1)).to(self.device)) * param_mask * augment_scale + flame_feats['expression_params'][series_indices]
                    flame_feats['expression_params'][series_indices] = torch.clamp(new_expressions, -4.0, 4.0) + (0 + 0.2 * torch.rand((len(series_indices), 1)).to(self.device)) * torch.randn((len(series_indices), feats_dim)).to(self.device) * augment_scale

                elif augmentation_type == "zero_expression":
                    flame_feats['expression_params'][series_indices] *= 0.0
                    flame_feats['expression_params'][series_indices] += (0 + 0.2 * torch.rand((len(series_indices), 1)).to(self.device)) * torch.randn((len(series_indices), feats_dim)).to(self.device) * augment_scale

                elif augmentation_type == "permutation_noise":
                    flame_feats['expression_params'][series_indices] = (0.25 + 1.25 * torch.rand((len(series_indices), 1)).to(self.device)) * flame_feats['expression_params'][series_indices][torch.randperm(len(series_indices))] + \
                                                                (0 + 0.2 * torch.rand((len(series_indices), 1)).to(self.device)) * torch.randn((len(series_indices), feats_dim)).to(self.device) * augment_scale

                elif augmentation_type == "template_injection":
                    # Load the expression series from a template
                    expression_series = self.load_random_template()
                    # Shorten the series to match the expression sequence length
                    min_length = min(len(series_indices), len(expression_series))
                    
                    # Replace the expression parameters with the template expressions
                    flame_feats['expression_params'][series_indices[:min_length], :self.config.arch.num_expression] = \
                        torch.Tensor(expression_series[:min_length, :self.config.arch.num_expression]).to(self.device)

                    # Keep the original shape and pose information for the shortened series
                    flame_feats['shape_params'][series_indices[:min_length]] = flame_feats['shape_params'][series_indices[:min_length]]
                    flame_feats['pose_params'][series_indices[:min_length]] = flame_feats['pose_params'][series_indices[:min_length]]
                    
                    # Shorten the batch-related structures to match the new series length
                    series_indices_new = torch.ones(len(img)).bool()
                    series_indices_new[series_start_idx + min_length:series_end_idx] = False

                    img = img[series_indices_new]
                    masks = masks[series_indices_new]
                    for k, v in flame_feats.items():
                        flame_feats[k] = flame_feats[k][series_indices_new]
                    
                    # Update the new series length after truncation
                    series_len[i] = min_length
                    series_indices = list(range(series_start_idx, series_start_idx + min_length))

                else:
                    print("huh.")

                # Apply general noise to jaw and eyelid parameters across all paths
                flame_feats['jaw_params'][series_indices] += torch.randn(flame_feats['jaw_params'][series_indices].size()).to(self.device) * 0.2 * augment_scale
                flame_feats['jaw_params'][series_indices][..., 0] = torch.clamp(flame_feats['jaw_params'][series_indices][..., 0], 0.0, 0.5)

            if self.config.arch.use_eyelids:
                flame_feats['eyelid_params'][series_indices] += (-1 + 2 * torch.rand(size=flame_feats['eyelid_params'][series_indices].size()).to(self.device)) * 0.25 * augment_scale
                flame_feats['eyelid_params'][series_indices] = torch.clamp(flame_feats['eyelid_params'][series_indices], 0.0, 1.0)

            # Update series_start_idx for the next series
            series_start_idx += series_len[i]

        # Detach all the parameters
        flame_feats['expression_params'] = flame_feats['expression_params'].detach()
        flame_feats['pose_params'] = flame_feats['pose_params'].detach()
        flame_feats['shape_params'] = flame_feats['shape_params'].detach()
        flame_feats['jaw_params'] = flame_feats['jaw_params'].detach()
        flame_feats['eyelid_params'] = flame_feats['eyelid_params'].detach()

        # print(batch["audio_feat"][0].shape, batch["audio_feat"][1].shape, flame_feats['expression_params'].shape)

        return img, masks, flame_feats, series_len, batch, all_sampled

    def step2(self, encoder_output, batch, batch_idx, series_len, phase='train'):
        img = batch['img'].clone()
        masks = batch['mask'].clone()
        
        # number of multiple versions for the second path
        Ke = self.config.train.Ke
        
        # start from the same encoder output and add noise to expression params
        # hard clone flame_feats
        dont_clone = ['expression_residuals_down', 'res_series_len', 'expression_residuals_final', 'shape_residuals_down', 'pose_residuals_down', 'pose_residuals_final']
        flame_feats = {}
        for k, v in encoder_output.items():
            if k in dont_clone:
                continue

            tmp = v.clone().detach()
            flame_feats[k] = torch.cat(Ke * [tmp], dim=0)

        # Use the augment_series function to apply the augmentations on the series
        img, masks, flame_feats, series_len, batch, sampled_indices = self.augment_series(img, masks, flame_feats, series_len, batch)

        B, C, H, W = img.shape

        # after defining param augmentation, we can render the new faces
        with torch.no_grad():
            flame_output = self.flame.forward(encoder_output)
            rendered_output = self.renderer.forward(flame_output['vertices'], encoder_output['cam'])
            flame_output.update(rendered_output)
     
            # render the tweaked face
            flame_output_2nd_path = self.flame.forward(flame_feats)
            renderer_output_2nd_path = self.renderer.forward(flame_output_2nd_path['vertices'], flame_feats['cam'], landmarks_fan=flame_output_2nd_path['landmarks_fan'], landmarks_mp=flame_output_2nd_path['landmarks_mp'])
            rendered_img_2nd_path = renderer_output_2nd_path['rendered_img'].detach()
            flame_output_2nd_path.update(renderer_output_2nd_path)

            
            # sample points for the image reconstruction

            # with transfer pixel we can use more points if needed in cycle to quickly learn realistic generations!
            tmask_ratio = self.config.train.mask_ratio #* 2.0

            # use the initial flame estimation to sample points from the initial image
            if sampled_indices is not None:
                verts_sampled = flame_output['transformed_vertices'][sampled_indices]
            else:
                verts_sampled = flame_output['transformed_vertices'][:len(img)]

            if self.use_series_pixel_sampling:
                points1, sampled_coords = masking_utils.mesh_based_mask_uniform_faces_series(verts_sampled, 
                                                                        flame_faces=self.flame.faces_tensor,
                                                                        face_probabilities=self.face_probabilities,
                                                                        series_len=series_len,
                                                                        mask_ratio=tmask_ratio)
            else:
                points1, sampled_coords = masking_utils.mesh_based_mask_uniform_faces(verts_sampled, 
                                                                        flame_faces=self.flame.faces_tensor,
                                                                        face_probabilities=self.face_probabilities,
                                                                        mask_ratio=tmask_ratio)

            
           
            # apply repeat on sampled_coords elements
            sampled_coords['sampled_faces_indices'] = sampled_coords['sampled_faces_indices'].repeat(Ke, 1)
            sampled_coords['barycentric_coords'] = sampled_coords['barycentric_coords'].repeat(Ke, 1, 1)
            
            # get the sampled points that correspond to the face deformations
            points2, sampled_coords = masking_utils.mesh_based_mask_uniform_faces(renderer_output_2nd_path['transformed_vertices'], 
                                                                     flame_faces=self.flame.faces_tensor,
                                                                     face_probabilities=self.face_probabilities,
                                                                     mask_ratio=tmask_ratio,
                                                                     coords=sampled_coords)

            # Generate a mask to select frames for augmentation (50% chance)
            augment_mask = torch.rand(len(img)) > 0.5

            # Create specific masks for swapping and masking among the augmented frames
            swap_mask = augment_mask & (torch.rand(len(img)) > 0.5)  # 50% of augmented frames for swapping
            mask_mask = augment_mask & (torch.rand(len(img)) > 0.5)  # 50% of augmented frames for masking

            # Apply swap_mouth augmentation
            if self.swap_mouth:
                img_sampled = img.clone()

                series_start = 0
                for s_len in series_len:
                    # Create series-specific mask for swapping
                    series_swap_mask = swap_mask[series_start:series_start + s_len]

                    # Only apply swapping to frames where swap_mask is True
                    if series_swap_mask.any():  # Proceed if there are True values in the mask
                        img_sampled[series_start:series_start + s_len][series_swap_mask] = self.crop_switch_inject_mouth(
                            img_sampled[series_start:series_start + s_len][series_swap_mask],
                            flame_output_2nd_path['landmarks_mp'][series_start:series_start + s_len][series_swap_mask],
                            masks
                        )

                    series_start += s_len
            else:
                img_sampled = img
            

            # transfer pixels from initial image to the new image
            extra_points = masking_utils.transfer_pixels(img_sampled.repeat(Ke, 1, 1, 1), points1.repeat(Ke, 1, 1), points2)
        
            
            rendered_mask = (rendered_img_2nd_path > 0).all(dim=1, keepdim=True).float()
                
        masked_img_2nd_path = masking_utils.masking(img_sampled.repeat(Ke, 1, 1, 1), masks.repeat(Ke, 1, 1, 1), extra_points, self.config.train.mask_dilation_radius, 
                                      rendered_mask=rendered_mask, extra_noise=False, random_mask=0.005)
    
        # Apply mask_mouth augmentation
        if self.mask_mouth:
            series_start = 0
            for s_len in series_len:
                # Create series-specific mask for masking
                series_mask_mask = mask_mask[series_start:series_start + s_len]

                # Only apply masking to frames where mask_mask is True
                if series_mask_mask.any():  # Proceed if there are True values in the mask
                    masked_img_2nd_path[series_start:series_start + s_len][series_mask_mask] = self.apply_mask_mouth(
                        masked_img_2nd_path[series_start:series_start + s_len][series_mask_mask],
                        flame_output_2nd_path['landmarks_mp'][series_start:series_start + s_len][series_mask_mask]
                    )

                series_start += s_len

        if self.config.arch.enable_temporal_generator:
            reconstructed_img_2nd_path = self.smirk_generator(torch.cat([rendered_img_2nd_path, masked_img_2nd_path], dim=1).detach(), series_len)
        else:
            reconstructed_img_2nd_path = self.smirk_generator(torch.cat([rendered_img_2nd_path, masked_img_2nd_path], dim=1).detach())
        
        if self.config.train.freeze_generator_in_second_path:
            reconstructed_img_2nd_path = reconstructed_img_2nd_path.detach()

        if self.use_audio:
            if self.use_phoneme_onehot:
                all_phoneme_onehot = torch.zeros(min(series_len) * len(batch["audio_feat"]), 44).to(batch["audio_feat"][0].device)
                series_start = 0
                for i, phoneme_timestamps in enumerate(reversed(batch["phoneme_timestamps"])):
                    for _, phoneme_id, s_idx, e_idx in phoneme_timestamps:
                        if e_idx <= min(series_len):
                            phoneme_embed = F.one_hot(torch.tensor([phoneme_id] * (e_idx - s_idx)).to(batch["audio_feat"][0].device), num_classes=44)
                            # print("A", all_phoneme_onehot)
                            all_phoneme_onehot[series_start + s_idx:series_start + e_idx] += phoneme_embed
                            # print("B", all_phoneme_onehot)
                    series_start += min(series_len)
                
                # print(all_phoneme_onehot.shape, reconstructed_img_2nd_path.view(Ke * B, C, H, W).shape)
                
                recon_feats = self.tater(reconstructed_img_2nd_path.view(Ke * B, C, H, W), series_len, audio_batch=batch["audio_feat"], phoneme_batch=all_phoneme_onehot)
            else:
                recon_feats = self.tater(reconstructed_img_2nd_path.view(Ke * B, C, H, W), series_len, audio_batch=batch["audio_feat"])
        else:
            recon_feats = self.tater(reconstructed_img_2nd_path.view(Ke * B, C, H, W), series_len)

        flame_output_2nd_path_2 = self.flame.forward(recon_feats)
        rendered_img_2nd_path_2 = self.renderer.forward(flame_output_2nd_path_2['vertices'], recon_feats['cam'])['rendered_img']

        losses = {}
        
        cycle_loss = 1.0 * F.mse_loss(recon_feats['expression_params'], flame_feats['expression_params']) + \
                     10.0 * F.mse_loss(recon_feats['jaw_params'], flame_feats['jaw_params'])
        
        if self.config.arch.use_eyelids:
            cycle_loss += 10.0 * F.mse_loss(recon_feats['eyelid_params'], flame_feats['eyelid_params'])

        if not self.config.train.freeze_generator_in_second_path:                
            cycle_loss += 1.0 * F.mse_loss(recon_feats['shape_params'], flame_feats['shape_params']) 

        losses['cycle_loss']  = cycle_loss
        loss_second_path = losses['cycle_loss'] * self.config.train.loss_weights.cycle_loss

        for key, value in losses.items():
            losses[key] = value.item() if isinstance(value, torch.Tensor) else value


        # ---------------- visualization struct ---------------- #
        
        outputs = {}
        if batch_idx % self.config.train.visualize_every == 0:
            outputs['2nd_path'] = torch.stack([rendered_img_2nd_path.detach().cpu().view(Ke, B,  C, H, W).permute(1, 0 , 2, 3, 4).reshape(-1, C, H, W), 
                                             masked_img_2nd_path.detach().cpu().view(Ke, B, C, H, W).permute(1, 0 , 2, 3, 4).reshape(-1, C, H, W),
                                             reconstructed_img_2nd_path.detach().cpu().view(Ke, B,  C, H, W).permute(1, 0 , 2, 3, 4).reshape(-1, C, H, W), 
                                             rendered_img_2nd_path_2.detach().cpu().view(Ke, B, C, H, W).permute(1, 0 , 2, 3, 4).reshape(-1, C, H, W)], dim=1).reshape(-1, C, H, W)
            
        return outputs, losses, loss_second_path

    def set_base_freeze(self):
        if not self.config.train.optimize_base_expression:
            utils.freeze_module(self.smirk_encoder.expression_encoder, 'expression encoder')
        if not self.config.train.optimize_base_shape:
            utils.freeze_module(self.smirk_encoder.shape_encoder, 'shape encoder')
        if not self.config.train.optimize_base_pose:
            utils.freeze_module(self.smirk_encoder.pose_encoder, 'pose encoder')

    def set_base_unfreeze(self):
        if self.config.train.optimize_base_expression:
            utils.freeze_module(self.smirk_encoder.expression_encoder, 'expression encoder')
        if self.config.train.optimize_base_shape:
            utils.freeze_module(self.smirk_encoder.shape_encoder, 'shape encoder')
        if self.config.train.optimize_base_pose:
            utils.freeze_module(self.smirk_encoder.pose_encoder, 'pose encoder')

    def set_freeze_status(self, config, batch_idx, epoch_idx):
        self.config.train.freeze_encoder_in_first_path = False
        self.config.train.freeze_generator_in_first_path = False
        self.config.train.freeze_discriminator_in_first_path = False

        self.config.train.freeze_encoder_in_second_path = False
        self.config.train.freeze_generator_in_second_path = False

        decision_idx = batch_idx

        self.config.train.freeze_discriminator_in_first_path = decision_idx % 5 < 4
        self.config.train.freeze_generator_in_first_path = decision_idx % 5 == 4
        self.config.train.freeze_encoder_in_first_path = decision_idx % 5 == 4

        self.config.train.freeze_encoder_in_second_path = decision_idx % 2 == 0
        self.config.train.freeze_generator_in_second_path = decision_idx % 2 == 1

    def step(self, batch, batch_idx, epoch, phase='train'):
        if phase == 'train':
            self.train()
            torch.set_grad_enabled(True)
        else:
            self.eval()
            torch.set_grad_enabled(False)

        self.set_base_freeze()
        series_len = [b.shape[0] for b in batch["img"]]

        if phase == "train":
            # Freeze depending on whether we're training the generator or disciminator
            if self.config.train.freeze_encoder_in_first_path:
                utils.freeze_module(self.tater, 'tater')
            if self.config.train.freeze_generator_in_first_path:
                utils.freeze_module(self.smirk_generator, 'fuse generator')
            if self.config.train.freeze_discriminator_in_first_path:
                utils.freeze_module(self.discriminator, 'discriminator')
            
        # print("onqoidwijdio")

        # Apply token masking
        token_mask = None
        if self.token_masking is not None and phase == "train":
            # Mask tokens (using the masking indices if defined)
            mask_size = int(max(series_len) * len(series_len))  # Total number of tokens
            token_mask = torch.zeros(mask_size, dtype=torch.int32)
            mask_start = 0

            if self.token_masking == "Random":
                effective_masking_rate = self.masking_rate
                for s_len in series_len:
                    num_masked = int(s_len * self.masking_rate)
                    masked_idxs = torch.randperm(s_len)[:num_masked] + mask_start
                    token_mask[masked_idxs] = 1
                    mask_start += s_len  # Correct increment
            else:
                mask_start = 0
                total_sequences = len(series_len)
                masking_rates = []  # Store per-sequence masking rates

                for i in range(total_sequences):  # Keep sequence order unchanged
                    phoneme_timestamps = batch["phoneme_timestamps"][i]  # Original reference
                    shuffled_timestamps = random.sample(phoneme_timestamps, len(phoneme_timestamps))  # Create shuffled copy

                    min_masked_tokens = int(0.05 * series_len[i])  # At least 5% of the sequence
                    max_masked_tokens = int(0.2 * series_len[i])  # At most 20% of the sequence

                    num_tokens_to_mask = random.randint(min_masked_tokens, max_masked_tokens)
                    selected_tokens = set()  # Track masked tokens

                    # Step 1: Try probabilistic masking
                    for _, phoneme_id, s_idx, e_idx in shuffled_timestamps:
                        if len(selected_tokens) >= num_tokens_to_mask:
                            break  # Stop if we reached the max masking limit

                        token_idx = random.randint(s_idx, e_idx)  # Sample random token within the timestamp
                        prob = self.phoneme_id_to_mask_prob[phoneme_id]

                        if random.random() <= (prob * self.masking_rate):
                            token_mask[token_idx + mask_start] = 1
                            selected_tokens.add(token_idx)

                    # Step 2: If under-masked, force additional tokens from available phoneme timestamps
                    if len(selected_tokens) < min_masked_tokens:
                        remaining_tokens_needed = min_masked_tokens - len(selected_tokens)

                        available_tokens = [
                            random.randint(s_idx, e_idx)
                            for _, _, s_idx, e_idx in shuffled_timestamps
                            if random.randint(s_idx, e_idx) not in selected_tokens
                        ]

                        additional_tokens = available_tokens[:remaining_tokens_needed]  # Take only what's available
                        for token_idx in additional_tokens:
                            token_mask[token_idx + mask_start] = 1
                            selected_tokens.add(token_idx)

                    # Step 3: If still under-masked, randomly sample across the entire sequence
                    if len(selected_tokens) < min_masked_tokens:
                        remaining_tokens_needed = min_masked_tokens - len(selected_tokens)
                        all_possible_tokens = list(range(mask_start, mask_start + series_len[i]))
                        
                        backup_tokens = random.sample(all_possible_tokens, remaining_tokens_needed)
                        for token_idx in backup_tokens:
                            token_mask[token_idx] = 1
                            selected_tokens.add(token_idx)

                    # Compute final masking rate for this sequence
                    sequence_masking_rate = len(selected_tokens) / series_len[i] if series_len[i] > 0 else 0
                    masking_rates.append(sequence_masking_rate)

                    mask_start += max(series_len)  # Correct increment

                # Compute the average masking portion across sequences
                effective_masking_rate = sum(masking_rates) / total_sequences if total_sequences > 0 else 0
                token_mask = token_mask.reshape(len(series_len), -1)
                # print(token_mask.shape, token_mask)

        # print("WEEEEE", batch["audio_feat"][0].shape)

        # Apply modality dropout
        video_mask=None
        audio_mask=None
        if self.modality_dropout and phase == "train":
            batch_size = len(series_len)

            # Initialize modality masks (1 = keep, 0 = drop)
            audio_mask = torch.ones(batch_size, dtype=torch.bool, device=batch["img"][0].device)
            video_mask = torch.ones(batch_size, dtype=torch.bool, device=batch["img"][0].device)

            for i in range(batch_size):
                drop_audio = random.random() < self.audio_dropout_rate
                drop_video = random.random() < self.video_dropout_rate

                # Ensure at least one modality is active
                if drop_audio and drop_video:
                    drop_audio = False
                    drop_video = False

                audio_mask[i] = not drop_audio
                video_mask[i] = not drop_video

        # print("onqwdqwdqw222oidwijdio")
        
        if self.tater.exp_use_audio:
            if self.use_phoneme_onehot:
                all_phoneme_onehot = torch.zeros(sum(series_len), 44).to(batch["audio_feat"][0].device)
                series_start = 0
                for i, phoneme_timestamps in enumerate(batch["phoneme_timestamps"]):
                    if audio_mask[i]:
                        for _, phoneme_id, s_idx, e_idx in phoneme_timestamps:
                            # print(i, phoneme_id, s_idx, e_idx, all_phoneme_onehot.shape)
                            phoneme_embed = F.one_hot(torch.tensor([phoneme_id] * (e_idx - s_idx)).to(batch["audio_feat"][0].device), num_classes=44)
                            # print(phoneme_embed.shape, all_phoneme_onehot.shape, series_start + s_idx, series_start + e_idx)

                            all_phoneme_onehot[series_start + s_idx:series_start + e_idx] += phoneme_embed
                            # print("UWU", all_phoneme_onehot)
                    series_start += series_len[i]
                all_params = self.tater(batch["img"], series_len, audio_batch=batch["audio_feat"], token_mask=token_mask, video_mask=video_mask, audio_mask=audio_mask, phoneme_batch=all_phoneme_onehot)

            else:
                all_params = self.tater(batch["img"], series_len, audio_batch=batch["audio_feat"], token_mask=token_mask, video_mask=video_mask, audio_mask=audio_mask)
        else:
            all_params = self.tater(batch["img"], series_len, token_mask)

        to_concat = ['flag_landmarks_fan', 'img', 'img_mica', 'landmarks_fan', 'landmarks_mp', 'mask']
        for key in to_concat:
            batch[key] = torch.concat(batch[key])
        
        # print("WUEUH0x")

        outputs1, losses1, loss_first_path, encoder_output = self.step1(batch, all_params, batch_idx, series_len)

        # print("WUEUHx")

        # Recording this as a loss for easy logging
        if self.token_masking is not None:
            losses1["mask_rate"] = effective_masking_rate
        
        if phase == 'train':
            self.optimizers_zero_grad()  # Zero the gradients
            loss_first_path.backward()  # Accumulate gradients
            
            if (batch_idx + 1) % self.accumulate_steps == 0:
                self.optimizers_step(step_encoder=True, step_fuse_generator=True)  # Apply accumulated gradients
                self.scheduler_step()  # Step the scheduler
                self.global_step += 1  # Increment global step
            
            iteration = batch_idx if epoch == 0 else -1
            self.tater.update_residual_scale(iteration)

            if self.config.train.freeze_encoder_in_first_path:
                utils.unfreeze_module(self.tater, 'tater')
            if self.config.train.freeze_generator_in_first_path:
                utils.unfreeze_module(self.smirk_generator, 'fuse generator')
            if self.config.train.freeze_discriminator_in_first_path:
                utils.unfreeze_module(self.discriminator, 'discriminator')
            
        if (self.config.train.loss_weights['cycle_loss'] > 0) and (phase == 'train'):
            if self.config.train.freeze_encoder_in_second_path:
                utils.freeze_module(self.tater, 'tater')
            if self.config.train.freeze_generator_in_second_path:
                utils.freeze_module(self.smirk_generator, 'fuse generator')
                    
            outputs2, losses2, loss_second_path = self.step2(encoder_output, batch, batch_idx, series_len, phase)

            # print("WUEUHx2")
            
            self.optimizers_zero_grad()  # Zero the gradients for second path
            loss_second_path.backward()  # Accumulate gradients

            if not self.config.train.freeze_generator_in_second_path:
                torch.nn.utils.clip_grad_norm_(self.smirk_generator.parameters(), 0.1)

            if (batch_idx + 1) % self.accumulate_steps == 0:
                if self.config.train.optimize_generator:
                    self.optimizers_step(step_encoder=not self.config.train.freeze_encoder_in_second_path, 
                                        step_fuse_generator=not self.config.train.freeze_generator_in_second_path)  # Apply accumulated gradients
                else:
                    self.optimizers_step(step_encoder=True)
                self.scheduler_step()  # Step the scheduler
                self.global_step += 1  # Increment global step

            losses1.update(losses2)
            outputs1.update(outputs2)

            if self.config.train.freeze_encoder_in_second_path:
                utils.unfreeze_module(self.tater, 'tater')
            if self.config.train.freeze_generator_in_second_path:
                utils.unfreeze_module(self.smirk_generator, 'fuse generator')

        losses = losses1

        self.logging(batch_idx, losses, phase)

        return outputs1

    def create_visualizations(self, batch, outputs):
        zero_pose_cam = torch.tensor([7,0,0]).unsqueeze(0).repeat(batch["img"].shape[0], 1).float().to(self.device)

        visualizations = {}
        visualizations['img'] = batch['img']
        visualizations['rendered_img'] = outputs['rendered_img']
        
        base_output = self.base_encode(batch['img'].to(self.device))
        flame_output_base = self.flame.forward(base_output)
        rendered_img_base = self.renderer.forward(flame_output_base['vertices'], base_output['cam'])['rendered_img']
        visualizations['rendered_img_base'] = rendered_img_base
    
        flame_output_zero = self.flame.forward(outputs['encoder_output'], zero_expression=True, zero_pose=True)
        rendered_img_zero = self.renderer.forward(flame_output_zero['vertices'].to(self.device), zero_pose_cam)['rendered_img']
        visualizations['rendered_img_zero'] = rendered_img_zero
    
        if self.config.arch.enable_fuse_generator:
            visualizations['reconstructed_img'] = outputs['reconstructed_img']
            visualizations['masked_1st_path'] = outputs['masked_1st_path']
            visualizations['loss_img'] = outputs['loss_img']

        for key in visualizations.keys():
            visualizations[key] = visualizations[key].detach().cpu()

        if self.config.train.loss_weights['mica_loss'] > 0:   
            mica_output_shape = self.mica(batch['img_mica'])
            mica_output = copy.deepcopy(base_output)
            mica_output['shape_params'] = mica_output_shape['shape_params']

            if self.config.arch.num_shape < 300:
                mica_output['shape_params'] = mica_output['shape_params'][:, :self.config.arch.num_shape]

            flame_output_mica = self.flame.forward(mica_output, zero_expression=True, zero_pose=True)
            rendered_img_mica_zero = self.renderer.forward(flame_output_mica['vertices'], zero_pose_cam)['rendered_img']
            visualizations['rendered_img_mica_zero'] = rendered_img_mica_zero

            visualizations['img_mica'] = batch['img_mica'].reshape(-1, 3, 112, 112)
            visualizations['img_mica'] = F.interpolate(visualizations['img_mica'], self.config.image_size).detach().cpu()

        if self.config.train.loss_weights['cycle_loss'] > 0:
            if '2nd_path' in outputs:
                visualizations['2nd_path'] = outputs['2nd_path']

        visualizations['landmarks_mp'] = outputs['landmarks_mp']
        visualizations['landmarks_mp_gt'] = outputs['landmarks_mp_gt']
        visualizations['landmarks_fan'] = outputs['landmarks_fan']
        visualizations['landmarks_fan_gt'] = outputs['landmarks_fan_gt']

        return visualizations

    def save_split_image(self, image, output_path, cutoff=224 * 60):
        """
        Splits an image along the first dimension into slices of height `cutoff` and saves them.

        Args:
            image (numpy.ndarray): The input image of shape (M, N, 3).
            output_path (str): Full output path to determine the save directory and base name.
            cutoff (int): The height cutoff for each slice (default is 224 * 60 = 13440).
        
        Returns:
            None
        """
        # Split the output path into directory and base name without extension
        save_dir = os.path.dirname(output_path)
        base_name = os.path.splitext(os.path.basename(output_path))[0]

        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Get the total height (M) of the image
        total_height = image.shape[0]
        num_slices = (total_height + cutoff - 1) // cutoff  # Calculate number of slices (ceil division)

        # Split the image and save each slice
        for i in range(num_slices):
            start = i * cutoff
            end = min((i + 1) * cutoff, total_height)
            slice_image = image[start:end]

            # Save the slice with a sequential file name
            save_path = os.path.join(save_dir, f"{base_name}_slice_{i+1}.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(slice_image, cv2.COLOR_RGB2BGR))

    def save_visualizations(self, outputs_array, image_save_path, video_save_path, frame_overlap=0, show_landmarks=False, fps=25):
        all_image_grids = []
        all_video_frames = []

        image_keys = [
            'img', 'img_mica', 'rendered_img_base', 'rendered_img',
            'overlap_image', 'overlap_image_pixels', 'rendered_img_mica_zero',
            'rendered_img_zero', 'masked_1st_path', 'reconstructed_img',
            'loss_img', '2nd_path'
        ]

        for i, outputs in enumerate(outputs_array):  # Iterate over each output dictionary
            nrow = 1

            # Reindex tensors to discard overlap frames for all outputs after the first one
            if i > 0 and frame_overlap > 0:
                for key in image_keys:
                    if key in outputs:
                        outputs[key] = outputs[key][frame_overlap:]

            # Generate overlap images if required tensors are available
            if 'img' in outputs and 'rendered_img' in outputs and 'masked_1st_path' in outputs:
                outputs['overlap_image'] = outputs['img'] * 0.7 + outputs['rendered_img'] * 0.3
                outputs['overlap_image_pixels'] = outputs['img'] * 0.7 + 0.3 * outputs['masked_1st_path']

            # Landmarks visualization if enabled
            if show_landmarks:
                original_img_with_landmarks = batch_draw_keypoints(outputs['img'], outputs['landmarks_mp'], color=(0, 255, 0))
                original_img_with_landmarks = batch_draw_keypoints(original_img_with_landmarks, outputs['landmarks_mp_gt'], color=(0, 0, 255))
                original_img_with_landmarks = batch_draw_keypoints(original_img_with_landmarks, outputs['landmarks_fan'][:, :17], color=(255, 0, 255))
                original_img_with_landmarks = batch_draw_keypoints(original_img_with_landmarks, outputs['landmarks_fan_gt'][:, :17], color=(255, 255, 255))
                original_grid = make_grid_from_opencv_images(original_img_with_landmarks, nrow=nrow)
            else:
                original_grid = make_grid(outputs['img'].detach().cpu(), nrow=nrow)

            grids = [original_grid]

            # Padding function for height alignment
            def pad_to_match_height(tensor, target_height):
                _, H, _ = tensor.shape if tensor.dim() == 3 else tensor.shape[1:3]
                if H < target_height:
                    padding = (0, 0, 0, target_height - H)
                    tensor = F.pad(tensor, padding, mode='constant', value=0)
                return tensor

            max_height = original_grid.shape[1]

            for key in image_keys:
                if key in outputs:
                    grid = make_grid(outputs[key].detach().cpu(), nrow=4 * self.config.train.Ke if key == '2nd_path' else nrow)
                    grid = pad_to_match_height(grid, max_height)
                    grids.append(grid)

            combined_grid = torch.cat(grids, dim=2)
            combined_grid = combined_grid.permute(1, 2, 0).cpu().numpy() * 255.0
            combined_grid = np.clip(combined_grid, 0, 255).astype(np.uint8)
            combined_grid = cv2.cvtColor(combined_grid, cv2.COLOR_RGB2BGR)

            # if combined_grid.shape[1] != 2280:
            #     cv2.imwrite(f"{image_save_path[:-4]}_error_{i}.png", combined_grid)

            all_image_grids.append(combined_grid[:, :, :])

            # Convert tensors to numpy arrays
            img_array = outputs['img'].permute(0, 2, 3, 1).cpu().numpy()
            rendered_img_base_array = outputs['rendered_img_base'].permute(0, 2, 3, 1).cpu().numpy()
            rendered_img_array = outputs['rendered_img'].permute(0, 2, 3, 1).cpu().numpy()

            concatenated_frames = np.concatenate((img_array, rendered_img_base_array, rendered_img_array), axis=2)
            all_video_frames.extend(concatenated_frames)  # Accumulate frames for the final video

        # Concatenate all image grids vertically and save as a single image
        # print([x.shape for x in all_image_grids])
        concatenated_image_grid = np.concatenate(all_image_grids, axis=0)  # Stack vertically
        success = self.save_split_image(concatenated_image_grid, image_save_path)
        
        # if success:
        #     print("Image saved successfully.")
        # else:
        #     print("Failed to save the image.")

        # Write all collected frames to a single video
        height, width, _ = all_video_frames[0].shape
        video_writer = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        for frame in all_video_frames:
            frame_uint8 = (frame * 255).astype(np.uint8)  # Scale frame to uint8 format
            video_writer.write(frame_uint8)

        video_writer.release()

    def load_model(self, resume, load_fuse_generator=True, load_encoder=True, device='cuda', strict_load=False):
        loaded_state_dict = torch.load(resume, map_location=device)
        model_state_dict = self.state_dict()  # Current model state

        print(f'Loading checkpoint from {resume}, load_encoder={load_encoder}, load_fuse_generator={load_fuse_generator}')

        # Filter out keys with mismatched dimensions
        filtered_state_dict = {
            k: v for k, v in loaded_state_dict.items()
            if k in model_state_dict and model_state_dict[k].shape == v.shape
        }

        # Load only valid keys
        self.load_state_dict(filtered_state_dict, strict=False)

        # Report skipped keys
        skipped_keys = [k for k in loaded_state_dict.keys() if k not in filtered_state_dict]
        if skipped_keys:
            print(f"Skipped loading weights for {len(skipped_keys)} keys due to shape mismatches: {skipped_keys}")

        # Helper function to strip the specific prefix from state_dict keys
        def strip_exact_prefix(state_dict, prefix):
            return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

        # Load TATER submodules with strict=True
        if self.config.arch.TATER.Expression.pretrain_path:
            exp_pretrain_dict = torch.load(self.config.arch.TATER.Expression.pretrain_path, map_location=device)

            exp_encoder_state = strip_exact_prefix(exp_pretrain_dict, "tater.expression_encoder.")
            exp_transformer_state = strip_exact_prefix(exp_pretrain_dict, "tater.exp_transformer.")
            exp_layer_state = strip_exact_prefix(exp_pretrain_dict, "tater.exp_layer.")
            exp_layer_down_state = strip_exact_prefix(exp_pretrain_dict, "tater.exp_layer_down.")

            self.tater.expression_encoder.load_state_dict(exp_encoder_state, strict=True)
            self.tater.exp_transformer.load_state_dict(exp_transformer_state, strict=True)
            self.tater.exp_layer.load_state_dict(exp_layer_state, strict=True)
            self.tater.exp_layer_down.load_state_dict(exp_layer_down_state, strict=True)
            
            if self.tater.exp_use_audio:
                exp_layer_down_audio_state = strip_exact_prefix(exp_pretrain_dict, "tater.exp_layer_audio_down.")
                self.tater.exp_layer_audio_down.load_state_dict(exp_layer_down_audio_state, strict=True)
        
        if self.config.arch.TATER.Shape.pretrain_path:
            # print(self.tater.shape_encoder.shape_layers[0].weight)
            shape_pretrain_dict = torch.load(self.config.arch.TATER.Shape.pretrain_path, map_location=device)
            shape_encoder_state = strip_exact_prefix(shape_pretrain_dict, "tater.shape_encoder.")
            shape_transformer_state = strip_exact_prefix(shape_pretrain_dict, "tater.shape_transformer.")
            shape_layer_state = strip_exact_prefix(shape_pretrain_dict, "tater.shape_layer.")
            
            self.tater.shape_encoder.load_state_dict(shape_encoder_state, strict=True)
            self.tater.shape_transformer.load_state_dict(shape_transformer_state, strict=True)
            self.tater.shape_layer.load_state_dict(shape_layer_state, strict=True)
            # print(self.tater.shape_encoder.shape_layers[0].weight)

        if self.config.arch.TATER.Pose.pretrain_path:
            pose_pretrain_dict = torch.load(self.config.arch.TATER.Pose.pretrain_path, map_location=device)
            pose_encoder_state = strip_exact_prefix(pose_pretrain_dict, "tater.pose_encoder.")
            pose_transformer_state = strip_exact_prefix(pose_pretrain_dict, "tater.pose_transformer.")
            pose_layer_state = strip_exact_prefix(pose_pretrain_dict, "tater.pose_layer.")
            
            self.tater.pose_encoder.load_state_dict(pose_encoder_state, strict=True)
            self.tater.pose_transformer.load_state_dict(pose_transformer_state, strict=True)
            self.tater.pose_layer.load_state_dict(pose_layer_state, strict=True)

        self.discriminator.requires_grad_(False).to(self.device)
        with open_url(self.D_cfg.load_path) as f:
            resume_data = load_network_pkl(f)
            copy_params_and_buffers(resume_data['D'], self.discriminator, require_all=False)
        self.discriminator.requires_grad_(True)

    def save_model(self, state_dict, save_path):
        new_state_dict = {}
        for key in list(state_dict.keys()):
            if key.startswith('tater') or key.startswith('smirk_generator'):
                new_state_dict[key] = state_dict[key]

        torch.save(new_state_dict, save_path)
