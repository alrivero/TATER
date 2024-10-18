import copy
import cv2
import random
import torch.utils.data
import torch.nn.functional as F
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

class TATERTrainer(SmirkTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.accumulate_steps = config.train.accumulate_steps if hasattr(config.train, 'accumulate_steps') else 1
        self.global_step = 0  # to track global steps

        if self.config.arch.enable_fuse_generator:
            self.smirk_generator = SmirkGenerator(in_channels=6, out_channels=3, init_features=32, res_blocks=5)
        
        self.tater = TATEREncoder(self.config, n_exp=self.config.arch.num_expression, n_shape=self.config.arch.num_shape)
        self.smirk_encoder = self.tater  # Backwards compatibility
        
        self.flame = FLAME(n_exp=self.config.arch.num_expression, n_shape=self.config.arch.num_shape)
        self.renderer = Renderer(render_full_head=False)
        self.setup_losses()

        self.templates = utils.load_templates()
            
        # --------- setup flame masks for sampling --------- #
        self.face_probabilities = masking_utils.load_probabilities_per_FLAME_triangle()

    def configure_optimizers(self, n_steps):
        self.n_steps = n_steps
        encoder_scale = .25

        if hasattr(self, 'encoder_optimizer'):
            for g in self.encoder_optimizer.param_groups:
                g['lr'] = encoder_scale * self.config.train.lr
        else:
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
                    params += list(self.tater.exp_layer.parameters())
                if not self.config.arch.TATER.Shape.use_base_encoder:
                    params += list(self.tater.shape_transformer.parameters())
                    params += list(self.tater.shape_layer.parameters())
                if not self.config.arch.TATER.Pose.use_base_encoder:
                    params += list(self.tater.pose_transformer.parameters())
                    params += list(self.tater.pose_layer.parameters())

            self.encoder_optimizer = torch.optim.Adam(params, lr= encoder_scale * self.config.train.lr)
                
        self.encoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.encoder_optimizer, T_max=n_steps,
                                                                                  eta_min=0.01 * encoder_scale * self.config.train.lr)

        if self.config.arch.enable_fuse_generator and self.config.train.optimize_generator:
            if hasattr(self, 'fuse_generator_optimizer'):
                for g in self.smirk_generator_optimizer.param_groups:
                    g['lr'] = self.config.train.lr
            else:
                self.smirk_generator_optimizer = torch.optim.Adam(self.smirk_generator.parameters(), lr= self.config.train.lr, betas=(0.5, 0.999))

            self.smirk_generator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.smirk_generator_optimizer, T_max=n_steps,
                                                                                    eta_min=0.01 * self.config.train.lr)

    def scheduler_step(self):
        self.encoder_scheduler.step()
        if self.config.arch.enable_fuse_generator and self.config.train.optimize_generator:
            self.smirk_generator_scheduler.step()

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

    def optimizers_step(self, step_encoder=True, step_fuse_generator=True):
        if step_encoder:
            self.encoder_optimizer.step()
        if step_fuse_generator and self.config.arch.enable_fuse_generator and self.config.train.optimize_generator:
            self.smirk_generator_optimizer.step()

    def create_base_encoder(self):
        self.base_exp_encoder = copy.deepcopy(self.tater.expression_encoder)
        self.base_shape_encoder = copy.deepcopy(self.tater.shape_encoder)
        self.base_pose_encoder = copy.deepcopy(self.tater.pose_encoder)
        self.base_exp_encoder.eval()
    
    def base_encode(self, img):
        expression_outputs = self.base_exp_encoder(img)
        shape_outputs = self.base_shape_encoder(img)
        pose_outputs = self.base_pose_encoder(img)

        outputs = {}
        outputs.update(expression_outputs)
        outputs.update(shape_outputs)
        outputs.update(pose_outputs)

        return outputs

    def step1(self, batch, encoder_output, batch_idx):
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
            base_output = {key[0]: torch.zeros(B, key[1]).to(self.config.device) for key in zip(['expression_params', 'shape_params', 'jaw_params'], [self.config.arch.num_expression, self.config.arch.num_shape, 3])}

        losses['expression_regularization'] = torch.mean((encoder_output['expression_params'] - base_output['expression_params'])**2)
        losses['shape_regularization'] = torch.mean((encoder_output['shape_params'] - base_output['shape_params'])**2)
        losses['pose_regularization'] = torch.mean((encoder_output['pose_params'] - base_output['pose_params'])**2)
        losses['cam_regularization'] = torch.mean((encoder_output['cam'] - base_output['cam'])**2)
        losses['jaw_regularization'] = torch.mean((encoder_output['jaw_params'] - base_output['jaw_params'])**2)
        losses['eyelid_regularization'] = torch.mean((encoder_output['eyelid_params'] - base_output['eyelid_params'])**2)

        if batch_idx % 10 == 0:
            print(f"Ours: {encoder_output['jaw_params'].mean(dim=0)}")
            print(f"Base: {base_output['jaw_params'].mean(dim=0)}")
            print(f"Ours: {encoder_output['eyelid_params'].mean(dim=0)}")
            print(f"Base: {base_output['eyelid_params'].mean(dim=0)}")

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

            if self.config.train.loss_weights['emotion_loss'] > 0:
                if self.config.train.optimize_generator:
                    for param in self.smirk_generator.parameters():
                        param.requires_grad_(False)
                self.smirk_generator.eval()
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
                                    losses['mica_loss'] * self.config.train.loss_weights['mica_loss']

        expression_losses = losses['expression_regularization'] * self.config.train.loss_weights['expression_regularization'] + \
                            losses['jaw_regularization'] * self.config.train.loss_weights['jaw_regularization']
        
        landmark_losses = losses['landmark_loss_fan'] * self.config.train.loss_weights['landmark_loss'] + \
                            losses['landmark_loss_mp'] * self.config.train.loss_weights['landmark_loss'] 

        fuse_generator_losses = losses['perceptual_vgg_loss'] * self.config.train.loss_weights['perceptual_vgg_loss'] + \
                                losses['reconstruction_loss'] * self.config.train.loss_weights['reconstruction_loss'] + \
                                losses['emotion_loss'] * self.config.train.loss_weights['emotion_loss']
               
        loss_first_path = (
            (shape_losses if self.config.train.optimize_base_shape else 0) +
            (expression_losses if self.config.train.optimize_base_expression else 0) +
            (landmark_losses) +
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

    def load_random_template(self):
        temp_len = 0
        while temp_len < 9:
            random_key = random.choice(list(self.templates.keys()))
            templates = self.templates[random_key]
            temp_len = len(templates)

        return templates

    def augment_series(self, img, masks, flame_feats, series_len, augment_scale=0.5):                            
        # Handle series
        num_series = len(series_len)
        series_start_idx = 0

        augmentations = ["random_expression", "zero_expression", "permutation_noise", "template_injection"]
        sampled_augmentations = []
        available_augmentations = augmentations.copy()

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
                param_mask = torch.bernoulli(torch.ones((len(series_indices), feats_dim)) * 0.5).to(self.config.device)
                new_expressions = (torch.randn((len(series_indices), feats_dim)).to(self.config.device)) * (1 + 2 * torch.rand((len(series_indices), 1)).to(self.config.device)) * param_mask * augment_scale + flame_feats['expression_params'][series_indices]
                flame_feats['expression_params'][series_indices] = torch.clamp(new_expressions, -4.0, 4.0) + (0 + 0.2 * torch.rand((len(series_indices), 1)).to(self.config.device)) * torch.randn((len(series_indices), feats_dim)).to(self.config.device) * augment_scale

            elif augmentation_type == "zero_expression":
                flame_feats['expression_params'][series_indices] *= 0.0
                flame_feats['expression_params'][series_indices] += (0 + 0.2 * torch.rand((len(series_indices), 1)).to(self.config.device)) * torch.randn((len(series_indices), feats_dim)).to(self.config.device) * augment_scale

            elif augmentation_type == "permutation_noise":
                flame_feats['expression_params'][series_indices] = (0.25 + 1.25 * torch.rand((len(series_indices), 1)).to(self.config.device)) * flame_feats['expression_params'][series_indices][torch.randperm(len(series_indices))] + \
                                                            (0 + 0.2 * torch.rand((len(series_indices), 1)).to(self.config.device)) * torch.randn((len(series_indices), feats_dim)).to(self.config.device) * augment_scale

            elif augmentation_type == "template_injection":
                # Load the expression series from a template
                expression_series = self.load_random_template()
                # Shorten the series to match the expression sequence length
                min_length = min(len(series_indices), len(expression_series))
                
                # Replace the expression parameters with the template expressions
                flame_feats['expression_params'][series_indices[:min_length], :self.config.arch.num_expression] = \
                    torch.Tensor(expression_series[:min_length, :self.config.arch.num_expression]).to(self.config.device)

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
            flame_feats['jaw_params'][series_indices] += torch.randn(flame_feats['jaw_params'][series_indices].size()).to(self.config.device) * 0.2 * augment_scale
            flame_feats['jaw_params'][series_indices][..., 0] = torch.clamp(flame_feats['jaw_params'][series_indices][..., 0], 0.0, 0.5)

            if self.config.arch.use_eyelids:
                flame_feats['eyelid_params'][series_indices] += (-1 + 2 * torch.rand(size=flame_feats['eyelid_params'][series_indices].size()).to(self.config.device)) * 0.25 * augment_scale
                flame_feats['eyelid_params'][series_indices] = torch.clamp(flame_feats['eyelid_params'][series_indices], 0.0, 1.0)

            # Update series_start_idx for the next series
            series_start_idx += series_len[i]

        # Detach all the parameters
        flame_feats['expression_params'] = flame_feats['expression_params'].detach()
        flame_feats['pose_params'] = flame_feats['pose_params'].detach()
        flame_feats['shape_params'] = flame_feats['shape_params'].detach()
        flame_feats['jaw_params'] = flame_feats['jaw_params'].detach()
        flame_feats['eyelid_params'] = flame_feats['eyelid_params'].detach()

        return img, masks, flame_feats, series_len

    def step2(self, encoder_output, batch, batch_idx, series_len, phase='train'):
        img = batch['img'].clone()
        masks = batch['mask'].clone()
        
        # number of multiple versions for the second path
        Ke = self.config.train.Ke
        
        # start from the same encoder output and add noise to expression params
        # hard clone flame_feats
        flame_feats = {}
        for k, v in encoder_output.items():
            if k == 'expression_residuals_down' or k == 'res_series_len':
                continue

            tmp = v.clone().detach()
            flame_feats[k] = torch.cat(Ke * [tmp], dim=0)

        # Use the augment_series function to apply the augmentations on the series
        img, masks, flame_feats, series_len = self.augment_series(img, masks, flame_feats, series_len)

        B, C, H, W = img.shape

        # after defining param augmentation, we can render the new faces
        with torch.no_grad():
            flame_output = self.flame.forward(encoder_output)
            rendered_output = self.renderer.forward(flame_output['vertices'], encoder_output['cam'])
            flame_output.update(rendered_output)
     
            # render the tweaked face
            flame_output_2nd_path = self.flame.forward(flame_feats)
            renderer_output_2nd_path = self.renderer.forward(flame_output_2nd_path['vertices'], encoder_output['cam'][:len(img)])
            rendered_img_2nd_path = renderer_output_2nd_path['rendered_img'].detach()

            
            # sample points for the image reconstruction

            # with transfer pixel we can use more points if needed in cycle to quickly learn realistic generations!
            tmask_ratio = self.config.train.mask_ratio #* 2.0

            # use the initial flame estimation to sample points from the initial image
            points1, sampled_coords = masking_utils.mesh_based_mask_uniform_faces(flame_output['transformed_vertices'][:len(img)], 
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
            

            # transfer pixels from initial image to the new image
            extra_points = masking_utils.transfer_pixels(img.repeat(Ke, 1, 1, 1), points1.repeat(Ke, 1, 1), points2)
        
            
            rendered_mask = (rendered_img_2nd_path > 0).all(dim=1, keepdim=True).float()
                
        masked_img_2nd_path = masking_utils.masking(img.repeat(Ke, 1, 1, 1), masks.repeat(Ke, 1, 1, 1), extra_points, self.config.train.mask_dilation_radius, 
                                      rendered_mask=rendered_mask, extra_noise=True, random_mask=0.005)
        
        reconstructed_img_2nd_path = self.smirk_generator(torch.cat([rendered_img_2nd_path, masked_img_2nd_path], dim=1).detach())
        if self.config.train.freeze_generator_in_second_path:
            reconstructed_img_2nd_path = reconstructed_img_2nd_path.detach()

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

        if batch_idx % 10 == 0:
            print(encoder_output["jaw_params"].mean(dim=0))
            print(recon_feats["jaw_params"].mean(dim=0))
            print(flame_feats["jaw_params"].mean(dim=0))


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

    def step(self, batch, batch_idx, epoch, phase='train'):
        if phase == 'train':
            self.train()
            torch.set_grad_enabled(True)
        else:
            self.eval()
            torch.set_grad_enabled(False)

        self.set_base_freeze()
        series_len = [b.shape[0] for b in batch["img"]]
        all_params = self.tater(batch["img"], series_len)

        to_concat = ['flag_landmarks_fan', 'img', 'img_mica', 'landmarks_fan', 'landmarks_mp', 'mask']
        for key in to_concat:
            batch[key] = torch.concat(batch[key])

        outputs1, losses1, loss_first_path, encoder_output = self.step1(batch, all_params, batch_idx)
        
        if phase == 'train':
            self.optimizers_zero_grad()  # Zero the gradients
            loss_first_path.backward()  # Accumulate gradients
            
            if (batch_idx + 1) % self.accumulate_steps == 0:
                self.optimizers_step(step_encoder=True, step_fuse_generator=True)  # Apply accumulated gradients
                self.scheduler_step()  # Step the scheduler
                self.global_step += 1  # Increment global step
            
            iteration = batch_idx if epoch == 0 else -1
            self.tater.update_residual_scale(iteration)

            
        if (self.config.train.loss_weights['cycle_loss'] > 0) and (phase == 'train'):
            if self.config.train.freeze_encoder_in_second_path:
                utils.freeze_module(self.tater, 'tater')
            if self.config.train.freeze_generator_in_second_path:
                utils.freeze_module(self.smirk_generator, 'fuse generator')
                    
            outputs2, losses2, loss_second_path = self.step2(encoder_output, batch, batch_idx, series_len, phase)
            
            self.optimizers_zero_grad()  # Zero the gradients for second path
            loss_second_path.backward()  # Accumulate gradients

            if not self.config.train.freeze_generator_in_second_path:
                torch.nn.utils.clip_grad_norm_(self.smirk_generator.parameters(), 0.1)

            if (batch_idx + 1) % self.accumulate_steps == 0:
                self.optimizers_step(step_encoder=not self.config.train.freeze_encoder_in_second_path, 
                                     step_fuse_generator=not self.config.train.freeze_generator_in_second_path)  # Apply accumulated gradients
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
        zero_pose_cam = torch.tensor([7,0,0]).unsqueeze(0).repeat(batch["img"].shape[0], 1).float().to(self.config.device)

        visualizations = {}
        visualizations['img'] = batch['img']
        visualizations['rendered_img'] = outputs['rendered_img']
        
        base_output = self.base_encode(batch['img'])
        flame_output_base = self.flame.forward(base_output)
        rendered_img_base = self.renderer.forward(flame_output_base['vertices'], base_output['cam'])['rendered_img']
        visualizations['rendered_img_base'] = rendered_img_base
    
        flame_output_zero = self.flame.forward(outputs['encoder_output'], zero_expression=True, zero_pose=True)
        rendered_img_zero = self.renderer.forward(flame_output_zero['vertices'].to(self.config.device), zero_pose_cam)['rendered_img']
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

    def save_renders_as_video(self, output_dict, output_path, fps=30):
        img_tensor = output_dict['img']  
        rendered_img_base_tensor = output_dict['rendered_img_base']  
        rendered_img_tensor = output_dict['rendered_img']  
        
        assert img_tensor.shape[0] == rendered_img_base_tensor.shape[0] == rendered_img_tensor.shape[0]
        
        img_array = img_tensor.permute(0, 2, 3, 1).cpu().numpy()  
        rendered_img_base_array = rendered_img_base_tensor.permute(0, 2, 3, 1).cpu().numpy()  
        rendered_img_array = rendered_img_tensor.permute(0, 2, 3, 1).cpu().numpy()  

        concatenated_frames = np.concatenate((img_array, rendered_img_base_array, rendered_img_array), axis=2)

        height, width, _ = concatenated_frames[0].shape
        video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        for frame in concatenated_frames:
            frame_uint8 = (frame * 255).astype(np.uint8)
            video_writer.write(frame_uint8)

        video_writer.release()

    def save_visualizations(self, outputs, save_path, show_landmarks=False):
        nrow = 1

        # Generate overlap images if the required tensors are available
        if 'img' in outputs and 'rendered_img' in outputs and 'masked_1st_path' in outputs:
            outputs['overlap_image'] = outputs['img'] * 0.7 + outputs['rendered_img'] * 0.3
            outputs['overlap_image_pixels'] = outputs['img'] * 0.7 +  0.3 * outputs['masked_1st_path']

        # Create landmarks visualization if required
        if show_landmarks:
            original_img_with_landmarks = batch_draw_keypoints(outputs['img'], outputs['landmarks_mp'], color=(0, 255, 0))
            original_img_with_landmarks = batch_draw_keypoints(original_img_with_landmarks, outputs['landmarks_mp_gt'], color=(0, 0, 255))
            original_img_with_landmarks = batch_draw_keypoints(original_img_with_landmarks, outputs['landmarks_fan'][:, :17], color=(255, 0, 255))
            original_img_with_landmarks = batch_draw_keypoints(original_img_with_landmarks, outputs['landmarks_fan_gt'][:, :17], color=(255, 255, 255))
            original_grid = make_grid_from_opencv_images(original_img_with_landmarks, nrow=nrow)
        else:
            original_grid = make_grid(outputs['img'].detach().cpu(), nrow=nrow)

        # Define the keys for images and set appropriate row sizes
        image_keys = ['img', 'img_mica', 'rendered_img_base', 'rendered_img', 
                    'overlap_image', 'overlap_image_pixels', 'rendered_img_mica_zero', 
                    'rendered_img_zero', 'masked_1st_path', 'reconstructed_img', 
                    'loss_img', '2nd_path']

        grids = [original_grid]
        
        # Padding function to ensure same number of rows
        def pad_to_match_height(tensor, target_height):
            if tensor.dim() == 3:
                _, H, _ = tensor.shape  # Handle 3D case (C, H, W)
            elif tensor.dim() == 4:
                _, _, H, _ = tensor.shape  # Handle 4D case (B, C, H, W)
            else:
                raise ValueError(f"Unexpected tensor dimensions: {tensor.dim()}")

            if H < target_height:
                padding = (0, 0, 0, target_height - H)
                tensor = F.pad(tensor, padding, mode='constant', value=0)
            return tensor

        # Get the maximum height (rows) among the images
        max_height = original_grid.shape[1]

        for key in image_keys:
            if key in outputs:
                grid = make_grid(outputs[key].detach().cpu(), nrow=4 * self.config.train.Ke if key == '2nd_path' else nrow)
                grid = pad_to_match_height(grid, max_height)  # Pad the grid to match the maximum height
                grids.append(grid)

        # Concatenate the grids and save as an image
        combined_grid = torch.cat(grids, dim=2)  # Concatenate along width
        combined_grid = combined_grid.permute(1, 2, 0).cpu().numpy() * 255.0
        combined_grid = np.clip(combined_grid, 0, 255).astype(np.uint8)
        combined_grid = cv2.cvtColor(combined_grid, cv2.COLOR_RGB2BGR)

        cv2.imwrite(save_path, combined_grid)
        self.save_renders_as_video(outputs, save_path[:-4] + ".mp4")


    def save_model(self, state_dict, save_path):
        new_state_dict = {}
        for key in list(state_dict.keys()):
            if key.startswith('tater') or key.startswith('smirk_generator'):
                new_state_dict[key] = state_dict[key]

        torch.save(new_state_dict, save_path)
