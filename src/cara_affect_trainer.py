import random
import torch
import wandb
import copy
import itertools
import src.utils.utils as utils
import scipy.stats

from torch import nn
from src.base_trainer import BaseTrainer 
from .base_trainer import BaseTrainer
from .cara_encoder import CARAEncoder
from .affect_decoder import AffectDecoder

class CARAAffectTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)

        self.tater = CARAEncoder(
            self.config,
            n_exp=self.config.arch.num_expression,
            n_shape=self.config.arch.num_shape
        )
        self.affect_decoder = AffectDecoder(config)
        self.MSELoss = nn.MSELoss()
        self.HuberLoss = nn.SmoothL1Loss(beta=1.0)

        self.token_masking = self.config.train.token_masking
        self.masking_rate = self.config.train.masking_rate
        self.max_masked = self.config.train.max_masked
        self.min_masked = self.config.train.min_masked
        if not (self.token_masking == "Random" or self.token_masking == "Phoneme"):
            self.token_masking = None

        self.modality_dropout = self.config.train.modality_dropout
        self.video_dropout_rate = self.config.train.video_dropout_rate
        self.audio_dropout_rate = self.config.train.audio_dropout_rate

        self.accumulate_steps = config.train.accumulate_steps if hasattr(config.train, 'accumulate_steps') else 1
        self.global_step = 0  # to track global steps

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

                wandb_log_data[f"{phase}/lr"] = self.scheduler.get_last_lr()[0]
                wandb_log_data[f"{phase}/batch_idx"] = batch_idx

                wandb.log(wandb_log_data)

    def configure_optimizers(self, num_steps, use_default_annealing=False):
        # Adjust total steps for gradient accumulation
        effective_total_steps = num_steps // self.config.train.accumulate_steps // self.config.train.batch_size

        # Define max_lr, min_lr, and warmup steps directly from config
        max_lr = self.config.train.max_lr  # Use max_lr as defined in config
        gen_max_lr = self.config.train.max_lr  # Use unscaled LR values for generator

        # Encoder Optimizer setup
        params = []
        if self.config.train.optimize_base_expression:
            params += list(self.tater.expression_encoder.parameters()) 
        if self.config.train.optimize_tater:
            if not self.config.arch.TATER.Expression.use_base_encoder:
                params += list(self.tater.exp_transformer.parameters())
                if self.config.arch.TATER.Expression.use_linear:
                    params += list(self.tater.exp_layer.parameters())
                if self.config.arch.TATER.Expression.use_linear_downsample:
                    params += list(self.tater.exp_layer_down.parameters())
                if self.config.arch.TATER.Expression.use_audio:
                    params += list(self.tater.residual_linear.parameters())
        params += list(self.affect_decoder.parameters())

        # Initialize the encoder optimizer
        self.optimizer = torch.optim.Adam(params, lr=max_lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=effective_total_steps, eta_min=gen_max_lr
        )

    def scheduler_step(self):
        self.scheduler.step()

    def optimizers_zero_grad(self):
        self.optimizer.zero_grad()

    def optimizers_step(self, step_encoder=True, step_fuse_generator=True):
        self.optimizer.step()

    def train(self):
        self.tater.eval()
    
    def eval(self):
        self.tater.eval()

    def generate_phoneme_token_mask(self, batch, series_len):
        token_mask = None
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

        return token_mask, effective_masking_rate
    
    def generate_audio_video_mask(self, batch, series_len):
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
        
        return audio_mask, video_mask
    
    def pearson_r(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the sample Pearson correlation r for each column in x and y,
        exactly as scipy.stats.pearsonr does internally (ddof=1).
        Returns a [num_features]-shaped tensor.
        """
        # x, y: [N, F]
        assert x.shape == y.shape
        # 1) center
        xm = x.mean(dim=0, keepdim=True)    # [1, F]
        ym = y.mean(dim=0, keepdim=True)    # [1, F]
        x_d = x - xm                         # [N, F]
        y_d = y - ym                         # [N, F]
        # 2) numerator = sum_j (x_j - xm)*(y_j - ym)
        num = (x_d * y_d).sum(dim=0)         # [F]
        # 3) denominator = sqrt( sum_j (x_j - xm)^2 * sum_j (y_j - ym)^2 )
        den = torch.sqrt((x_d**2).sum(dim=0) * (y_d**2).sum(dim=0))  # [F]
        # 4) r = num / den   → matches scipy exactly (NaN if constant input)
        r = num / den
        return r

    def pearson_p(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute two‐tailed p‐values by deferring to scipy.stats.pearsonr,
        which itself computes r in the same sample‐based way and then
        applies the Student’s t–test.
        """
        assert x.shape == y.shape and x.shape[1] == 2, "Expected shape [N, 2]"
        p_vals = []
        for i in range(2):
            x_np = x[:, i].detach().cpu().numpy()
            y_np = y[:, i].detach().cpu().numpy()
            _, p = scipy.stats.pearsonr(x_np, y_np)
            p_vals.append(p)
        return torch.tensor(p_vals)
    
    def relative_performance(self, pred, gt):
        """
        Compute relative performance metrics between predicted and ground-truth scores.
        
        Args:
            pred (torch.Tensor): Predicted scores, shape (B, N)
            gt (torch.Tensor): Ground-truth scores, shape (B, N)
        
        Returns:
            dict: Containing MSE, RMSE, R2, NRMSE_range, and NRMSE_std.
        """
        # Flatten both tensors into shape (B*N,)
        pred_flat = pred.view(-1)
        gt_flat = gt.view(-1)

        # MSE
        mse = torch.mean((pred_flat - gt_flat) ** 2)
        # RMSE
        rmse = torch.sqrt(mse)
        # Variance of ground truth
        var_gt = torch.var(gt_flat, unbiased=False)
        # R-squared
        r2 = 1 - mse / var_gt
        # NRMSE normalized by range
        range_gt = torch.max(gt_flat) - torch.min(gt_flat)
        nrmse_range = rmse / range_gt
        # NRMSE normalized by standard deviation
        std_gt = torch.sqrt(var_gt)
        nrmse_std = rmse / std_gt

        return {
            'mse': mse.item(),
            'rmse': rmse.item(),
            'r2': r2.item(),
            'nrmse_range': nrmse_range.item(),
            'nrmse_std': nrmse_std.item()
        }
    
    def ccc_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        1 - Concordance Correlation Coefficient as a loss.
        """
        mu_p = y_pred.mean(dim=0)
        mu_t = y_true.mean(dim=0)
        cov  = ((y_pred - mu_p) * (y_true - mu_t)).mean(dim=0)
        var_p = ((y_pred - mu_p)**2).mean(dim=0)
        var_t = ((y_true - mu_t)**2).mean(dim=0)
        ccc = 2 * cov / (var_p + var_t + (mu_p - mu_t)**2 + eps)
        # shape [2], one for valence and arousal
        return 1 - ccc.mean()  # scalar loss
    
    def step1(self, batch, affect_scores, batch_idx, series_len):
        losses = {}
        affect_scores_gt = torch.stack(batch["valence_arousal"], dim=0)  # (B,2)

        # compute raw losses
        mse   = self.MSELoss(affect_scores, affect_scores_gt)
        huber = self.HuberLoss(affect_scores, affect_scores_gt)
        ccc_l = self.ccc_loss(affect_scores, affect_scores_gt)

        # mix them: λ_cc*(1-CCC) + (1-λ_cc)*Huber
        λ_cc = 0.5
        loss = λ_cc * ccc_l + (1 - λ_cc) * huber

        # log metrics
        r_v, r_a = self.pearson_r(affect_scores, affect_scores_gt)
        losses.update({
            "Pearson r V": r_v,
            "Pearson r A": r_a,
            "MSE": mse,
            "Huber": huber,
            "CCC_loss": ccc_l,
            "TrainLoss": loss
        })

        # relative performance
        rel = self.relative_performance(affect_scores, affect_scores_gt)
        losses.update({
            "Rel MSE": rel['mse'],
            "Rel RMSE": rel['rmse'],
            "Rel R2": rel['r2'],
            "Rel NRMSE_range": rel['nrmse_range'],
            "Rel NRMSE_std": rel['nrmse_std'],
        })

        return {}, losses, loss
    
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

    def step(self, batch, batch_idx, epoch, phase='train'):
        if phase == 'train':
            self.train()
            torch.set_grad_enabled(True)
        else:
            self.eval()
            torch.set_grad_enabled(False)
        series_len = [b.shape[0] for b in batch["img"]]

        token_mask = None
        if self.token_masking is not None and phase == "train":
            token_mask, effective_masking_rate = self.generate_phoneme_token_mask(batch, series_len)
        
        audio_mask = None
        video_mask = None
        if self.modality_dropout and phase == "train":
            audio_mask, video_mask = self.generate_audio_video_mask(batch, series_len)
    
        if self.tater.exp_use_audio:
            encode_feat = self.tater(
                batch["img"],
                series_len,
                audio_batch=batch["audio_feat"],
                token_mask=token_mask,
                video_mask=video_mask,
                audio_mask=audio_mask
            )
        else:
            encode_feat = self.tater(batch["img"], series_len, token_mask)
    
        affect_scores = self.affect_decoder(encode_feat["exp_class"])
        outputs, losses, loss = self.step1(batch, affect_scores, batch_idx, series_len)

        if self.token_masking is not None:
            losses["mask_rate"] = effective_masking_rate

        if phase == 'train':
            self.optimizers_zero_grad()  # Zero the gradients
            loss.backward()  # Accumulate gradients

            # gradient clipping
            max_norm = getattr(self.config.train, "max_grad_norm", 1.0)
            torch.nn.utils.clip_grad_norm_(
                itertools.chain(self.tater.parameters(),
                                self.affect_decoder.parameters()),
                max_norm
            )

            # ——— NEW: compute gradient norms ———
            total_norm_sq = 0.0
            # iterate over all parameters you care about
            for p in itertools.chain(self.tater.parameters(), self.affect_decoder.parameters()):
                if p.grad is not None:
                    # L2 norm of this param's gradient
                    param_norm = p.grad.data.norm(2)
                    total_norm_sq += param_norm.item() ** 2
            total_grad_norm = total_norm_sq ** 0.5
            # stick it into outputs for logging/printing downstream
            losses['grad_norm_total'] = total_grad_norm
            # ————————————————————————————————
            
            if (batch_idx + 1) % self.accumulate_steps == 0:
                self.optimizers_step(step_encoder=True, step_fuse_generator=True)  # Apply accumulated gradients
                self.scheduler_step()  # Step the scheduler
                self.global_step += 1  # Increment global step
            
            iteration = batch_idx if epoch == 0 else -1
            self.tater.update_residual_scale(iteration)

            self.logging(batch_idx, losses, phase)

        outputs['base_encode'] = encode_feat['base_encode']
        outputs["valence_arousal_out"] = affect_scores
        outputs["valence_arousal_gt"] = torch.cat([x[None] for x in batch["valence_arousal"]])
        return outputs

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

        # Remove the shape and pose encoders
        del self.tater.shape_transformer
        # del self.tater.pose_transformer

    def save_model(self, state_dict, save_path):
        torch.save(state_dict, save_path)



    

