import torch
from torch import nn
import torch.nn.functional as F
from .models.transformer.temporaltransformer import TemporalTransformer
from .smirk_encoder import SmirkEncoder

def initialize_transformer_xavier(m, scale=1e-4):
    """Xavier initialization for transformer layers with small random values."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        m.weight.data.mul_(scale)  # Scale down weights to be close to zero
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

class TATEREncoder(SmirkEncoder):
    def __init__(self, config, n_exp=50, n_shape=300) -> None:
        super().__init__(n_exp, n_shape)
        self.n_exp = n_exp
        self.n_shape = n_shape

        self.interp_down_residual = config.arch.TATER.interp_down_residual
        self.downsample_rate = config.arch.TATER.downsample_rate

        # Conditional residual scaling parameters
        self.enable_residual_scaling = getattr(config.train, "enable_residual_scaling", False)
        self.initial_residual_scale = getattr(config.train, "initial_residual_scale", 0.1)
        self.max_iterations = getattr(config.train, "max_res_scale_iterations", 15000)
        self.residual_schedule = getattr(config.train, "residual_schedule", "linear")
        self.residual_scale = self.initial_residual_scale  # Start with initial residual scale

        # Transformer initialization (Xavier init with scaling)
        init_xavier = lambda m: initialize_transformer_xavier(m, config.arch.TATER.Expression.init_scale)

        self.use_base_exp = config.arch.TATER.Expression.use_base_encoder
        self.use_latent_exp = config.arch.TATER.Expression.use_latent
        if not self.use_base_exp:
            self.exp_transformer = TemporalTransformer(config.arch.TATER.Expression.Transformer)
            self.exp_emb_size = config.arch.TATER.Expression.linear_size
            self.exp_layer = nn.Linear(self.exp_emb_size, n_exp + 2 + 3)

            if config.arch.TATER.Expression.init_near_zero:
                self.exp_transformer.apply(init_xavier)
                self.exp_transformer.attention_blocks[2].linear2.weight = nn.Parameter(
                    torch.zeros_like(self.exp_transformer.attention_blocks[-1].linear2.weight))

        self.use_base_shape = config.arch.TATER.Shape.use_base_encoder
        self.use_latent_shape = config.arch.TATER.Shape.use_latent
        if not self.use_base_shape:
            self.shape_transformer = TemporalTransformer(config.arch.TATER.Shape.Transformer)
            self.shape_emb_size = config.arch.TATER.Shape.linear_size
            self.shape_layer = nn.Linear(self.shape_emb_size, n_shape)

            if config.arch.TATER.Shape.init_near_zero:
                self.shape_transformer.apply(init_xavier)
                self.shape_transformer.attention_blocks[2].linear2.weight = nn.Parameter(
                    torch.zeros_like(self.shape_transformer.attention_blocks[-1].linear2.weight))

        self.use_base_pose = config.arch.TATER.Pose.use_base_encoder
        self.use_latent_pose = config.arch.TATER.Pose.use_latent
        if not self.use_base_pose:
            self.pose_transformer = TemporalTransformer(config.arch.TATER.Pose.Transformer)
            self.pose_emb_size = config.arch.TATER.Pose.linear_size
            self.pose_layer = nn.Linear(self.pose_emb_size, 6)

            if config.arch.TATER.Pose.init_near_zero:
                self.pose_transformer.apply(init_xavier)
                self.pose_transformer.attention_blocks[2].linear2.weight = nn.Parameter(
                    torch.zeros_like(self.pose_transformer.attention_blocks[-1].linear2.weight))

        self.use_interp_linear_layer = config.arch.TATER.use_interp_linear_layer
        if self.use_interp_linear_layer:
            self.residual_layer = nn.Linear(self.emb_size * 2, self.emb_size)

        self.apply_linear_after_res = config.arch.TATER.use_interp_linear_layer
        self.apply_linear_before_res = not self.apply_linear_after_res


    def update_residual_scale(self, iteration):
        """Update the residual scaling factor based on the current iteration."""
        if not self.enable_residual_scaling or iteration == -1:
            return

        if iteration >= self.max_iterations:
            self.residual_scale = 1.0  # Full residual after max_iterations
        else:
            progress = iteration / self.max_iterations
            if self.residual_schedule == 'linear':
                self.residual_scale = self.initial_residual_scale + progress * (1 - self.initial_residual_scale)
            elif self.residual_schedule == 'cosine':
                self.residual_scale = self.initial_residual_scale + (0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159))) - self.initial_residual_scale)
            elif self.residual_schedule == 'exponential':
                self.residual_scale = self.initial_residual_scale + (1 - torch.exp(-progress) - self.initial_residual_scale)
            else:
                raise ValueError(f"Unsupported residual schedule: {self.residual_schedule}")

        # Ensure the scaling factor is clamped between initial scale and 1
        self.residual_scale = min(1.0, max(self.initial_residual_scale, self.residual_scale))

    def add_residual_to_encodings(self, exp_encodings_all, exp_residual_down, series_len, og_series_len):
        """Apply the residual (optionally scaled) to the residuals before adding them to the original encodings."""
        start_idx = 0
        updated_encodings = exp_encodings_all.clone()

        for i, length in enumerate(og_series_len):
            original_encodings_slice = exp_encodings_all[start_idx:start_idx + length]

            if not self.interp_down_residual:
                # No downsampling: Simply add the residual directly to the original encodings
                residual_slice = exp_residual_down[i]
                if self.enable_residual_scaling:
                    residual_slice *= self.residual_scale  # Apply scaling only if enabled
                updated_encodings[start_idx:start_idx + length] += residual_slice
            else:
                # Downsampling: Use linear interpolation to spread residuals across the original encodings
                residual_slice = self.interpolate_tensor(exp_residual_down[i, :series_len[i]], length, self.downsample_rate)

                if self.use_interp_linear_layer:
                    # Concatenate original and residual slices and pass through the residual layer
                    combined_input = torch.cat([original_encodings_slice, residual_slice], dim=-1)
                    transformed = self.residual_layer(combined_input)
                    updated_encodings[start_idx:start_idx + length] = transformed
                else:
                    if self.enable_residual_scaling:
                        residual_slice *= self.residual_scale  # Apply scaling only if enabled
                    updated_encodings[start_idx:start_idx + length] += residual_slice

            start_idx += length

        return updated_encodings

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

    def interpolate_tensor(self, downsampled_tensor, target_length, sample_rate):
        # Get the number of samples and the feature size from the downsampled tensor
        num_samples, feature_size = downsampled_tensor.shape
        
        # Create an empty tensor to hold the interpolated values
        interpolated = torch.zeros(target_length, feature_size, device=downsampled_tensor.device)
        
        # Indices in the target tensor where the downsampled tensor was sampled
        sample_indices = torch.arange(0, num_samples * sample_rate, sample_rate, device=downsampled_tensor.device)
        
        # Place the known sampled values into the interpolated tensor
        interpolated[sample_indices] = downsampled_tensor

        # Compute weights for interpolation between each pair of sampled points
        weights = torch.linspace(0, 1, steps=sample_rate + 1, device=downsampled_tensor.device)[1:-1]  # Ignore 0 and 1

        # Extract the start and end points for each interval
        start_vals = downsampled_tensor[:-1]  # Shape: [num_samples - 1, feature_size]
        end_vals = downsampled_tensor[1:]     # Shape: [num_samples - 1, feature_size]

        # Compute the interpolated values for each interval
        interpolated_values = (1 - weights[:, None]) * start_vals[:, None, :] + weights[:, None] * end_vals[:, None, :]

        # Reshape the interpolated values into a flat shape to match target positions
        interpolated_values = interpolated_values.reshape(-1, feature_size)

        # Calculate the positions for the interpolated values
        interpolation_indices = torch.cat([
            torch.arange(start + 1, start + sample_rate, device=downsampled_tensor.device)
            for start in sample_indices[:-1]
        ])

        # Insert the interpolated values into the appropriate positions
        interpolated[interpolation_indices] = interpolated_values

        # Handle the tail end of the tensor if the target length isn't perfectly divisible by sample_rate
        last_sampled_idx = sample_indices[-1]
        if last_sampled_idx < target_length - 1:
            remaining_length = target_length - last_sampled_idx - 1  # Number of remaining entries after the last sampled point
            # Generate weights for the remaining interpolation
            tail_weights = torch.linspace(0, 1, steps=remaining_length + 2, device=downsampled_tensor.device)[1:-1]
            last_sample = downsampled_tensor[-1]  # The last sampled point
            next_point = downsampled_tensor[-1]  # For simplicity, interpolate towards this last point
            # Compute interpolated values for the remaining length
            tail_interpolated_values = (1 - tail_weights[:, None]) * last_sample + tail_weights[:, None] * next_point
            
            # Insert the tail interpolated values into the remaining positions
            interpolated[last_sampled_idx + 1:] = tail_interpolated_values

        return interpolated

    def forward(self, img_batch, og_series_len):
        outputs = {}
        if isinstance(img_batch, list):
            img_cat = torch.cat(img_batch)
        else:
            img_cat = img_batch

        if self.use_base_exp:
            outputs.update(self.expression_encoder(img_cat))
        else:
            if self.use_latent_exp:
                exp_encodings_all = self.expression_encoder.encoder(img_cat)
                exp_encodings_all = F.adaptive_avg_pool2d(exp_encodings_all[-1], (1, 1)).squeeze(-1).squeeze(-1)
            else:
                exp_encodings_all = self.expression_encoder(img_cat)
                exp_encodings_all = torch.cat(list(exp_encodings_all.values()), dim=-1)
                exp_encodings_all = F.pad(exp_encodings_all, (0, 1), "constant", 0) # Slighlty too short

            exp_encodings_down, attention_mask, series_len = self.pad_and_create_mask(
                exp_encodings_all,
                og_series_len,
                downsample=self.interp_down_residual,
                sample_rate=self.downsample_rate
            )

            exp_residual_down, exp_class = self.exp_transformer(exp_encodings_down, attention_mask, series_len)

            if self.apply_linear_before_res:
                exp_parameters = self.exp_layer(updated_exp_encodings_all).reshape(exp_encodings_all.size(0), -1)

            updated_exp_encodings_all = self.add_residual_to_encodings(
                exp_encodings_all,
                exp_residual_down,
                series_len,
                og_series_len
            )

            if self.apply_linear_after_res:
                exp_parameters = self.exp_layer(updated_exp_encodings_all).reshape(exp_encodings_all.size(0), -1)

            outputs['expression_params'] = exp_parameters[...,:self.n_exp]
            outputs['expression_residuals_down'] = exp_residual_down
            outputs['res_series_len'] = series_len
            outputs['eyelid_params'] = torch.clamp(exp_parameters[...,self.n_exp:self.n_exp+2], 0, 1)
            outputs['jaw_params'] = torch.cat([F.relu(exp_parameters[...,self.n_exp+2].unsqueeze(-1)) * 2, 
                                           torch.clamp(exp_parameters[...,self.n_exp+3:self.n_exp+5], -.2, .2)], dim=-1)
        
        if self.use_base_shape:
            outputs.update(self.shape_encoder(img_cat))
        else:
            if self.use_latent_shape:
                shape_encodings_all = self.shape_encoder.encoder(img_cat)
                shape_encodings_all = F.adaptive_avg_pool2d(shape_encodings_all[-1], (1, 1)).squeeze(-1).squeeze(-1)
            else:
                shape_encodings_all = self.shape_encoder(img_cat)
                shape_encodings_all = shape_encodings_all["shape_params"]
        
            shape_encodings_down, attention_mask, series_len = self.pad_and_create_mask(
                shape_encodings_all,
                og_series_len,
                downsample=self.interp_down_residual,
                sample_rate=self.downsample_rate
            )

            shape_residual_down, shape_class = self.shape_transformer(shape_encodings_down, attention_mask, series_len)
            shape_embeds_all = [(x[:s] + r[:s]).mean(dim=0)[None].expand(o, -1) for x, r, s, o in zip(shape_encodings_down, shape_residual_down, series_len, og_series_len)]
            shape_embeds_all = torch.cat(shape_embeds_all)
            shape_parameters = self.shape_layer(shape_embeds_all).reshape(shape_embeds_all.size(0), -1)
            outputs['shape_params'] = shape_parameters
            outputs['shape_residuals_down'] = shape_residual_down
            outputs['res_series_len'] = series_len
        
        if self.use_base_pose:
            outputs.update(self.pose_encoder(img_cat))
        else:
            if self.use_latent_pose:
                pose_encodings_all = self.pose_encoder.encoder(img_cat)
                pose_encodings_all = F.adaptive_avg_pool2d(pose_encodings_all[-1], (1, 1)).squeeze(-1).squeeze(-1)
            else:
                pose_encodings_all = self.pose_encoder(img_cat)
                pose_encodings_all = torch.cat(list(pose_encodings_all.values()), dim=-1)

            pose_encodings_down, attention_mask, series_len = self.pad_and_create_mask(
                pose_encodings_all,
                og_series_len,
                downsample=self.interp_down_residual,
                sample_rate=self.downsample_rate
            )

            pose_residual_down, pose_class = self.pose_transformer(pose_encodings_down, attention_mask, series_len)

            if self.apply_linear_before_res:
                pose_parameters = self.pose_layer(updated_pose_encodings_all).reshape(exp_encodings_all.size(0), -1)

            updated_pose_encodings_all = self.add_residual_to_encodings(
                pose_encodings_all,
                pose_residual_down,
                series_len,
                og_series_len
            )

            if self.apply_linear_after_res:
                pose_parameters = self.pose_layer(updated_pose_encodings_all).reshape(exp_encodings_all.size(0), -1)
            
            outputs['pose_params'] = pose_parameters[...,:3]
            outputs['cam'] = pose_parameters[...,3:]
            outputs['pose_residuals_down'] = pose_residual_down
            outputs['res_series_len'] = series_len

        return outputs
