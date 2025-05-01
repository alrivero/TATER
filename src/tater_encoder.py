import torch
from torch import nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from .models.transformer.temporaltransformer import TemporalTransformer
from .models.transformer.crossattentiontransformer_debug import CrossAttentionTransformer
from .smirk_encoder import SmirkEncoder
import torch.nn.functional as F
from .utils.debug_utils import save_attention_maps

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

D_EXP_DEFAULT = 960
D_AUDIO_DEFAULT = 768
D_SHAPE_DEFAULT = 960
D_POSE_DEFAULT = 576

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

        self.apply_linear_after_res = config.arch.TATER.use_interp_linear_layer
        self.apply_linear_before_res = not self.apply_linear_after_res

        self.use_base_exp = config.arch.TATER.Expression.use_base_encoder
        self.use_latent_exp = config.arch.TATER.Expression.use_latent
        self.add_to_flame_exp = config.arch.TATER.Expression.add_to_flame
        self.exp_use_audio = False
        self.use_exp_linear_downsample = False
        if not self.use_base_exp:
            self.exp_use_audio = config.arch.TATER.Expression.use_audio
            self.exp_emb_size = config.arch.TATER.Expression.linear_size
            self.use_exp_linear = config.arch.TATER.Expression.use_linear
            self.use_exp_linear_downsample = config.arch.TATER.Expression.use_linear_downsample

            if self.exp_use_audio:
                self.exp_transformer = CrossAttentionTransformer(config.arch.TATER.Expression.CrossAttentionTransformer)
                self.residual_linear = nn.Linear(self.exp_emb_size, n_exp + 2 + 3)
            else:
                self.exp_transformer = TemporalTransformer(config.arch.TATER.Expression.Transformer)

            if self.use_exp_linear:
                exp_offset = 0 # 1 if self.apply_linear_before_res else 0
                if self.exp_use_audio:
                    self.exp_layer = nn.Linear(2 * (self.exp_emb_size), n_exp + 2 + 3 + exp_offset)
                else:
                    self.exp_layer = nn.Linear(self.exp_emb_size, n_exp + 2 + 3 + exp_offset)

            if self.use_exp_linear_downsample:
                if self.exp_use_audio:
                    self.exp_layer_down = nn.Linear(D_EXP_DEFAULT, self.exp_emb_size)
                    self.exp_layer_audio_down = nn.Linear(D_AUDIO_DEFAULT, self.exp_emb_size)
                else:
                    self.exp_layer_down = nn.Linear(D_EXP_DEFAULT, self.exp_emb_size)

            if config.arch.TATER.Expression.init_near_zero:
                self.exp_transformer.apply(init_xavier)
                torch.nn.init.zeros_(self.exp_layer.weight)
                torch.nn.init.zeros_(self.exp_layer.bias)

            if self.exp_use_audio:
                self.exp_transformer.encoder_2.apply(init_xavier)
                self.exp_transformer.encoder_2_cross_layers.apply(init_xavier)
                self.exp_transformer.encoder_1_cross_layers.apply(init_xavier)
                self.exp_transformer.concat_layers.apply(init_xavier)
                torch.nn.init.zeros_(self.exp_layer.weight)
                torch.nn.init.zeros_(self.exp_layer.bias)

        self.use_base_shape = config.arch.TATER.Shape.use_base_encoder
        self.use_latent_shape = config.arch.TATER.Shape.use_latent
        self.add_to_flame_shape = config.arch.TATER.Shape.add_to_flame
        self.use_shape_linear_downsample = False
        if not self.use_base_shape:
            self.shape_transformer = TemporalTransformer(config.arch.TATER.Shape.Transformer)
            self.shape_emb_size = config.arch.TATER.Shape.linear_size
            self.use_shape_linear = config.arch.TATER.Shape.use_linear
            self.use_shape_linear_downsample = config.arch.TATER.Shape.use_linear_downsample

            if self.use_shape_linear:
                self.shape_layer = nn.Linear(self.shape_emb_size, n_shape)
            
            if self.use_shape_linear_downsample:
                self.shape_layer_down = nn.Linear(D_SHAPE_DEFAULT, self.shape_emb_size)

            if config.arch.TATER.Shape.init_near_zero:
                self.shape_transformer.apply(init_xavier)
                self.shape_transformer.attention_blocks[2].linear2.weight = nn.Parameter(
                    torch.zeros_like(self.shape_transformer.attention_blocks[-1].linear2.weight))

        self.use_base_pose = config.arch.TATER.Pose.use_base_encoder
        self.use_latent_pose = config.arch.TATER.Pose.use_latent
        self.add_to_flame_pose = config.arch.TATER.Pose.add_to_flame
        self.use_pose_linear_downsample = False
        if not self.use_base_pose:
            self.pose_transformer = TemporalTransformer(config.arch.TATER.Pose.Transformer)
            self.pose_emb_size = config.arch.TATER.Pose.linear_size
            self.use_pose_linear = config.arch.TATER.Pose.use_linear
            self.use_pose_linear_downsample = config.arch.TATER.Pose.use_linear_downsample

            if self.use_pose_linear:
                self.pose_layer = nn.Linear(self.pose_emb_size, 18)

            if self.use_pose_linear_downsample:
                self.pose_layer_down = nn.Linear(D_POSE_DEFAULT, self.pose_emb_size)

            if config.arch.TATER.Pose.init_near_zero:
                self.pose_transformer.apply(init_xavier)
                self.pose_transformer.attention_blocks[2].linear2.weight = nn.Parameter(
                    torch.zeros_like(self.pose_transformer.attention_blocks[-1].linear2.weight))

        self.use_interp_linear_layer = config.arch.TATER.use_interp_linear_layer
        if self.use_interp_linear_layer:
            self.residual_layer = nn.Linear(self.emb_size * 2, self.emb_size)

        self.device = next(self.parameters()).device

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

        final_residuals = []
        for i, length in enumerate(og_series_len):
            original_encodings_slice = exp_encodings_all[start_idx:start_idx + length]

            if not self.interp_down_residual:
                # No downsampling: Simply add the residual directly to the original encodings
                residual_slice = exp_residual_down[i][:length]
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

            final_residuals.append(residual_slice)
            start_idx += length

        final_residuals = torch.cat(final_residuals)

        return updated_encodings, final_residuals

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
        padded_tensor = torch.zeros((num_tensors, max_len, combined_tensor.shape[1]), dtype=combined_tensor.dtype).to(self.device)
        
        # Initialize the attention mask (T/F) with False
        attention_mask = torch.zeros((num_tensors, max_len), dtype=torch.bool).to(self.device)
        
        start_idx = 0
        for i, length in enumerate(lengths):
            # Fill the padded tensor with the corresponding slice from the combined_tensor
            padded_tensor[i, :length] = tensor_list[i]
            
            # Set the attention mask to True for the actual tokens
            attention_mask[i, :length] = 1
            
            # Move to the next slice in the combined tensor
            start_idx += length
        
        # Invert the mask for use with nn.TransformerEncoderLayer (True for padding positions)
        key_padding_mask = ~(attention_mask.bool())  # Now True for padding, False for valid tokens
        
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
    
    def rotation_matrix_to_euler(self, matrix):
        """Convert rotation matrix to Euler angles."""
        sy = torch.sqrt(matrix[:, 0, 0] ** 2 + matrix[:, 1, 0] ** 2)
        singular = sy < 1e-6
        x = torch.atan2(matrix[:, 2, 1], matrix[:, 2, 2])
        y = torch.atan2(-matrix[:, 2, 0], sy)
        z = torch.atan2(matrix[:, 1, 0], matrix[:, 0, 0])

        x_sing = torch.atan2(-matrix[:, 1, 2], matrix[:, 1, 1])
        z_sing = torch.zeros_like(x)
        
        x = torch.where(singular, x_sing, x)
        z = torch.where(singular, z_sing, z)
        return torch.stack((x, y, z), dim=1)

    def euler_to_rotation_matrix(self, euler_angles):
        """
        Convert Euler angles (B, N, 3) to rotation matrices (B, N, 3, 3).
        Euler angles are assumed to be in ZYX order.
        """
        alpha, beta, gamma = euler_angles[..., 0], euler_angles[..., 1], euler_angles[..., 2]
        
        # Compute sine and cosine of angles
        cos_a, sin_a = torch.cos(alpha), torch.sin(alpha)
        cos_b, sin_b = torch.cos(beta), torch.sin(beta)
        cos_g, sin_g = torch.cos(gamma), torch.sin(gamma)
        
        # Rotation matrices for each Euler angle set
        R = torch.stack([
            cos_g * cos_b, cos_g * sin_b * sin_a - sin_g * cos_a, cos_g * sin_b * cos_a + sin_g * sin_a,
            sin_g * cos_b, sin_g * sin_b * sin_a + cos_g * cos_a, sin_g * sin_b * cos_a - cos_g * sin_a,
            -sin_b,        cos_b * sin_a,                      cos_b * cos_a
        ], dim=-1).reshape(*euler_angles.shape[:-1], 3, 3)
        
        return R

    def convert_and_apply_residual(self, base_euler, residual, series_lengths):
        """
        Convert base Euler angles to rotation matrices, concatenate them, and apply the residual rotation matrices.

        Args:
        - base_euler: Tensor of shape (M, 6) containing two sets of Euler angles.
        - residual: Tensor of shape (B, N, 18) containing two concatenated residual rotation matrices.
        - series_lengths: List of lists where each inner list contains series lengths for each batch B_i.

        Returns:
        - combined: Tensor of shape (M, 18) representing the combined rotation matrices.
        """
        # Initialize the start index for base_euler slicing
        start_idx = 0

        # List to store the combined results
        combined_results = []

        for b_idx in range(residual.shape[0]):  # Iterate over each batch B_i
            batch_residual = residual[b_idx]  # Shape: (N, 18)
            s_len = series_lengths[b_idx]

            # Extract the relevant slices
            base_euler_slice = base_euler[start_idx : start_idx + s_len]  # Shape: (s_len, 6)
            residual_slice = batch_residual[:s_len]  # Shape: (s_len, 18)

            # Split base Euler angles into two sets
            euler_angles_1 = base_euler_slice[..., :3]  # Shape: (s_len, 3)
            euler_angles_2 = base_euler_slice[..., 3:]  # Shape: (s_len, 3)

            # Convert Euler angles to rotation matrices
            R1_base = self.euler_to_rotation_matrix(euler_angles_1)  # Shape: (s_len, 3, 3)
            R2_base = self.euler_to_rotation_matrix(euler_angles_2)  # Shape: (s_len, 3, 3)

            # Reshape residual to separate into two rotation matrices
            residual_slice = residual_slice.reshape(-1, 2, 3, 3)  # Shape: (s_len, 2, 3, 3)
            R1_residual = residual_slice[:, 0, :, :]  # Shape: (s_len, 3, 3)
            R2_residual = residual_slice[:, 1, :, :]  # Shape: (s_len, 3, 3)

            # Apply the residual rotations via LHS multiplication
            R1_combined = torch.matmul(R1_residual, R1_base)  # Shape: (s_len, 3, 3)
            R2_combined = torch.matmul(R2_residual, R2_base)  # Shape: (s_len, 3, 3)

            # Reshape and concatenate the combined rotation matrices
            R1_combined_flat = R1_combined.reshape(-1, 9)  # Shape: (s_len, 9)
            R2_combined_flat = R2_combined.reshape(-1, 9)  # Shape: (s_len, 9)

            combined_slice = torch.cat([R1_combined_flat, R2_combined_flat], dim=-1)  # Shape: (s_len, 18)

            # Append the combined slice to the results
            combined_results.append(combined_slice)

            # Increment the start index
            start_idx += s_len

        # Concatenate all combined results into a single 2D tensor
        combined = torch.cat(combined_results, dim=0)  # Shape: (M, 18)

        return combined

    def forward(self, img_batch, og_series_len, audio_batch=None, token_mask=None, video_mask=None, audio_mask=None, phoneme_batch=None):
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
            
            if self.add_to_flame_exp:
                base_encoding = self.expression_encoder(img_cat)
                base_encoding = torch.cat(list(base_encoding.values()), dim=-1)
            else:
                base_encoding = self.expression_encoder.encoder(img_cat)
                base_encoding = F.adaptive_avg_pool2d(base_encoding[-1], (1, 1)).squeeze(-1).squeeze(-1)
            
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
                # print("WII", audio_batch.shape)
                if self.use_exp_linear_downsample:
                    audio_batch = self.exp_layer_audio_down(audio_batch)
                    
                if phoneme_batch is not None:
                    # print("WTF", img_cat.shape, audio_batch.shape, phoneme_batch.shape, series_len)
                    audio_batch = torch.cat([audio_batch, phoneme_batch], dim=-1)

                audio_encodings_down, _, series_len = self.pad_and_create_mask(
                    audio_batch,
                    og_series_len,
                    downsample=self.interp_down_residual,
                    sample_rate=self.downsample_rate
                )

                seq, tok, res_seq, res_tok, attn = self.exp_transformer(exp_encodings_down, audio_encodings_down, attention_mask, series_len, token_mask, return_attn=True)

                # visualise first item in batch
                saved = save_attention_maps(attn, save_dir="debug_attn", batch_idx=0)
                print("Saved", len(saved), "attention heat-maps to", saved[0].parent)

                print(kill_me_lol)
                # exp_residual_out, exp_class, exp_video_residual, exp_video_class = self.exp_transformer(exp_encodings_down, audio_encodings_down, attention_mask, series_len, token_mask)
            else:
                exp_residual_out, exp_class = self.exp_transformer(exp_encodings_down, attention_mask, series_len, token_mask)

            outputs['expression_residuals_down'] = exp_residual_out

            if self.apply_linear_before_res and self.use_exp_linear:
                exp_residual_down = self.exp_layer(exp_residual_out)
            else:
                exp_encodings_down = exp_residual_out

            # if self.exp_use_audio:
            #     exp_residual_down += self.residual_linear(exp_video_residual)

            updated_exp_encodings_all, exp_residuals_final = self.add_residual_to_encodings(
                base_encoding,
                exp_residual_down,
                series_len,
                og_series_len
            )

            if self.apply_linear_after_res and self.use_exp_linear:
                exp_parameters = self.exp_layer(updated_exp_encodings_all)
            else:
                exp_parameters = updated_exp_encodings_all
            exp_parameters = exp_parameters.reshape(exp_parameters.size(0), -1)

            outputs['expression_params'] = exp_parameters[...,:self.n_exp]
            outputs['expression_residuals_final'] = exp_residuals_final
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

            if self.add_to_flame_exp:
                base_shape_encoding = self.shape_encoder(img_cat)["shape_params"]
                base_shape_encoding = [base_shape_encoding[:o].mean(dim=0)[None].expand(o, -1) for o in og_series_len]
                base_shape_encoding = torch.cat(base_shape_encoding)
            else:
                base_shape_encoding = self.shape_encoder.encoder(img_cat)
                base_shape_encoding = F.adaptive_avg_pool2d(exp_encodings_all[-1], (1, 1)).squeeze(-1).squeeze(-1)

            base_shape_encoding = [x[:o].mean(dim=0)[None].expand(o, -1) for x, o in zip(base_shape_encoding, og_series_len)]
            base_shape_encoding = torch.cat(base_shape_encoding)

            if self.use_shape_linear_downsample:
                shape_encodings_all = self.shape_layer_down(shape_encodings_all)
        
            shape_encodings_down, attention_mask, series_len = self.pad_and_create_mask(
                shape_encodings_all,
                og_series_len,
                downsample=self.interp_down_residual,
                sample_rate=self.downsample_rate
            )

            shape_residual_down, shape_class = self.shape_transformer(shape_encodings_down, attention_mask, series_len, token_mask)
            shape_residual_down = [r[:s].mean(dim=0)[None].expand(o, -1) for r, s, o in zip(shape_residual_down, series_len, og_series_len)]
            shape_residual_down = torch.cat(shape_residual_down)
            shape_residual_flame = self.shape_layer(shape_residual_down).reshape(shape_residual_down.size(0), -1)

            # print(base_shape_encoding.unique(), shape_residual_flame.unique())

            shape_parameters = base_shape_encoding + shape_residual_flame

            # print(shape_parameters.unique())
            # print("\n")

            outputs['shape_params'] = shape_parameters
            outputs['shape_residuals_down'] = shape_residual_down
            outputs['res_series_len'] = series_len

            shape_parameters.sum()
        
        if self.use_base_pose:
            outputs.update(self.pose_encoder(img_cat))
        else:
            if self.use_latent_pose:
                pose_encodings_all = self.pose_encoder.encoder(img_cat)
                pose_encodings_all = F.adaptive_avg_pool2d(pose_encodings_all[-1], (1, 1)).squeeze(-1).squeeze(-1)
            else:
                pose_encodings_all = self.pose_encoder(img_cat)
                pose_encodings_all = torch.cat(list(pose_encodings_all.values()), dim=-1)

            if self.add_to_flame_exp:
                base_pose_encoding = self.pose_encoder(img_cat)
                base_pose_encoding = torch.cat(list(base_pose_encoding.values()), dim=-1)
            else:
                base_pose_encoding = self.pose_encoder.encoder(img_cat)
                base_pose_encoding = F.adaptive_avg_pool2d(base_pose_encoding[-1], (1, 1)).squeeze(-1).squeeze(-1)

            if self.use_pose_linear_downsample:
                pose_encodings_all = self.pose_layer_down(pose_encodings_all)

            pose_encodings_down, attention_mask, series_len = self.pad_and_create_mask(
                pose_encodings_all,
                og_series_len,
                downsample=self.interp_down_residual,
                sample_rate=self.downsample_rate
            )

            pose_residual_out, pose_class = self.pose_transformer(pose_encodings_down, attention_mask, series_len, token_mask)
            outputs['pose_residuals_down'] = pose_residual_out

            if self.apply_linear_before_res and self.use_pose_linear:
                pose_residual_down = self.pose_layer(pose_residual_out)
            else:
                pose_residual_down = pose_residual_out

            # Convert each 9D residual block to Euler angles, handling (B, N, D) shape
            B, N, _ = pose_residual_down.shape
            converted_residuals = torch.zeros((B, N, 6), device=pose_residual_down.device)  # To hold converted residuals

            for b in range(B):
                for n in range(N):
                    euler_angles_set = []
                    for j in range(2):  # Two sets of 9D rotations
                        start_idx = j * 9
                        pose_vec = pose_residual_down[b, n, start_idx:start_idx + 9].view(3, 3)  # Reshape each 9D block to 3x3
                        u, _, vh = torch.linalg.svd(pose_vec, full_matrices=False)
                        rotation_matrix = torch.matmul(u, vh)
                        euler_angles = self.rotation_matrix_to_euler(rotation_matrix.unsqueeze(0)).squeeze(0)  # Convert to Euler angles
                        euler_angles_set.append(euler_angles)
                    converted_residuals[b, n, :] = torch.cat(euler_angles_set)  # Store both sets of Euler angles

            outputs["pose_params_mat"] = self.convert_and_apply_residual(base_pose_encoding, pose_residual_down, og_series_len)

            # print(base_pose_encoding.unique(), converted_residuals.unique())

            updated_pose_encodings_all, pose_residuals_final = self.add_residual_to_encodings(
                base_pose_encoding,
                converted_residuals,
                series_len,
                og_series_len
            )

            if self.apply_linear_after_res and self.use_pose_linear:
                pose_parameters = self.pose_layer(updated_pose_encodings_all)
            else:
                pose_parameters = updated_pose_encodings_all
            pose_parameters = pose_parameters.reshape(pose_parameters.size(0), -1)

            outputs['pose_params'] = pose_parameters[...,:3]
            outputs['pose_residuals_final'] = pose_residuals_final
            outputs['cam'] = pose_parameters[...,3:]
            outputs['res_series_len'] = series_len

        return outputs
