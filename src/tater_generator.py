import torch
import torch.nn as nn
from collections import OrderedDict
from .smirk_generator import SmirkGenerator
import torch.nn.functional as F

class TATERGenerator(SmirkGenerator):

    def __init__(self, in_channels=3, out_channels=1, init_features=16, res_blocks=3):
        super(TATERGenerator, self).__init__(in_channels, out_channels, init_features, res_blocks)

        # Add temporal and spatial consistency layers
        self.temporal_spatial = nn.Sequential(
            nn.Conv3d(
                in_channels=init_features * 16,
                out_channels=init_features * 16,
                kernel_size=(3, 1, 1),  # Temporal kernel size set to 3
                stride=(1, 1, 1),
                padding=(1, 0, 0),  # Adjust padding for kernel size 3
                bias=False
            ),
            nn.Conv3d(
                in_channels=init_features * 16,
                out_channels=init_features * 16,
                kernel_size=(1, 3, 3),
                stride=(1, 1, 1),
                padding=(0, 1, 1),
                bias=False
            ),
            nn.BatchNorm3d(init_features * 16),
            nn.ReLU(inplace=True)
        )

        # Learnable weight for residual connection
        self.residual_weight = nn.Parameter(torch.tensor(1e-5, dtype=torch.float32))

        # Initialize temporal-spatial layers as identity
        self.initialize_temporal_spatial()

    def initialize_temporal_spatial(self):
        for layer in self.temporal_spatial:
            if isinstance(layer, nn.Conv3d):
                self.initialize_identity_3d(layer)
            elif isinstance(layer, nn.BatchNorm3d):
                self.initialize_batchnorm_3d(layer)

    @staticmethod
    def initialize_identity_3d(conv3d_layer):
        with torch.no_grad():
            weight = conv3d_layer.weight
            weight.zero_()  # Fill all weights with zeros

            # Set the center element of the kernel to 1 for each channel
            c_out, c_in, k_t, k_h, k_w = weight.shape
            center_t = k_t // 2
            center_h = k_h // 2
            center_w = k_w // 2

            for i in range(c_out):
                for j in range(c_in):
                    if i == j:  # Identity only when output channel matches input channel
                        weight[i, j, center_t, center_h, center_w] = 1.0

            # Set bias to zero
            if conv3d_layer.bias is not None:
                conv3d_layer.bias.zero_()

    @staticmethod
    def initialize_batchnorm_3d(batchnorm_layer):
        with torch.no_grad():
            batchnorm_layer.weight.fill_(1.0)  # Set gamma to 1
            batchnorm_layer.bias.zero_()      # Set beta to 0

    def forward(self, x, series_len):
        B, C, H, W = x.shape  # Input shape

        # Encoding
        x = x.view(-1, C, H, W)  # Flatten batch
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))  # Bottleneck shape: (B, C, H, W)

        # Apply 3D convolution across series at the bottleneck
        kernel_size_b = 3  # Updated temporal kernel size
        processed_series = []
        start_idx = 0

        for length in series_len:
            # Extract series
            series = bottleneck[start_idx : start_idx + length]  # Shape: (length, C, H, W)

            # Pad along batch dimension if series is too short
            if length < kernel_size_b:
                pad_amount = kernel_size_b - length
                series = F.pad(series, (0, 0, 0, 0, 0, 0, 0, pad_amount))  # Pad batch dim

            # Reshape for 3D convolution
            original_series = series.clone()  # Save the original series for residual
            series = series.unsqueeze(0)  # Add dummy batch dim: (1, length, C, H, W)
            series = series.permute(0, 2, 1, 3, 4)  # Rearrange to (1, C, length, H, W)

            # Apply temporal-spatial consistency (3D convolution)
            series = self.temporal_spatial(series)  # (1, C, length, H, W)

            # Restore original shape
            series = series.permute(0, 2, 1, 3, 4).squeeze(0)  # (length, C, H, W)

            # Apply weighted residual
            series = original_series + self.residual_weight * series

            # Remove padding if applied
            if length < kernel_size_b:
                series = series[:length]

            # Store processed series
            processed_series.append(series)
            start_idx += length

        # Concatenate all processed series back together
        bottleneck = torch.cat(processed_series, dim=0)  # Shape: (B, C, H, W)

        # Decoding
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        # Output layer
        out = torch.sigmoid(self.conv(dec1))

        return out
