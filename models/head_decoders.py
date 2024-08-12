import torch
from torch import nn

import torch.nn.functional as f

from models.basic_blocks import ResidualBlocksWithInputConv


class PixelShufflePack(nn.Module):
    """ Pixel Shuffle upsample layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio.
        upsample_kernel (int): Kernel size of Conv layer to expand channels.

    Returns:
        Upsampled feature map.
    """

    def __init__(self, in_channels, out_channels, scale_factor,
                 upsample_kernel):
        super(PixelShufflePack, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)

    def forward(self, x):
        """Forward function for PixelShufflePack.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        x = self.upsample_conv(x)
        x = f.pixel_shuffle(x, self.scale_factor)
        return x


class ReconstructionHead(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int):
        super(ReconstructionHead, self).__init__()
        self.reconstruction_trunk = ResidualBlocksWithInputConv(in_channels, mid_channels, num_blocks=3)
        self.deconv = nn.ConvTranspose2d(mid_channels, 3, 5, stride=2, padding=2, output_padding=1)

    def forward(self, features: torch.Tensor):
        reconstruction = self.reconstruction_trunk(features)
        reconstruction = self.deconv(reconstruction)
        return reconstruction
