import torch
from torch import nn

from models.basic_blocks import ResidualBlocksWithInputConv
from models.head_decoders import PixelShufflePack, ReconstructionHead
from models.hyperprior_compressor import HyperpriorCompressAI


class VSRVCEncoder(nn.Module):
    def __init__(self,  sliding_window: int = 1, mid_channels: int = 64, out_channels: int = 64, num_blocks: int = 3):
        super(VSRVCEncoder, self).__init__()
        self.layers = nn.Sequential(*[
            nn.Conv2d(3 * sliding_window, mid_channels, 5, 2, 2),
            ResidualBlocksWithInputConv(in_channels=mid_channels, out_channels=out_channels, num_blocks=num_blocks)
        ])

    def forward(self, x):
        B, N, C, H, W = x.size()
        features = self.layers(x.view(B, N*C, H, W))
        return [features, (features, x[:, -1])]


class ISRDecoder(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int):
        super(ISRDecoder, self).__init__()
        self.reconstruction_trunk = ResidualBlocksWithInputConv(in_channels, mid_channels, 3)
        self.upsampler1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsampler2 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(mid_channels, 3, 3, 1, 1)
        self.interpolation = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        features, lqs = x
        reconstruction = self.reconstruction_trunk(features)
        reconstruction = self.lrelu(self.upsampler1(reconstruction))
        reconstruction = self.lrelu(self.upsampler2(reconstruction))
        reconstruction = self.lrelu(self.conv_hr(reconstruction))
        reconstruction = self.conv_last(reconstruction)
        reconstruction += self.interpolation(lqs)
        return reconstruction


class ICDecoder(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int):
        super(ICDecoder, self).__init__()
        self.compressor = HyperpriorCompressAI(in_channels, mid_channels, mid_channels)
        self.reconstruction_head = ReconstructionHead(in_channels=mid_channels, mid_channels=mid_channels)

    def compress(self, x: torch.Tensor):
        prior_string, hyperprior_string, shape = self.compressor.compress(x)
        return prior_string, hyperprior_string, shape

    def decompress(self, prior_string, hyperprior_string, shape):
        reconstructed_features = self.compressor.decompress(prior_string, hyperprior_string, shape)
        reconstructed_frame = self.reconstruction_head(reconstructed_features)
        return reconstructed_frame

    def forward(self, x: torch.Tensor):
        decompressed_data, prior_bits, hyperprior_bits = self.compressor.train_compression_decompression(x)
        reconstructed_frame = self.reconstruction_head(decompressed_data)
        return reconstructed_frame, prior_bits, hyperprior_bits
