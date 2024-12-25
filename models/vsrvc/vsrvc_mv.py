import torch
from torch import nn

from models.basic_blocks import ResidualBlocksWithInputConv
from models.head_decoders import ReconstructionHead, PixelShufflePack
from models.hyperprior_compressor import HyperpriorCompressAI
from models.motion_blocks import MotionEstimator, MotionCompensator


class VSRVCMotionResidualEncoder(nn.Module):
    def __init__(self,  in_channels: int = 3, mid_channels: int = 64, out_channels: int = 64, num_blocks: int = 3):
        super(VSRVCMotionResidualEncoder, self).__init__()
        self.feat_extractor = nn.Sequential(*[
            nn.Conv2d(in_channels, mid_channels, 5, 1, 2),
            ResidualBlocksWithInputConv(in_channels=mid_channels, out_channels=mid_channels, num_blocks=num_blocks)
        ])
        self.motion_estimator = MotionEstimator(mid_channels, 144)
        self.motion_compressor = HyperpriorCompressAI(144, mid_channels, 144)
        self.motion_compensator = MotionCompensator(out_channels)

    def extract_feats(self, x):
        return self.feat_extractor(x)

    def compress(self, prev_recon, x):
        B, N, C, H, W = x.size()
        assert (N == 2)
        prev_feat = self.extract_feats(x[:, 0])
        curr_feat = self.extract_feats(x[:, 1])
        prev_recon_feat = self.extract_feats(prev_recon)

        cp_data = []
        for prev in [prev_feat, prev_recon_feat]:
            offsets = self.motion_estimator(prev, curr_feat)
            mv_p_string, mv_hp_string, shape = self.motion_compressor.compress(offsets)
            recon_offsets = self.motion_compressor.decompress(mv_p_string, mv_hp_string, shape)
            align_feat = self.motion_compensator(prev, recon_offsets)
            cp_data.append((align_feat, mv_p_string, mv_hp_string, shape))
        out_vsr = (cp_data[0][0], x[:, -1])
        out_vc = (cp_data[1][0], curr_feat,) + cp_data[1][1:]
        return [out_vc, out_vsr]

    def decompress(self, x):
        mv_p_string, mv_hp_string, shape = x
        recon_offsets = self.motion_compressor.decompress(mv_p_string, mv_hp_string, shape)
        return recon_offsets

    def align_features(self, prev_feat, offsets):
        aligned_features = self.motion_compensator(prev_feat, offsets)
        return aligned_features

    def forward(self, x):
        B, N, C, H, W = x.size()
        assert (N == 2)
        prev_feat = self.extract_feats(x[:, 0])
        curr_feat = self.extract_feats(x[:, 1])
        offsets = self.motion_estimator(prev_feat, curr_feat)
        decompressed_offsets, p_bits, hp_bits = self.motion_compressor.train_compression_decompression(offsets)
        aligned_features = self.motion_compensator(prev_feat, decompressed_offsets)
        return [(aligned_features, curr_feat, p_bits, hp_bits), (aligned_features, x[:, -1])]


class VCMotionResidualDecoder(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int):
        super(VCMotionResidualDecoder, self).__init__()
        self.compressor = HyperpriorCompressAI(in_channels, mid_channels, mid_channels)
        self.reconstruction_head = ReconstructionHead(in_channels=mid_channels, mid_channels=mid_channels)

    def compress(self, x):
        align_feat, curr_feat, mv_p_string, mv_hp_string, mv_shape = x
        res = curr_feat - align_feat
        res_p_string, res_hp_string, res_shape = self.compressor.compress(res)
        return [(res_p_string, res_hp_string, res_shape), (mv_p_string, mv_hp_string, mv_shape)]

    def decompress(self, x):
        align_feat, prior_string, hyperprior_string, shape = x
        recon_res = self.compressor.decompress(prior_string, hyperprior_string, shape)
        recon_feat = recon_res + align_feat
        recon_frame = self.reconstruction_head(recon_feat)
        return recon_frame

    def forward(self, x: torch.Tensor):
        align_feat, curr_feat, mv_p_bits, mv_hp_bits = x
        res = curr_feat - align_feat
        decompressed_data, res_p_bits, res_hp_bits = self.compressor.train_compression_decompression(res)
        recon_feat = decompressed_data + align_feat
        recon_frame = self.reconstruction_head(recon_feat)
        return recon_frame, [res_p_bits, res_hp_bits, mv_p_bits, mv_hp_bits]


class VSRMotionResidualDecoder(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, scale: int = 2):
        super(VSRMotionResidualDecoder, self).__init__()
        self.reconstruction_trunk = ResidualBlocksWithInputConv(in_channels, mid_channels, 3)
        num_layers = int(torch.log2(torch.tensor(scale)))
        upscale_layers = []
        for _ in range(num_layers):
            upscale_layers.append(PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3))
            upscale_layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.upsampler = nn.Sequential(*upscale_layers)
        self.conv_hr = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(mid_channels, 3, 3, 1, 1)
        self.interpolation = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        features, lqs = x
        reconstruction = self.reconstruction_trunk(features)
        reconstruction = self.upsampler(reconstruction)
        reconstruction = self.lrelu(self.conv_hr(reconstruction))
        reconstruction = self.conv_last(reconstruction)
        reconstruction += self.interpolation(lqs)
        return reconstruction

