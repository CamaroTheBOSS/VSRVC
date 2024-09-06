import torch
from torch import nn

from models.basic_blocks import ResidualBlocksWithInputConv
from models.head_decoders import ReconstructionHead
from models.hyperprior_compressor import HyperpriorCompressAI
from models.motion_blocks import MotionEstimator, MotionCompensator


class VSRVCShallowEncoder(nn.Module):
    def __init__(self,  in_channels: int = 3, mid_channels: int = 64, out_channels: int = 64, num_blocks: int = 3):
        super(VSRVCShallowEncoder, self).__init__()
        self.feat_extractor = nn.Sequential(*[
            nn.Conv2d(in_channels, mid_channels, 5, 1, 2),
            ResidualBlocksWithInputConv(in_channels=mid_channels, out_channels=out_channels, num_blocks=num_blocks)
        ])

    def compress(self, prev_recon, x):
        B, N, C, H, W = x.size()
        assert (N == 2)
        curr_feat = self.extract_feats(x[:, 1])
        prev_recon_feat = self.extract_feats(prev_recon)
        return [(prev_recon_feat, curr_feat), (curr_feat, x[:, -1])]

    def extract_feats(self, x):
        return self.feat_extractor(x)

    def forward(self, x):
        B, N, C, H, W = x.size()
        assert (N == 2)
        prev_feat = self.extract_feats(x[:, 0])
        curr_feat = self.extract_feats(x[:, 1])
        return [(prev_feat, curr_feat), (curr_feat, x[:, -1])]


class VCShallowDecoder(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int):
        super(VCShallowDecoder, self).__init__()
        self.motion_estimator = MotionEstimator(mid_channels, 144)
        self.motion_compressor = HyperpriorCompressAI(144, mid_channels, 144)
        self.motion_compensator = MotionCompensator(mid_channels)
        self.residual_compressor = HyperpriorCompressAI(in_channels, mid_channels, mid_channels)
        self.reconstruction_head = ReconstructionHead(in_channels=mid_channels, mid_channels=mid_channels)

    def compress(self, x):
        prev_recon_feat, curr_feat = x
        offsets = self.motion_estimator(prev_recon_feat, curr_feat)
        mv_p_string, mv_hp_string, mv_shape = self.motion_compressor.compress(offsets)
        recon_offsets = self.motion_compressor.decompress(mv_p_string, mv_hp_string, mv_shape)
        align_feat = self.motion_compensator(prev_recon_feat, recon_offsets)
        res = curr_feat - align_feat
        res_p_string, res_hp_string, res_shape = self.residual_compressor.compress(res)
        return [(res_p_string, res_hp_string, res_shape), (mv_p_string, mv_hp_string, mv_shape)]

    def decompress(self, x):
        prev_recon_feat, res_p_string, res_hp_string, res_shape, mv_p_string, mv_hp_string, mv_shape = x
        recon_offsets = self.motion_compressor.decompress(mv_p_string, mv_hp_string, mv_shape)
        align_feat = self.motion_compensator(prev_recon_feat, recon_offsets)
        recon_res = self.residual_compressor.decompress(res_p_string, res_hp_string, res_shape)
        recon_feat = recon_res + align_feat
        recon_frame = self.reconstruction_head(recon_feat)
        return recon_frame

    def forward(self, x: torch.Tensor):
        prev_feat, curr_feat = x
        offsets = self.motion_estimator(prev_feat, curr_feat)
        decompressed_offsets, mv_p_bits, mv_hp_bits = self.motion_compressor.train_compression_decompression(offsets)
        align_feat = self.motion_compensator(prev_feat, decompressed_offsets)
        res = curr_feat - align_feat
        decompressed_data, res_p_bits, res_hp_bits = self.residual_compressor.train_compression_decompression(res)
        recon_feat = decompressed_data + align_feat
        recon_frame = self.reconstruction_head(recon_feat)
        return recon_frame, [res_p_bits, res_hp_bits, mv_p_bits, mv_hp_bits]
