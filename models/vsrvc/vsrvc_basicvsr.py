import torch
from torch import nn
from torchvision.ops import DeformConv2d

from models.basic_blocks import ResidualBlocksWithInputConv
from models.head_decoders import ReconstructionHead, PixelShufflePack
from models.hyperprior_compressor import HyperpriorCompressAI
from models.motion_blocks import MotionEstimator, MotionCompensator


class VSRVCBasicEncoder(nn.Module):
    def __init__(self, motion_compensator: MotionCompensator, in_channels: int = 3, mid_channels: int = 64,
                 num_blocks: int = 3, ):
        super(VSRVCBasicEncoder, self).__init__()
        self.feat_extractor = nn.Sequential(*[
            nn.Conv2d(in_channels, mid_channels, 5, 1, 2),
            ResidualBlocksWithInputConv(in_channels=mid_channels, out_channels=mid_channels, num_blocks=num_blocks)
        ])
        self.motion_estimator = MotionEstimator(mid_channels, 144)
        self.shared_motion_compensator = motion_compensator

    def extract_feats(self, x):
        return self.feat_extractor(x)

    def compress(self, prev_recon, x):
        B, N, C, H, W = x.size()
        assert (N == 2)
        prev_feat = self.extract_feats(x[:, 0])
        curr_feat = self.extract_feats(x[:, 1])
        prev_recon_feat = self.extract_feats(prev_recon)
        offsets_forward = self.motion_estimator(prev_feat, curr_feat)
        offsets_backward = self.motion_estimator(curr_feat, prev_feat)
        return [
            (prev_recon_feat, curr_feat, offsets_forward),
            (prev_feat, curr_feat, offsets_forward, offsets_backward, x[:, -1])
        ]

    def forward(self, x):
        B, N, C, H, W = x.size()
        assert (N == 2)
        prev_feat = self.extract_feats(x[:, 0])
        curr_feat = self.extract_feats(x[:, 1])
        offsets_forward = self.motion_estimator(prev_feat, curr_feat)
        offsets_backward = self.motion_estimator(curr_feat, prev_feat)
        return [
            (prev_feat, curr_feat, offsets_forward),
            (prev_feat, curr_feat, offsets_forward, offsets_backward, x[:, -1])
        ]


class VCBasicDecoder(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int):
        super(VCBasicDecoder, self).__init__()
        self.motion_compressor = HyperpriorCompressAI(144, mid_channels, 144)
        self.residual_compressor = HyperpriorCompressAI(in_channels, mid_channels, mid_channels)
        self.reconstruction_head = ReconstructionHead(in_channels=mid_channels, mid_channels=mid_channels)
        self.shared_motion_compensator = None

    def share(self, encoder):
        self.shared_motion_compensator = encoder.shared_motion_compensator

    def compress(self, x):
        prev_recon_feat, curr_feat, offsets = x
        mv_p_string, mv_hp_string, mv_shape = self.motion_compressor.compress(offsets)
        recon_offsets = self.motion_compressor.decompress(mv_p_string, mv_hp_string, mv_shape)
        align_feat = self.shared_motion_compensator(prev_recon_feat, recon_offsets)
        res = curr_feat - align_feat
        res_p_string, res_hp_string, res_shape = self.residual_compressor.compress(res)
        return [(res_p_string, res_hp_string, res_shape), (mv_p_string, mv_hp_string, mv_shape)]

    def decompress(self, x):
        prev_recon_feat, res_p_string, res_hp_string, res_shape, mv_p_string, mv_hp_string, mv_shape = x
        recon_offsets = self.motion_compressor.decompress(mv_p_string, mv_hp_string, mv_shape)
        align_feat = self.shared_motion_compensator(prev_recon_feat, recon_offsets)
        recon_res = self.residual_compressor.decompress(res_p_string, res_hp_string, res_shape)
        recon_feat = recon_res + align_feat
        recon_frame = self.reconstruction_head(recon_feat)
        return recon_frame

    def forward(self, x: torch.Tensor):
        prev_feat, curr_feat, offsets = x
        decompressed_offsets, mv_p_bits, mv_hp_bits = self.motion_compressor.train_compression_decompression(offsets)
        align_feat = self.shared_motion_compensator(prev_feat, decompressed_offsets)
        res = curr_feat - align_feat
        decompressed_data, res_p_bits, res_hp_bits = self.residual_compressor.train_compression_decompression(res)
        recon_feat = decompressed_data + align_feat
        recon_frame = self.reconstruction_head(recon_feat)
        return recon_frame, [res_p_bits, res_hp_bits, mv_p_bits, mv_hp_bits]


class VSRBasicDecoder(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, scale: int = 2):
        super(VSRBasicDecoder, self).__init__()
        self.reconstruction_trunk = ResidualBlocksWithInputConv(5 * in_channels, mid_channels, 5)
        self.shared_motion_compensator = None
        self.dcn_module = nn.ModuleDict()
        self.refiner = nn.ModuleDict()
        self.modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
        for i, module in enumerate(self.modules):
            self.dcn_module[module] = FirstOrderDCN(2 * mid_channels, mid_channels, 16)
            self.refiner[module] = ResidualBlocksWithInputConv((i + 2) * mid_channels, mid_channels, 5)

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

    def share(self, encoder):
        self.shared_motion_compensator = encoder.shared_motion_compensator

    def propagate(self, feats_dict, offsets_forward, offsets_backward, module_name):
        order = [0, 1]
        offsets = offsets_forward
        if "backward" in module_name:
            order = [1, 0]
            offsets = offsets_backward

        propagated_feats = torch.zeros_like(feats_dict["spatial"][0])
        for i, frame_idx in enumerate(order):
            current_features = feats_dict["spatial"][frame_idx]
            if i > 0:
                align_feat = self.shared_motion_compensator(propagated_feats, offsets)
                extra_feat = torch.cat([align_feat, current_features], dim=1)
                propagated_feats = self.dcn_module[module_name](propagated_feats, extra_feat, offsets)

            # concatenate and residual blocks
            stage_features = ([current_features]
                              + [feats_dict[key][frame_idx] for key in feats_dict if key not in ['spatial', module_name]]
                              + [propagated_feats])
            stage_features = torch.cat(stage_features, dim=1)
            propagated_feats = propagated_feats + self.refiner[module_name](stage_features)
            feats_dict[module_name].append(propagated_feats)

        if 'backward' in module_name:
            feats_dict[module_name] = feats_dict[module_name][::-1]

        return feats_dict

    def upsample(self, feats_dict, lqs):
        features = [feats_dict["spatial"][-1]] + [feats_dict[key][-1] for key in feats_dict if key != 'spatial']
        features = torch.cat(features, dim=1)
        reconstruction = self.reconstruction_trunk(features)
        reconstruction = self.upsampler(reconstruction)
        reconstruction = self.lrelu(self.conv_hr(reconstruction))
        reconstruction = self.conv_last(reconstruction)
        reconstruction += self.interpolation(lqs)
        return reconstruction

    def forward(self, x):
        prev_feat, curr_feat, offsets_forward, offsets_backward, lqs = x
        feats_dict = {"spatial": [prev_feat, curr_feat]}
        for module in self.modules:
            feats_dict[module] = []
            feats_dict = self.propagate(feats_dict, offsets_forward, offsets_backward, module)
        return self.upsample(feats_dict, lqs)


class FirstOrderDCN(nn.Module):
    def __init__(self, channels, out_channels, deform_groups):
        super(FirstOrderDCN, self).__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.deform_groups = deform_groups
        self.dcn_conv = DeformConv2d(self.channels, self.out_channels, 3, padding=1, groups=self.deform_groups)
        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * self.out_channels + 144, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 216, 3, 1, 1),
        )

    def forward(self, x, extra_feat, offsets):
        extra_feat = torch.cat([extra_feat, offsets], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        dcn_offset = torch.cat([o1, o2], dim=1)
        dcn_offset = 10 * torch.tanh(dcn_offset)
        dcn_offset += offsets.flip(1)  # .repeat(1, 2, 1, 1)

        # mask
        mask = torch.sigmoid(mask)

        return self.dcn_conv(x, dcn_offset, mask)
