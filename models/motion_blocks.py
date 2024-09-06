import torch
from torch import nn
from torchvision.ops import DeformConv2d

from models.basic_blocks import ResidualBlockNoBN


class MotionEstimator(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(MotionEstimator, self).__init__()
        self.offset_conv1 = nn.Conv2d(2 * in_channels, out_channels, 3, 1, 1, bias=True)
        self.offset_conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, previous_features: torch.Tensor, current_features: torch.Tensor):
        offsets = torch.cat([previous_features, current_features], dim=1)
        offsets = self.lrelu(self.offset_conv1(offsets))
        offsets = self.lrelu(self.offset_conv2(offsets))
        return offsets


class MotionCompensator(nn.Module):
    def __init__(self, channels: int, dcn_groups: int = 8):
        super(MotionCompensator, self).__init__()
        self.channels = channels
        self.dcn_groups = dcn_groups
        self.deformable_convolution = DeformConv2d(channels, channels, 3, stride=1, padding=1, dilation=1,
                                                   groups=dcn_groups)
        self.refine_conv1 = nn.Conv2d(channels * 2, channels, 3, 1, 1)
        self.refine_conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, previous_features: torch.Tensor, offsets: torch.Tensor):
        aligned_features = self.deformable_convolution(previous_features, offsets)

        refined_features = torch.cat([aligned_features, previous_features], dim=1)
        refined_features = self.lrelu(self.refine_conv1(refined_features))
        refined_features = self.lrelu(self.refine_conv2(refined_features))
        return aligned_features + refined_features

    def to_json(self):
        return {
            "class": self.__name__,
            "kwargs": {
                "channels": self.channels,
                "dcn_groups": self.dcn_groups,
            }
        }
