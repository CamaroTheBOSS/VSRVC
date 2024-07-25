import math
from dataclasses import dataclass

import torch
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from torch import nn

from models.basic_blocks import ResidualBlockNoBN, make_layer
from models.bit_estimators import HyperpriorEntropyCoder


@dataclass
class Quantized:
    qint: torch.Tensor
    scales: torch.Tensor
    zero_points: torch.Tensor


def qint_to_float(x: torch.Tensor):
    if x.dtype == torch.qint8:
        scale = x.q_scale()
        zero_point = x.q_zero_point()
        return (x.int_repr().float() - zero_point) * scale
    return x


def int_to_float(x: torch.Tensor, scale: float, zero_point: float):
    return (x.float() - zero_point) * scale


def norm_qint_to_float(x: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor):
    return (x.float() + 0.5) * scales + zero_points


class HyperpriorCompressor(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
        super(HyperpriorCompressor, self).__init__()
        self.data_encoder = nn.Sequential(*[
            nn.Conv2d(in_channels, mid_channels, 5, stride=2, padding=2),
            make_layer(ResidualBlockNoBN, 2, **dict(channels=mid_channels)),
            nn.Conv2d(mid_channels, mid_channels, 5, stride=2, padding=2),
            make_layer(ResidualBlockNoBN, 2, **dict(channels=mid_channels)),
            nn.Conv2d(mid_channels, mid_channels, 5, stride=2, padding=2),
        ])
        self.data_decoder = nn.Sequential(*[
            nn.ConvTranspose2d(3 * mid_channels, mid_channels, 5, stride=2, padding=2, output_padding=1),
            make_layer(ResidualBlockNoBN, 2, **dict(channels=mid_channels)),
            nn.ConvTranspose2d(mid_channels, mid_channels, 5, stride=2, padding=2, output_padding=1),
            make_layer(ResidualBlockNoBN, 2, **dict(channels=mid_channels)),
            nn.ConvTranspose2d(mid_channels, out_channels, 5, stride=2, padding=2, output_padding=1),
        ])

        self.hyperprior_encoder = nn.Sequential(*[
            nn.Conv2d(mid_channels, mid_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(mid_channels, mid_channels, 5, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(mid_channels, mid_channels, 5, stride=2, padding=2),
        ])
        self.hyperprior_decoder = nn.Sequential(*[
            nn.ConvTranspose2d(mid_channels, mid_channels, 5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(mid_channels, mid_channels * 3 // 2, 5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(mid_channels * 3 // 2, mid_channels * 2, 3, stride=1, padding=1),
        ])
        self.bit_estimator = HyperpriorEntropyCoder(mid_channels)
        self.mid_channels = mid_channels

    def quantize(self, x: torch.Tensor):
        if self.training:
            return x + torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5)
        return torch.round(x)

    def train_compression_decompression(self, data: torch.Tensor):
        data_prior = self.data_encoder(data)
        q_data_prior = self.quantize(data_prior)

        data_hyperprior = self.hyperprior_encoder(data_prior)
        q_data_hyperprior = self.quantize(data_hyperprior)

        mu_sigmas = self.hyperprior_decoder(q_data_hyperprior)
        prior_bits, hyperprior_bits = self.bit_estimator(q_data_prior, q_data_hyperprior, mu_sigmas)

        reconstructed_data = self.data_decoder(torch.cat([q_data_prior, mu_sigmas], dim=1))
        return reconstructed_data, [prior_bits, hyperprior_bits]

    def compress(self, data: torch.Tensor):
        data_prior = self.data_encoder(data)
        quantized_data_prior, scale_prior, zero_point_prior = self.quantize(data_prior)
        q_prior = Quantized(quantized_data_prior, scale_prior, zero_point_prior)

        data_hyperprior = self.hyperprior_encoder(data_prior)
        quantized_data_hyperprior, scale_hyperprior, zero_point_hyperprior = self.quantize(data_hyperprior)
        q_hyperprior = Quantized(quantized_data_hyperprior, scale_hyperprior, zero_point_hyperprior)

        return q_prior, q_hyperprior

    def decompress(self, prior: Quantized, hyperprior: Quantized, return_mu_sigmas=False):
        quantized_data_prior = self.get_float(prior.qint, prior.scales, prior.zero_points)
        quantized_data_hyperprior = self.get_float(hyperprior.qint, hyperprior.scales, hyperprior.zero_points)
        mu_sigmas = self.hyperprior_decoder(quantized_data_hyperprior)
        reconstructed_data = self.data_decoder(torch.cat([quantized_data_prior, mu_sigmas], dim=1))
        if return_mu_sigmas:
            return reconstructed_data, mu_sigmas
        return reconstructed_data


class HyperpriorCompressAI(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
        super(HyperpriorCompressAI, self).__init__()
        self.data_encoder = nn.Sequential(*[
            nn.Conv2d(in_channels, mid_channels, 5, stride=2, padding=2),
            make_layer(ResidualBlockNoBN, 2, **dict(channels=mid_channels)),
            nn.Conv2d(mid_channels, mid_channels, 5, stride=2, padding=2),
            make_layer(ResidualBlockNoBN, 2, **dict(channels=mid_channels)),
            nn.Conv2d(mid_channels, mid_channels, 5, stride=2, padding=2),
        ])
        self.data_decoder = nn.Sequential(*[
            nn.ConvTranspose2d(2 * mid_channels, mid_channels, 5, stride=2, padding=2, output_padding=1),
            make_layer(ResidualBlockNoBN, 2, **dict(channels=mid_channels)),
            nn.ConvTranspose2d(mid_channels, mid_channels, 5, stride=2, padding=2, output_padding=1),
            make_layer(ResidualBlockNoBN, 2, **dict(channels=mid_channels)),
            nn.ConvTranspose2d(mid_channels, out_channels, 5, stride=2, padding=2, output_padding=1),
        ])

        self.hyperprior_encoder = nn.Sequential(*[
            nn.Conv2d(mid_channels, mid_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(mid_channels, mid_channels, 5, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(mid_channels, mid_channels, 5, stride=2, padding=2),
        ])
        self.hyperprior_decoder = nn.Sequential(*[
            nn.ConvTranspose2d(mid_channels, mid_channels, 5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(mid_channels, mid_channels * 3 // 2, 5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(mid_channels * 3 // 2, mid_channels, 3, stride=1, padding=1),
        ])
        self.entropy_bottleneck = EntropyBottleneck(64)
        self.gaussian_conditional = GaussianConditional(None)
        self.mid_channels = mid_channels

    def get_bits(self, likelihoods):
        return torch.log(likelihoods + 1e-5).sum() / (-math.log(2))

    def train_compression_decompression(self, data: torch.Tensor):
        data_prior = self.data_encoder(data)
        data_hyperprior = self.hyperprior_encoder(torch.abs(data_prior))

        quantized_data_hyperprior, data_hyperprior_likelihoods = self.entropy_bottleneck(data_hyperprior)
        sigmas = self.hyperprior_decoder(quantized_data_hyperprior)
        quantized_data_prior, data_prior_likelihoods = self.gaussian_conditional(data_prior, sigmas)
        reconstructed_data = self.data_decoder(torch.cat([quantized_data_prior, sigmas], dim=1))

        prior_bits = self.get_bits(data_prior_likelihoods)
        hyperprior_bits = self.get_bits(data_hyperprior_likelihoods)

        return reconstructed_data, prior_bits, hyperprior_bits

    def compress(self, data: torch.Tensor):
        data_prior = self.data_encoder(data)
        data_hyperprior = self.hyperprior_encoder(torch.abs(data_prior))

        hyperprior_string = self.entropy_bottleneck.compress(data_hyperprior)
        decompressed_data_hyperprior = self.entropy_bottleneck.decompress(hyperprior_string, data_hyperprior.size()[-2:])
        sigmas = self.hyperprior_decoder(decompressed_data_hyperprior)

        indexes = self.gaussian_conditional.build_indexes(sigmas)
        prior_string = self.gaussian_conditional.compress(data_prior, indexes)

        return prior_string, hyperprior_string, data_hyperprior.size()[-2:]

    def decompress(self, prior_string, hyperprior_string, shape):
        decompressed_data_hyperprior = self.entropy_bottleneck.decompress(hyperprior_string, shape)
        sigmas = self.hyperprior_decoder(decompressed_data_hyperprior)

        indexes = self.gaussian_conditional.build_indexes(sigmas)
        decompressed_data_prior = self.gaussian_conditional.decompress(prior_string, indexes,
                                                                       decompressed_data_hyperprior.dtype)
        reconstructed_data = self.data_decoder(torch.cat([decompressed_data_prior, sigmas], dim=1))
        return reconstructed_data
