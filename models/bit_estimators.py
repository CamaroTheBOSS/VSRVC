import math

import torch
import torch.nn.functional as f
from torch import nn

from models.basic_blocks import ResidualBlockNoBN, make_layer


# J. Ballé et al.: "Variational image compression with scale hyperprior" (2017)
class BitParam(nn.Module):
    def __init__(self, channel, final=False):
        super(BitParam, self).__init__()
        self.final = final
        self.h = nn.Parameter(nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        self.b = nn.Parameter(nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        self.a = None
        if not final:
            self.a = nn.Parameter(nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))

    def forward(self, x):
        x = x * f.softplus(self.h) + self.b
        if self.final:
            return f.sigmoid(x)

        return x + f.tanh(x) * f.tanh(self.a)


class DirectEntropyCoder(nn.Module):
    """
    Estimate bit by directly bit prediction
    Variational image compression with a scale hyperprior, 2017
    """
    def __init__(self, channel):
        super(DirectEntropyCoder, self).__init__()
        self.f1 = BitParam(channel)
        self.f2 = BitParam(channel)
        self.f3 = BitParam(channel)
        self.f4 = BitParam(channel, True)

    def cdf(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        return x

    def forward(self, x):
        prob = self.cdf(x + 0.5) - self.cdf(x - 0.5)
        if self.training:
            return torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))
        return torch.sum(torch.clamp(-1.0 * torch.log(prob) / math.log(2.0), 0, 50))


def qint_to_int(x: torch.Tensor, scale):
    if x.dtype == torch.qint8:
        return x.int_repr()
    return x / scale


def normalized_to_int(x: torch.Tensor, scales, zero_points):
    return (x - zero_points) / scales - 0.5


class HyperpriorEntropyCoder(nn.Module):
    """
    Estimate bits with hyperprior entropy model
    """
    def __init__(self, channels: int):
        super(HyperpriorEntropyCoder, self).__init__()
        self.distribution = DirectEntropyCoder(channels)
        self.mu_sigma_adjuster = nn.Sequential(*[
            nn.Conv2d(channels * 2, channels * 4, 1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(channels * 4, channels * 4, 1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.1),
            make_layer(ResidualBlockNoBN, 2, **dict(channels=channels * 4, ks=1)),
            nn.Conv2d(channels * 4, channels * 2, 1, stride=1, padding=0),
        ])
        self.channels = channels

    def get_mu_sigma(self, mu_sigmas):
        mu_sigmas = self.mu_sigma_adjuster(mu_sigmas)
        return mu_sigmas[:, :self.channels], mu_sigmas[:, self.channels:].pow(2).clamp(1e-5, 1e10)

    def estimate_prior_bits(self, x, mu_sigmas):
        mu, sigmas = self.get_mu_sigma(mu_sigmas)
        gaussian = torch.distributions.normal.Normal(mu, sigmas)
        probs = gaussian.cdf(x + 0.5) - gaussian.cdf(x - 0.5)
        if self.training:
            return torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50)), mu, sigmas
        return torch.sum(torch.clamp(-1.0 * torch.log(probs) / math.log(2.0), 0, 50)), mu, sigmas

    def sample_hyperprior(self, n_values, low=0, high=1, precision=1000, device=None):
        """
        Sampling function, which draws random values from learned cdf.
        :param n_values: How many values to sample
        :param low: Minimum value of sample (important because of cdf characteristics)
        :param high: Maximum value of sample (important because of cdf characteristics)
        :param precise: How precise cdf approximation should be
        :param device: Where to store tensor
        :return: Sampled values
        """
        assert n_values <= precision
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        x = torch.linspace(low, high, precision, device=device)
        cdf = self.distribution.cdf(x)[0]
        probs = torch.rand(x.size()).to(x.device)
        dist_indices = torch.argmin(torch.abs(probs.unsqueeze(-1) - cdf), dim=-1)
        values = torch.gather(x.repeat(len(cdf), 1), 1, dist_indices)
        return values[:, :n_values]

    def forward(self, data_prior: torch.Tensor, data_hyperprior: torch.Tensor, mu_sigmas: torch.Tensor):
        hyperprior_bits = self.distribution(data_hyperprior)
        prior_bits, mu, sigmas = self.estimate_prior_bits(data_prior, mu_sigmas)
        return prior_bits, hyperprior_bits, mu, sigmas



