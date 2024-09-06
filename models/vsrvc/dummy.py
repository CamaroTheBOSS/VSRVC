import torch
from torch import nn


class DummyVCDecoder(nn.Module):
    def __init__(self):
        super(DummyVCDecoder, self).__init__()
        self.out_shape = (1, 3, 256, 448)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def compress(self, x: torch.Tensor):
        if len(x) == 2:
            return [([b''], [b''], 0), ([b''], [b''], 0)]
        align_feat, curr_feat, mv_p_string, mv_hp_string, mv_shape = x
        return [([b''], [b''], 0), (mv_p_string, mv_hp_string, mv_shape)]

    def decompress(self, x):
        return self.forward(x)

    def forward(self, x):
        return torch.zeros(self.out_shape, device=self.device)


class DummyVSRDecoder(nn.Module):
    def __init__(self):
        super(DummyVSRDecoder, self).__init__()
        self.out_shape = (1, 3, 1024, 1792)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def forward(self, x):
        return torch.zeros(self.out_shape, device=self.device)
