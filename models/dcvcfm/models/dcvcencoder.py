from torch import nn


class DCVCEncoder(nn.Module):
    def __init__(self):
        super(DCVCEncoder, self).__init__()

    def extract_feats(self, x):
        return x

    def compress(self, prev_recon, x):
        return x

    def forward(self, x):
        return [x, 0]