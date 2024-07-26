from torch import nn

from datasets import Vimeo90k
from models.vsrvc import ICDecoder, ISRDecoder, ISRICEncoder


def icisr(params, kwargs):
    train_set = Vimeo90k("../Datasets/VIMEO90k", params.scale, sliding_window_size=1)
    test_set = Vimeo90k("../Datasets/VIMEO90k", params.scale, test_mode=True, sliding_window_size=1)
    encoder_class = ISRICEncoder
    kwargs["arch_args"]["encoder_kwargs"] = {
        'in_channels': 3,
        'mid_channels': 64,
        'out_channels': 64,
        'num_blocks': 3,
    }
    decoder_kwargs = {
        'vc': {
            'in_channels': 64,
            'mid_channels': 64,
        },
        'vsr': {
            'in_channels': 64,
            'mid_channels': 64,
        }
    }
    decoders = nn.ModuleDict({
        'vc': ICDecoder(**decoder_kwargs['vc']),
        'vsr': ISRDecoder(**decoder_kwargs['vsr'])
    })

    return train_set, test_set, encoder_class, decoders, kwargs, decoder_kwargs
