from torch import nn

from datasets import Vimeo90k
from models.vsrvc import ICDecoder, ISRDecoder, VSRVCEncoder, VCResidualDecoder, VSRResidualDecoder, \
    VSRVCResidualEncoder


def vsrvc(params, kwargs):
    train_set = Vimeo90k("../Datasets/VIMEO90k", params.scale, sliding_window_size=params.sliding_window)
    test_set = Vimeo90k("../Datasets/VIMEO90k", params.scale, test_mode=True, sliding_window_size=params.sliding_window)
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
    encoder_class = VSRVCEncoder
    kwargs["arch_args"]["encoder_kwargs"] = {
        'sliding_window': params.sliding_window,
        'mid_channels': 64,
        'out_channels': 64,
        'num_blocks': 3,
    }
    model_type = "IFrame"
    return train_set, test_set, encoder_class, decoders, kwargs, decoder_kwargs, model_type


def vsrvc_residual(params, kwargs):
    train_set = Vimeo90k("../Datasets/VIMEO90k", params.scale, sliding_window_size=2)
    test_set = Vimeo90k("../Datasets/VIMEO90k", params.scale, test_mode=True, sliding_window_size=2)
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
        'vc': VCResidualDecoder(**decoder_kwargs['vc']),
        'vsr': VSRResidualDecoder(**decoder_kwargs['vsr'])
    })
    encoder_class = VSRVCResidualEncoder
    kwargs["arch_args"]["encoder_kwargs"] = {
        'in_channels': 3,
        'mid_channels': 64,
        'out_channels': 64,
        'num_blocks': 3,
    }
    model_type = "PFrame"
    return train_set, test_set, encoder_class, decoders, kwargs, decoder_kwargs, model_type
