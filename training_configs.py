from torch import nn

from datasets import Vimeo90k
from models.vsrvc import ICDecoder, ISRDecoder, VSRVCEncoder, VCResidualDecoder, VSRResidualDecoder, \
    VSRVCResidualEncoder, VCMotionResidualDecoder, VSRVCMotionResidualEncoder


def vsrvc(params, kwargs):
    train_set = Vimeo90k("../Datasets/VIMEO90k", params.scale, sliding_window_size=params.sliding_window)
    test_set = Vimeo90k("../Datasets/VIMEO90k", params.scale, test_mode=True, sliding_window_size=params.sliding_window)
    decoder_kwargs = {}
    decoders = {}
    if params.vc:
        decoder_kwargs["vc"] = {
            'in_channels': 64,
            'mid_channels': 64,
        }
        decoders["vc"] = ICDecoder(**decoder_kwargs['vc'])
    if params.vsr:
        decoder_kwargs["vsr"] = {
            'in_channels': 64,
            'mid_channels': 64,
        }
        decoders["vsr"] = ISRDecoder(**decoder_kwargs['vsr'])
    decoders = nn.ModuleDict(decoders)
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
    decoder_kwargs = {}
    decoders = {}
    if params.vc:
        decoder_kwargs["vc"] = {
            'in_channels': 64,
            'mid_channels': 64,
        }
        decoders["vc"] = VCResidualDecoder(**decoder_kwargs['vc'])
    if params.vsr:
        decoder_kwargs["vsr"] = {
            'in_channels': 64,
            'mid_channels': 64,
        }
        decoders["vsr"] = VSRResidualDecoder(**decoder_kwargs['vsr'])
    decoders = nn.ModuleDict(decoders)
    encoder_class = VSRVCResidualEncoder
    kwargs["arch_args"]["encoder_kwargs"] = {
        'in_channels': 3,
        'mid_channels': 64,
        'out_channels': 64,
        'num_blocks': 3,
    }
    model_type = "PFrame"
    return train_set, test_set, encoder_class, decoders, kwargs, decoder_kwargs, model_type


def vsrvc_motion_residual(params, kwargs):
    train_set = Vimeo90k("../Datasets/VIMEO90k", params.scale, sliding_window_size=2)
    test_set = Vimeo90k("../Datasets/VIMEO90k", params.scale, test_mode=True, sliding_window_size=2)
    decoder_kwargs = {}
    decoders = {}
    if params.vc:
        decoder_kwargs["vc"] = {
            'in_channels': 64,
            'mid_channels': 64,
        }
        decoders["vc"] = VCMotionResidualDecoder(**decoder_kwargs['vc'])
    if params.vsr:
        decoder_kwargs["vsr"] = {
            'in_channels': 64,
            'mid_channels': 64,
        }
        decoders["vsr"] = ISRDecoder(**decoder_kwargs['vsr'])
    decoders = nn.ModuleDict(decoders)
    encoder_class = VSRVCMotionResidualEncoder
    kwargs["arch_args"]["encoder_kwargs"] = {
        'in_channels': 3,
        'mid_channels': 64,
        'out_channels': 64,
        'num_blocks': 3,
    }
    model_type = "PFrameWithMotion"
    return train_set, test_set, encoder_class, decoders, kwargs, decoder_kwargs, model_type
