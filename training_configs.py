from typing import Dict

from torch import nn

from datasets import Vimeo90k, Reds
from models.vsrvc import ICDecoder, ISRDecoder, VSRVCEncoder, VCResidualDecoder, VSRResidualDecoder, \
    VSRVCResidualEncoder, VCMotionResidualDecoder, VSRVCMotionResidualEncoder, DummyVCDecoder, DummyVSRDecoder


def get_dataset_info(params):
    if params.vimeo_path is not None:
        return Vimeo90k, params.vimeo_path
    return Reds, params.reds_path


def vsrvc(params, kwargs):
    dataset_class, dataset_path = get_dataset_info(params)
    train_set = dataset_class(dataset_path, sliding_window_size=params.sliding_window)
    test_set = dataset_class(dataset_path, test_mode=True, sliding_window_size=params.sliding_window)
    decoder_kwargs: Dict[str, dict] = {}
    decoders: nn.ModuleDict[str, nn.Module] = nn.ModuleDict({})
    if params.vc:
        decoder_kwargs["vc"] = {
            'in_channels': 64,
            'mid_channels': 64,
        }
        decoders["vc"] = ICDecoder(**decoder_kwargs['vc'])
    else:
        decoder_kwargs["vc"] = {}
        decoders["vc"] = DummyVCDecoder()
    if params.vsr:
        decoder_kwargs["vsr"] = {
            'in_channels': 64,
            'mid_channels': 64,
            "scale": params.scale,
        }
        decoders["vsr"] = ISRDecoder(**decoder_kwargs['vsr'])
    else:
        decoder_kwargs["vsr"] = {}
        decoders["vsr"] = DummyVSRDecoder()
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
    dataset_class, dataset_path = get_dataset_info(params)
    train_set = dataset_class(dataset_path, sliding_window_size=2)
    test_set = dataset_class(dataset_path, test_mode=True, sliding_window_size=2)
    decoder_kwargs: Dict[str, dict] = {}
    decoders: nn.ModuleDict[str, nn.Module] = nn.ModuleDict({})
    if params.vc:
        decoder_kwargs["vc"] = {
            'in_channels': 64,
            'mid_channels': 64,
        }
        decoders["vc"] = VCResidualDecoder(**decoder_kwargs['vc'])
    else:
        decoder_kwargs["vc"] = {}
        decoders["vc"] = DummyVCDecoder()
    if params.vsr:
        decoder_kwargs["vsr"] = {
            'in_channels': 64,
            'mid_channels': 64,
            "scale": params.scale,
        }
        decoders["vsr"] = VSRResidualDecoder(**decoder_kwargs['vsr'])
    else:
        decoder_kwargs["vsr"] = {}
        decoders["vsr"] = DummyVSRDecoder()
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
    dataset_class, dataset_path = get_dataset_info(params)
    if params.multi_input:
        train_set = {
            "vc": dataset_class(dataset_path, sliding_window_size=2, multi_input=params.multi_input),
            "vsr": dataset_class(dataset_path, sliding_window_size=2, multi_input=params.multi_input)
        }
        test_set = {
            "vc": dataset_class(dataset_path, test_mode=True, sliding_window_size=2, multi_input=params.multi_input),
            "vsr": dataset_class(dataset_path, test_mode=True, sliding_window_size=2, multi_input=params.multi_input)
        }
    else:
        train_set = dataset_class(dataset_path, sliding_window_size=2)
        test_set = dataset_class(dataset_path, test_mode=True, sliding_window_size=2)
    decoder_kwargs: Dict[str, dict] = {}
    decoders: nn.ModuleDict[str, nn.Module] = nn.ModuleDict({})
    if params.vc:
        decoder_kwargs["vc"] = {
            'in_channels': 64,
            'mid_channels': 64,
        }
        decoders["vc"] = VCMotionResidualDecoder(**decoder_kwargs['vc'])
    else:
        decoder_kwargs["vc"] = {}
        decoders["vc"] = DummyVCDecoder()
    if params.vsr:
        decoder_kwargs["vsr"] = {
            'in_channels': 64,
            'mid_channels': 64,
            "scale": params.scale,
        }
        decoders["vsr"] = ISRDecoder(**decoder_kwargs['vsr'])
    else:
        decoder_kwargs["vsr"] = {}
        decoders["vsr"] = DummyVSRDecoder()
    encoder_class = VSRVCMotionResidualEncoder
    kwargs["arch_args"]["encoder_kwargs"] = {
        'in_channels': 3,
        'mid_channels': 64,
        'out_channels': 64,
        'num_blocks': 3,
    }
    model_type = "PFrameWithMotion"
    return train_set, test_set, encoder_class, decoders, kwargs, decoder_kwargs, model_type
