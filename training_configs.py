from typing import Dict

from torch import nn

from datasets import Vimeo90k, Reds
from models.motion_blocks import MotionCompensator
from models.vsrvc.vsrvc_basicvsr import VCBasicDecoder, VSRBasicDecoder, VSRVCBasicEncoder
from models.vsrvc.vsrvc_mv import VSRVCMotionResidualEncoder, VCMotionResidualDecoder, VSRMotionResidualDecoder
from models.vsrvc.vsrvc_shallow import VSRVCShallowEncoder, VCShallowDecoder
from models.vsrvc.isric import ISRICEncoder, ISRDecoder, ICDecoder
from models.vsrvc.dummy import DummyVSRDecoder, DummyVCDecoder


def get_dataset_info(params):
    if params.vimeo_path is not None:
        return Vimeo90k, params.vimeo_path
    return Reds, params.reds_path


def vsrvc(params, kwargs):
    if params.sliding_window is None:
        params.sliding_window = 1
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
    encoder_class = ISRICEncoder
    kwargs["arch_args"]["encoder_kwargs"] = {
        'sliding_window': params.sliding_window,
        'mid_channels': 64,
        'out_channels': 64,
        'num_blocks': 3,
    }
    model_type = "IFrame"
    return train_set, test_set, encoder_class, decoders, kwargs, decoder_kwargs, model_type


def get_sliding_window_datasets(params):
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
    return train_set, test_set


def vsrvc_motion_residual(params, kwargs):
    train_set, test_set = get_sliding_window_datasets(params)
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
        decoders["vsr"] = VSRMotionResidualDecoder(**decoder_kwargs['vsr'])
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


def vsrvc_shallow_encoder(params, kwargs):
    train_set, test_set = get_sliding_window_datasets(params)
    decoder_kwargs: Dict[str, dict] = {}
    decoders: nn.ModuleDict[str, nn.Module] = nn.ModuleDict({})
    if params.vc:
        decoder_kwargs["vc"] = {
            'in_channels': 64,
            'mid_channels': 64,
        }
        decoders["vc"] = VCShallowDecoder(**decoder_kwargs['vc'])
    else:
        decoder_kwargs["vc"] = {}
        decoders["vc"] = DummyVCDecoder()
    if params.vsr:
        decoder_kwargs["vsr"] = {
            'in_channels': 64,
            'mid_channels': 64,
            "scale": params.scale,
        }
        decoders["vsr"] = VSRMotionResidualDecoder(**decoder_kwargs['vsr'])
    else:
        decoder_kwargs["vsr"] = {}
        decoders["vsr"] = DummyVSRDecoder()
    encoder_class = VSRVCShallowEncoder
    kwargs["arch_args"]["encoder_kwargs"] = {
        'in_channels': 3,
        'mid_channels': 64,
        'out_channels': 64,
        'num_blocks': 3,
    }
    model_type = "PFrameNoMotionEncoder"
    return train_set, test_set, encoder_class, decoders, kwargs, decoder_kwargs, model_type


def vsrvc_basic(params, kwargs):
    train_set, test_set = get_sliding_window_datasets(params)
    decoder_kwargs: Dict[str, dict] = {}
    decoders: nn.ModuleDict[str, nn.Module] = nn.ModuleDict({})
    motion_compensator = MotionCompensator(64)
    if params.vc:
        decoder_kwargs["vc"] = {
            'in_channels': 64,
            'mid_channels': 64,
        }
        decoders["vc"] = VCBasicDecoder(**decoder_kwargs['vc'])
    else:
        decoder_kwargs["vc"] = {}
        decoders["vc"] = DummyVCDecoder()
    if params.vsr:
        decoder_kwargs["vsr"] = {
            'in_channels': 64,
            'mid_channels': 64,
            "scale": params.scale,
        }
        decoders["vsr"] = VSRBasicDecoder(**decoder_kwargs['vsr'])
    else:
        decoder_kwargs["vsr"] = {}
        decoders["vsr"] = DummyVSRDecoder()
    encoder_class = VSRVCBasicEncoder
    kwargs["arch_args"]["encoder_kwargs"] = {
        'motion_compensator': motion_compensator,
        'in_channels': 3,
        'mid_channels': 64,
        'num_blocks': 5,
    }
    model_type = "PFrameNoMotionEncoder"
    return train_set, test_set, encoder_class, decoders, kwargs, decoder_kwargs, model_type
