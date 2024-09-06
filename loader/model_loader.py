import json
import math
import os
import time

import torch
from torch import nn
from torch.nn.utils import prune
from torchvision.transforms import ToTensor

import loader as modules
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from PIL import Image

import weighting as weighting_method
import archs as architecture_method
from utils import save_frame, interpolate_frame

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def process_arch_args(arch_args):
    for key, arg in arch_args["encoder_kwargs"].items():
        if isinstance(arg, dict):
            module = modules.__dict__[arg["class"]]
            kwargs = arg["kwargs"]
            arch_args["encoder_kwargs"][key] = module(**kwargs)

    return arch_args


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    """Returns table of logarithmically scales."""
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class ByteSize:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


class KeyFrameModel(nn.Module):
    def __init__(self, cfg, device):
        super(KeyFrameModel, self).__init__()
        self.compress = cfg["keyframe_compress_type"]
        self.save_root = cfg["keyframe_save_root"]
        self.scale = cfg["scale"]
        self.kf_counter = 0
        self.supported_compress_types = ["jpg", "png"]
        self.device = device
        self.to(device)

    def compress_one(self, inputs):
        if self.compress not in ["jpg", "png"]:
            raise ValueError(f"Unrecognized keyframe_compress_type. Supported are {self.supported_compress_types}")
        filepath = os.path.join(self.save_root, f"kf_{self.kf_counter}.{self.compress}")
        save_frame(filepath, inputs[0, 0])
        byte_size = os.stat(filepath).st_size
        self.kf_counter += 1
        out = {
            "vsr": interpolate_frame(inputs[0], size=(inputs.size()[-2] * self.scale, inputs.size()[-1] * self.scale)),
            "vc": [([ByteSize(byte_size)], [''], filepath), ([''], [''], 0)]
        }
        return out

    def decompress_one(self, inputs):
        if self.compress not in ["jpg", "png"]:
            raise ValueError(f"Unrecognized keyframe_compress_type. Supported are {self.supported_compress_types}")
        path = inputs[2]
        recon = ToTensor()(Image.open(path).convert("RGB")).to(self.device)
        return recon.unsqueeze(0)


def prune_model(model, cfg):
    if "pruning" not in cfg:
        return model
    if "pruning_ratio" not in cfg:
        raise ValueError("Pruning method specified, but pruning_ratio not!")

    supported_prunings = ["l1_unstructured"]
    if cfg["pruning"] == "l1_unstructured":
        method = prune.l1_unstructured
    else:
        raise ValueError(f"Pruning method not supported. Supported are {supported_prunings}")

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            method(module, name='weight', amount=cfg["pruning_ratio"])
    return model


def load_model(json_file, cfg=None):
    if cfg is None:
        cfg = {}
    if os.path.isdir(json_file):
        json_file = os.path.join(json_file, "model.json")
    with open(json_file, "r") as f:
        model_data = json.load(f)
    model_type = model_data["model_type"]
    if model_type.startswith("PFrame"):
        if "keyframe_compress_type" not in cfg.keys() and "iframe_model_path" not in cfg.keys():
            raise KeyError("iframe_model_path key doesn't exist. Please provide path (str) to it in config")
        if "keyframe_interval" not in cfg.keys():
            raise KeyError("keyframe_interval key doesn't exist. Please provide value (int) in config")
    encoder_class = modules.__dict__[model_data["encoder_class"]]
    decoders = nn.ModuleDict({d["task"]: modules.__dict__[d["module"]](**d["kwargs"]) for d in model_data["decoders"]})
    weighting = weighting_method.__dict__[model_data["weighting"]]
    architecture = architecture_method.__dict__[model_data["architecture"]]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    class BaseModel(architecture, weighting):
        def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, kwargs, cfg):
            super(BaseModel, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device,
                                            **kwargs)
            self.cfg = cfg

        def compress(self, x):
            pass

        def decompress(self, x):
            pass

        def update(self, scale_table=None, force=False, update_quantiles: bool = False):
            if scale_table is None:
                scale_table = get_scale_table()
            updated = False
            for _, module in self.named_modules():
                if isinstance(module, EntropyBottleneck):
                    updated |= module.update(force=force, update_quantiles=update_quantiles)
                if isinstance(module, GaussianConditional):
                    updated |= module.update_scale_table(scale_table, force=force)
            return updated

    class IFrameModel(BaseModel):
        def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, lmbda, kwargs, cfg):
            super(IFrameModel, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device,
                                              kwargs, cfg)
            self.init_param()
            self.lmbda = lmbda

        def compress_one(self, inp):
            out = {}
            s_rep = self.encoder(inp)
            same_rep = True if not isinstance(s_rep, list) and not self.multi_input else False
            for tn, task in enumerate(self.task_name):
                ss_rep = s_rep[tn] if isinstance(s_rep, list) else s_rep
                ss_rep = self._prepare_rep(ss_rep, task, same_rep)
                if task == "vc":
                    out[task] = self.decoders[task].compress(ss_rep)
                else:
                    out[task] = self.decoders[task](ss_rep)
            return out

        def decompress_one(self, inputs):
            return self.decoders["vc"].decompress(inputs)

        def compress(self, video):
            out = {task: [] for task in self.task_name}
            for i in range(0, video.size()[1]):
                inp = video[:, i:i + 1]
                results = self.compress_one(inp)
                for task, value in results.items():
                    out[task].append(value)
            return out

        def decompress(self, inputs):
            reconstructed_video = []
            for inp in inputs:
                reconstructed_video.append(self.decompress_one(inp[0]))
            return torch.stack(reconstructed_video, dim=1)

    class PFrameModelWithMotion(BaseModel):
        def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, lmbda, kwargs, cfg):
            super(PFrameModelWithMotion, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input,
                                                        device, kwargs, cfg)
            self.init_param()
            self.lmbda = lmbda
            if "keyframe_compress_type" in cfg.keys():
                self.iframe_model = KeyFrameModel(cfg, device)
            else:
                self.iframe_model: IFrameModel = load_model(cfg["iframe_model_path"])
            self.keyframe_interval = cfg["keyframe_interval"]

        def compress(self, video):
            out = {task: [] for task in self.task_name}
            prev_recon = None
            for i in range(0, video.size()[1]):
                if i % self.keyframe_interval == 0:
                    inp = video[:, i:i + 1]
                    results = self.iframe_model.compress_one(inp)
                    if len(results["vc"]) < 2:
                        results["vc"].append(([''], [''], 0))
                    prev_recon = self.iframe_model.decompress_one(results["vc"][0])
                    for task, value in results.items():
                        out[task].append(value)
                    continue
                inp = video[:, i - 1:i + 1]
                s_rep = self.encoder.compress(prev_recon, inp)
                same_rep = True if not isinstance(s_rep, list) and not self.multi_input else False
                for tn, task in enumerate(self.task_name):
                    ss_rep = s_rep[tn] if isinstance(s_rep, list) else s_rep
                    ss_rep = self._prepare_rep(ss_rep, task, same_rep)
                    if task == "vc":
                        results = self.decoders[task].compress(ss_rep)
                        inp = (ss_rep[0],) + results[0]
                        prev_recon = self.decoders[task].decompress(inp)
                        out[task].append(results)
                    else:
                        out[task].append(self.decoders[task](ss_rep))
            return out

        def decompress_shallow(self, inputs):
            reconstructed_video = []
            prev_feat = None
            for i, inp in enumerate(inputs):
                if i % self.keyframe_interval == 0:
                    recon = self.iframe_model.decompress_one(inp[0])
                else:
                    inp = (prev_feat,) + inp[0]
                    recon = self.decoders["vc"].decompress(inp)
                prev_feat = self.encoder.extract_feats(recon)
                reconstructed_video.append(recon)
            return torch.stack(reconstructed_video, dim=1)

        def decompress(self, inputs):
            reconstructed_video = []
            prev_feat = None
            for i, inp in enumerate(inputs):
                if i % self.keyframe_interval == 0:
                    recon = self.iframe_model.decompress_one(inp[0])
                else:
                    recon_offsets = self.encoder.decompress(inp[1])
                    align_feat = self.encoder.align_features(prev_feat, recon_offsets)
                    inp = (align_feat,) + inp[0]
                    recon = self.decoders["vc"].decompress(inp)
                prev_feat = self.encoder.extract_feats(recon)
                reconstructed_video.append(recon)
            return torch.stack(reconstructed_video, dim=1)

    class PFrameNoMotionEncoder(BaseModel):
        def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, lmbda, kwargs, cfg):
            super(PFrameNoMotionEncoder, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input,
                                                        device, kwargs, cfg)
            self.init_param()
            self.lmbda = lmbda
            if "keyframe_compress_type" in cfg.keys():
                self.iframe_model = KeyFrameModel(cfg, device)
            else:
                self.iframe_model: IFrameModel = load_model(cfg["iframe_model_path"])
            self.keyframe_interval = cfg["keyframe_interval"]

        def compress(self, video):
            out = {task: [] for task in self.task_name}
            prev_recon = None
            for i in range(0, video.size()[1]):
                if i % self.keyframe_interval == 0:
                    inp = video[:, i:i + 1]
                    results = self.iframe_model.compress_one(inp)
                    if len(results["vc"]) < 2:
                        results["vc"].append(([''], [''], 0))
                    prev_recon = self.iframe_model.decompress_one(results["vc"][0])
                    for task, value in results.items():
                        out[task].append(value)
                    continue
                inp = video[:, i - 1:i + 1]
                start = time.time()
                s_rep = self.encoder.compress(prev_recon, inp)
                same_rep = True if not isinstance(s_rep, list) and not self.multi_input else False
                for tn, task in enumerate(self.task_name):
                    ss_rep = s_rep[tn] if isinstance(s_rep, list) else s_rep
                    ss_rep = self._prepare_rep(ss_rep, task, same_rep)
                    if task == "vc":
                        results = self.decoders[task].compress(ss_rep)
                        inp = (ss_rep[0],) + results[0] + results[1]
                        prev_recon = self.decoders[task].decompress(inp)
                        out[task].append(results)
                    else:
                        out[task].append(self.decoders[task](ss_rep))
                end = time.time()
                print(f"Inference time: {end - start}")
            return out

        def decompress(self, inputs):
            reconstructed_video = []
            prev_feat = None
            for i, inp in enumerate(inputs):
                if i % self.keyframe_interval == 0:
                    recon = self.iframe_model.decompress_one(inp[0])
                else:
                    inp = (prev_feat,) + inp[0] + inp[1]
                    recon = self.decoders["vc"].decompress(inp)
                prev_feat = self.encoder.extract_feats(recon)
                reconstructed_video.append(recon)
            return torch.stack(reconstructed_video, dim=1)

    cfg["scale"] = model_data["scale"]
    model_data['arch_args'] = process_arch_args(model_data['arch_args'])
    kwargs = dict(task_name=model_data["task_name"],
                  encoder_class=encoder_class,
                  decoders=decoders,
                  rep_grad=model_data["rep_grad"],
                  multi_input=False,
                  device=device,
                  lmbda=model_data["lmbda"],
                  kwargs=model_data['arch_args'],
                  cfg=cfg)
    supported_model_types = ['IFrame', 'PFrameWithMotion', 'PFrameNoMotionEncoder']
    if model_type == "IFrame":
        model = IFrameModel(**kwargs).to(device)
    elif model_type == "PFrameWithMotion":
        model = PFrameModelWithMotion(**kwargs).to(device)
    elif model_type == "PFrameNoMotionEncoder":
        model = PFrameNoMotionEncoder(**kwargs).to(device)
    else:
        raise ValueError(f"Unrecognized model_type. Supported are {supported_model_types}")
    for decoder in model.decoders.values():
        try:
            decoder.share(model.encoder)
        except:
            pass
    strict = False if model_type.startswith("PFrame") else True
    model.load_state_dict(torch.load(model_data["checkpoint"]), strict=strict)
    model = prune_model(model, cfg)
    model.eval()
    model.update(force=True)
    print(f"Model loaded with cfg: {cfg}")
    return model
