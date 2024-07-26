import json
import math
import os

import torch
from torch import nn

import loader as modules
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

import LibMTL.weighting as weighting_method
import LibMTL.architecture as architecture_method

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    """Returns table of logarithmically scales."""
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


def load_model(json_file):
    if os.path.isdir(json_file):
        json_file = os.path.join(json_file, "model.json")
    with open(json_file, "r") as f:
        model_data = json.load(f)
    encoder_class = modules.__dict__[model_data["encoder_class"]]
    decoders = nn.ModuleDict({d["task"]: modules.__dict__[d["module"]](**d["kwargs"]) for d in model_data["decoders"]})
    weighting = weighting_method.__dict__[model_data["weighting"]]
    architecture = architecture_method.__dict__[model_data["architecture"]]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    class MTLmodel(architecture, weighting):
        def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, lmbda, kwargs):
            super(MTLmodel, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device,
                                           **kwargs)
            self.init_param()
            self.lmbda = lmbda
            self.sw = kwargs["encoder_kwargs"]["sliding_window"]

        def compress(self, inputs, task_name=None):
            out = {task: [] for task in self.task_name}
            for i in range(1, inputs.size()[1] + 1):
                if i < self.sw:
                    inp = torch.stack([*[torch.zeros_like(inputs[:, i]) for _ in range(self.sw - i)], inputs[:, i-1]])
                else:
                    inp = inputs[:, i - self.sw:i]
                s_rep = self.encoder(inp)
                same_rep = True if not isinstance(s_rep, list) and not self.multi_input else False
                for tn, task in enumerate(self.task_name):
                    if task_name is not None and task != task_name:
                        continue
                    ss_rep = s_rep[tn] if isinstance(s_rep, list) else s_rep
                    ss_rep = self._prepare_rep(ss_rep, task, same_rep)
                    if task == "vc":
                        out[task].append(self.decoders[task].compress(ss_rep))
                    else:
                        out[task].append(self.decoders[task](ss_rep))
            return out

        def decompress(self, inputs):
            reconstructed_video = []
            for inp in inputs:
                reconstructed_video.append(self.decoders["vc"].decompress(*inp))
            return torch.stack(reconstructed_video, dim=1)

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

    model = MTLmodel(task_name=model_data["task_name"],
                     encoder_class=encoder_class,
                     decoders=decoders,
                     rep_grad=model_data["rep_grad"],
                     multi_input=model_data["multi_input"],
                     device=device,
                     lmbda=model_data["lmbda"],
                     kwargs=model_data['arch_args']).to(device)
    model.load_state_dict(torch.load(model_data["checkpoint"]))
    return model
