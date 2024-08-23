import json
import os
import torch

from LibMTL.utils import set_random_seed
from datasets import UVGDataset
from loader.model_loader import load_model
from metrics import UVGMetrics
from utils import save_video


def get_eval_filename(cfg: dict):
    name = "eval"
    if 'iframe_model_path' in cfg.keys():
        name += ' ' + cfg['iframe_model_path'].split(' ')[1]
    elif 'keyframe_compress_type' in cfg.keys():
        name += ' ' + cfg['keyframe_compress_type']
    if 'keyframe_interval' in cfg.keys():
        name += ' ' + str(cfg['keyframe_interval'])
    if 'adaptation' in cfg.keys():
        name += ' ' + ('adapt' if cfg['adaptation'] else '')
    return name + ".json"


@torch.no_grad()
def _eval_example(model, dataset, index, meter=None, results=None, save_root=None):
    if meter is None:
        meter = UVGMetrics()
    if results is None:
        results = {"vc_psnr": [], "vc_ssim": [], "vsr_psnr": [], "vsr_ssim": [], "bpp": []}

    inp, gt = dataset[index]
    compress_preds = model.compress(inp)
    upscaled_video = torch.stack(compress_preds["vsr"], dim=1)
    reconstructed_video = model.decompress(compress_preds["vc"])
    meter.update(upscaled_video, compress_preds["vc"], reconstructed_video, gt)
    log = f"{dataset.get_name_with_index(index)}:\n"
    for key, value in meter.get_records_dict().items():
        log += f"   {key}: {value}\n"
        results[key].append(value)
    meter.reinit()
    print(log)

    if save_root is not None:
        save_video(upscaled_video, save_root, name="upscaled")
        save_video(reconstructed_video, save_root, name="compressed")

    return results


@torch.no_grad()
def eval_one(model_root: str, index, cfg=None, save_root=None):
    dataset = UVGDataset("../../Datasets/UVG", 4)
    model = load_model(model_root, cfg)
    results = _eval_example(model, dataset, index, save_root=save_root)
    return results


@torch.no_grad()
def eval_all(model_root: str, cfg=None):
    if cfg is None:
        cfg = {}
    uvg_set = UVGDataset("../../Datasets/UVG", 4)
    model = load_model(model_root, cfg)
    meter = UVGMetrics()
    results = {"vc_psnr": [], "vc_ssim": [], "vsr_psnr": [], "vsr_ssim": [], "bpp": []}
    for index in range(len(uvg_set)):
        results = _eval_example(model, uvg_set, index, meter=meter, results=results)
    results["meta"] = cfg
    with open(os.path.join(model_root, get_eval_filename(cfg)), "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    set_random_seed(777)
    eval_cfg = {
        # "keyframe_compress_type": "jpg",
        # "keyframe_save_root": "../weights/kfs",
        "iframe_model_path": "../weights/ISRIC 128 EW x4 vimeo",
        "keyframe_interval": 12,
        "adaptation": True
    }
    tested_model = "../weights/VSRVC mv 128 GradNorm x4 vimeo"
    eval_all(tested_model, eval_cfg)
    eval_one(tested_model, 5, eval_cfg, tested_model)
