import glob
import json
import os
import time

import torch

from LibMTL.utils import set_random_seed
from datasets import UVGDataset
from loader.model_loader import load_model
from metrics import UVGMetrics
from utils import save_video


def get_eval_filename(cfg: dict):
    name = "eval"
    if 'iframe_model_path' in cfg.keys():
        name += ' ' + cfg['iframe_model_path'].split('/')[-1]
    elif 'keyframe_compress_type' in cfg.keys():
        name += ' ' + cfg['keyframe_compress_type']
    if 'keyframe_interval' in cfg.keys():
        name += ' ' + str(cfg['keyframe_interval'])
    if 'pruning' in cfg.keys():
        name += ' ' + str(cfg['pruning']) + ' ' + str(cfg["pruning_ratio"])
    return name + ".json"


@torch.no_grad()
def _eval_example_no_bitstream(model, dataset, index, meter=None, results=None, save_root=None):
    if meter is None:
        meter = UVGMetrics()
    if results is None:
        results = {"vc_psnr": [], "vc_ssim": [], "vsr_psnr": [], "vsr_ssim": [], "bpp": []}

    inp, gt = dataset[index]
    start = time.time()
    compress_preds = model.compress_no_bitstream(inp)
    compress_time = time.time() - start
    upscaled_video = torch.stack(compress_preds["vsr"], dim=1)
    reconstructed_video = torch.stack(compress_preds["vc"], dim=1)
    meter.update(upscaled_video, None, reconstructed_video, gt)
    log = f"{dataset.get_name_with_index(index)}:\n"
    for key, value in meter.get_records_dict().items():
        log += f"   {key}: {value}\n"
        results[key].append(value)
    log += f"   time: {compress_time}\n"
    meter.reinit()
    print(log)

    if save_root is not None:
        save_video(upscaled_video, save_root, name="upscaled")
        save_video(reconstructed_video, save_root, name="compressed")

    return results, compress_time


@torch.no_grad()
def _eval_example(model, dataset, index, meter=None, results=None, save_root=None):
    if meter is None:
        meter = UVGMetrics()
    if results is None:
        results = {"vc_psnr": [], "vc_ssim": [], "vsr_psnr": [], "vsr_ssim": [], "bpp": []}

    inp, gt = dataset[index]
    start = time.time()
    compress_preds = model.compress(inp)
    compress_time = time.time() - start
    upscaled_video = torch.stack(compress_preds["vsr"], dim=1)
    start = time.time()
    reconstructed_video = model.decompress(compress_preds["vc"])
    decompress_time = time.time() - start
    meter.update(upscaled_video, compress_preds["vc"], reconstructed_video, gt)
    log = f"{dataset.get_name_with_index(index)}:\n"
    for key, value in meter.get_records_dict().items():
        log += f"   {key}: {value}\n"
        results[key].append(value)
    log += f"   compress time: {compress_time}\n"
    log += f"   decompress time: {decompress_time}\n"
    meter.reinit()
    print(log)

    if save_root is not None:
        save_video(upscaled_video, save_root, name="upscaled")
        save_video(reconstructed_video, save_root, name="compressed")

    return results, compress_time


@torch.no_grad()
def eval_one(model_root: str, index, cfg=None, save_root=None):
    dataset = UVGDataset("../../Datasets/UVG", 4)
    model = load_model(model_root, cfg)
    name = dataset.get_name_with_index(index)
    save_root = os.path.join(save_root, name) if save_root is not None else None
    results, compress_time = _eval_example(model, dataset, index, save_root=save_root)
    return results


@torch.no_grad()
def eval_all(model_root: str, cfg=None, save_root=None, write_bitstream=True):
    if cfg is None:
        cfg = {}
    uvg_set = UVGDataset("../../Datasets/UVG", 4)
    model = load_model(model_root, cfg)
    meter = UVGMetrics()
    results = {"vc_psnr": [], "vc_ssim": [], "vsr_psnr": [], "vsr_ssim": [], "bpp": []}
    times = []
    for index in range(len(uvg_set)):
        name = uvg_set.get_name_with_index(index)
        save_path = os.path.join(save_root, name) if save_root is not None else None
        if write_bitstream:
            results, compress_time = _eval_example(model, uvg_set, index, meter=meter, results=results, save_root=save_path)
        else:
            results, compress_time = _eval_example_no_bitstream(model, uvg_set, index, meter=meter, results=results, save_root=save_path)
        times.append(compress_time)
    print(f"AVG TIME: {sum(times) / len(times)}")
    results["meta"] = cfg
    with open(os.path.join(model_root, get_eval_filename(cfg)), "w") as f:
        json.dump(results, f)


def eval_all_models(shared_cfg):
    groups = glob.glob("../weights/*")
    for group in groups:
        if not os.path.isdir(group):
            continue
        models = glob.glob(os.path.join(group, "*"))
        for model in models:
            files = os.listdir(model)
            eval_files = list(filter(lambda x: x.startswith("eval"), files))
            if len(eval_files) == 0:
                print(f"Evaluating {model}...")
                eval_all(model, shared_cfg)
            else:
                print(f"Skipping {model}. Eval file already exists.")


if __name__ == "__main__":
    set_random_seed(777)
    eval_cfg = {
        # "keyframe_compress_type": "jpg",
        # "keyframe_save_root": "../weights/kfs",
        "iframe_model_path": "../weights/ISRIC/128",
        "keyframe_interval": 12,
    }
    # eval_all_models(eval_cfg)
    # "VSRVC shallow/128", "VC shallow/128", "VSR shallow/128",
    for model in ["VSRVC basic/256", "VSRVC basic/384", "VSRVC basic/512",
                  "VSRVC basic/640", "VSRVC basic/GradNorm 384", "VSRVC basic/GradNorm 640"]:
        tested_model = os.path.join(f"../weights/{model}")
        eval_all(tested_model, eval_cfg, save_root=None, write_bitstream=True)

