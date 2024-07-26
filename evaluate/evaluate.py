import json
import os
import torch

from LibMTL.utils import set_random_seed
from datasets import UVGDataset
from loader.model_loader import load_model
from metrics import UVGMetrics
from utils import save_video


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
def eval_one(model_root: str, index, save_root=None):
    dataset = UVGDataset("../../Datasets/UVG", 2)
    model = load_model(model_root)
    results = _eval_example(model, dataset, index, save_root=save_root)
    return results


@torch.no_grad()
def eval_all(model_root: str):
    uvg_set = UVGDataset("../../Datasets/UVG", 2)
    model = load_model(model_root)
    meter = UVGMetrics()
    results = {"vc_psnr": [], "vc_ssim": [], "vsr_psnr": [], "vsr_ssim": [], "bpp": []}
    for index in range(len(uvg_set)):
        results = _eval_example(model, uvg_set, index, meter=meter, results=results)

    with open(os.path.join(model_root, "eval.json"), "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    set_random_seed(777)
    eval_one("../weights/isric 1024", 4, "../weights/isric 1024")
    # eval_all("../weights/isric 1024")
