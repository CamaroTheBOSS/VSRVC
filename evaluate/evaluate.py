import json
import os
import torch

from LibMTL.utils import set_random_seed
from datasets import UVGDataset
from loader.model_loader import load_model
from metrics import UVGMetrics


def evaluate(model_root: str):
    set_random_seed(777)
    uvg_set = UVGDataset("../../Datasets/UVG", 2, max_frames=10)
    model = load_model(model_root)

    meter = UVGMetrics()
    model.eval()
    model.update(force=True)
    with torch.no_grad():
        results = {"vc_psnr": [], "vc_ssim": [], "vsr_psnr": [], "vsr_ssim": [], "bpp": []}
        for index in range(len(uvg_set)):
            inp, gt = uvg_set[index]
            compress_preds = model.compress(inp)
            compress_preds["vsr"] = torch.stack(compress_preds["vsr"], dim=1)
            reconstructed_video = model.decompress(compress_preds["vc"])
            meter.update(compress_preds["vsr"], compress_preds["vc"], reconstructed_video, gt)
            log = f"{uvg_set.get_name_with_index(index)}:\n"
            for key, value in meter.get_records_dict().items():
                log += f"   {key}: {value}\n"
                results[key].append(value)
            meter.reinit()
            print(log)

    with open(os.path.join(model_root, "eval.json"), "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    evaluate("../weights/isric 1024")
