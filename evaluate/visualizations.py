import glob
import os

import cv2
import numpy as np

from LibMTL.utils import set_random_seed
from datasets import UVGDataset
from evaluate import _eval_example
from plots import _validate_task
from loader.model_loader import load_model
from utils import to_cv2


def validate_box(box, H, W):
    y1 = max(box[1], 0)
    y2 = min(y1 + box[3], H)
    x1 = max(box[0], 0)
    x2 = min(x1 + box[2], W)
    return y1, y2, x1, x2


def crop_box(img, box):
    H, W, _ = img.shape
    y1, y2, x1, x2 = validate_box(box, H, W)
    return img[y1:y2, x1:x2]


def draw_box(img, box, color=(0, 0, 255), thickness=10):
    H, W, _ = img.shape
    y1, y2, x1, x2 = validate_box(box, H, W)
    img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img


def mosaic(task, model_roots, example, gen_mask=None, box=(0, 0, 100, 100), frame_idx=-1, save_root="."):
    _validate_task(task)
    if gen_mask is None:
        gen_mask = [0 for _ in model_roots]
    if len(gen_mask) != len(model_roots):
        raise ValueError("Incompatible sizes, gen_mask and model_jsons must have the same size!")
    dataset = UVGDataset("../../Datasets/UVG", 2)
    gt = to_cv2(dataset[example][1][task][0, frame_idx])
    H, W, _ = gt.shape
    folder = "compressed" if task == "vc" else "upscaled"
    for i, (root, generate) in enumerate(zip(model_roots, gen_mask)):
        if generate:
            model = load_model(root)
            _eval_example(model, dataset, example, save_root=root)
        path = glob.glob(os.path.join(root, f"{folder}/*.png"))[frame_idx]
        img = crop_box(cv2.imread(path), box)
        cv2.imwrite(os.path.join(save_root, f"{task}_{i}.png"), img)
    original_box = np.copy(crop_box(gt, box))
    original = draw_box(gt, box)
    cv2.imwrite(os.path.join(save_root, f"{task}_original_box.png"), original_box)
    cv2.imwrite(os.path.join(save_root, f"{task}_original.png"), original)


if __name__ == "__main__":
    set_random_seed(777)
    mosaic("vsr", ["../weights/isric 1024"], 4, save_root="../weights", box=(300, 100, 100, 100))
