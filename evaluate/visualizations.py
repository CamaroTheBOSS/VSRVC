import glob
import os
import wandb

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import matplotlib.path as mpath

from LibMTL.utils import set_random_seed
from datasets import UVGDataset
from evaluate import _eval_example
from plots import _validate_task
from loader.model_loader import load_model
from utils import to_cv2

wandb.require("core")

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


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def colorline(
    x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def loss_plot(run_string):
    api = wandb.Api()
    run = api.run(run_string)
    metrics_dataframe = run.scan_history(keys=["train_vc_loss", "train_vsr_loss"])
    vc_loss = moving_average(np.array([row["train_vc_loss"] for row in metrics_dataframe]), 10)
    vsr_loss = moving_average(np.array([row["train_vsr_loss"] for row in metrics_dataframe]), 10)
    z = np.power(np.linspace(0.0005, 1.0, len(vc_loss)), 1/8)
    colorline(vc_loss, vsr_loss, z=z, cmap=plt.get_cmap('YlOrRd'))
    plt.xlim([np.min(vc_loss), np.max(vc_loss)])
    plt.ylim([np.min(vsr_loss), np.max(vsr_loss)])
    plt.xlabel("vc loss")
    plt.ylabel("vsr loss")
    plt.show()


if __name__ == "__main__":
    set_random_seed(777)
    loss_plot("camarotheboss/VSRVC/zvg2v1fl")
    # mosaic("vsr", ["../weights/isric 1024"], 4, save_root="../weights", box=(300, 100, 100, 100))
