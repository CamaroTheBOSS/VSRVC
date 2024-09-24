import glob
import os

import wandb

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from torchvision.transforms import Compose, ToTensor
from PIL import Image

from LibMTL.utils import set_random_seed
from datasets import UVGDataset
from evaluate import _eval_example
from metrics import psnr
from plots import _validate_task, load_eval_file
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


def draw_box(img, box, color=(0, 0, 255), thickness=3):
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
    dataset = UVGDataset("../../Datasets/UVG", 4)
    gt = to_cv2(dataset[example][1][task][0, frame_idx])
    H, W, _ = gt.shape
    folder = "compressed" if task == "vc" else "upscaled"
    for i, (root, generate) in enumerate(zip(model_roots, gen_mask)):
        if generate:
            model = load_model(root)
            _eval_example(model, dataset, example, save_root=root)
        paths = glob.glob(os.path.join(root, f"{folder}/*.png"))
        if len(paths) == 0:
            paths = glob.glob(os.path.join(root, f"*.png"))
        path = paths[frame_idx]
        img = crop_box(cv2.imread(path), box)
        cv2.imwrite(os.path.join(save_root, f"{task}_{i}.png"), img)
    original_box = np.copy(crop_box(gt, box))
    original = draw_box(gt, box, thickness=2 if task == "vc" else 8)
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


def plot_loss(run_strings, legend):
    scan_data = scan_history_multiple(run_strings, ["train_vc_loss", "train_vsr_loss"])
    for i, data in enumerate(scan_data):
        fig = plt.figure()
        vc_loss = moving_average(data["train_vc_loss"], 10)
        vsr_loss = moving_average(data["train_vsr_loss"], 10)
        z = np.power(np.linspace(0.0005, 1.0, len(vc_loss)), 1 / 8)
        colorline(vc_loss, vsr_loss, z=z, cmap=plt.get_cmap('YlOrRd'))
        plt.xlim([np.min(vc_loss), np.max(vc_loss)])
        plt.ylim([np.min(vsr_loss), np.max(vsr_loss)])
        plt.xlabel("Funkcja kosztu zadania kompresji")
        plt.ylabel("Funkcja kosztu zadania super-rozdzielczości")
        plt.title(legend[i])
    plt.show()


def scan_history(run_string, keys):
    api = wandb.Api()
    run = api.run(run_string)
    dataframe = run.scan_history(keys=keys)
    array = np.array([[row[key] for key in keys] for row in dataframe]).transpose()
    return {key: arr for key, arr in zip(keys, array)}


def scan_history_multiple(run_strings, keys):
    return [scan_history(run_str, keys) for run_str in run_strings]


def set_x_ticks_to_epochs(fig, steps_per_epoch=2328, sparsity=1):
    epochs = int(fig.axes[0].get_xlim()[-1] / steps_per_epoch)
    ticks = np.linspace(0, epochs * steps_per_epoch, int(epochs / sparsity) + 1)
    tick_labels = np.linspace(0, epochs, int(epochs / sparsity) + 1, dtype=int)
    plt.xticks(ticks, tick_labels)
    plt.xlabel("Epoki")
    return fig


def get_y_label_dict():
    return {"grad_vsr_norm": "Norma gradientu zadania super-rozdzielczości",
            "grad_vc_norm": "Norma gradientu zadania kompresji",
            "grad_cos_angle": "Podobieństwo kosinusowe"}


def plot_history(scanned_history, key, mode="per batch", steps_per_epoch=2328, fig=None):
    supported_modes = ["per batch", "moving avg", "per epoch"]
    if mode not in supported_modes:
        raise ValueError(f"Unrecognized mode value. Supported are {supported_modes}")
    if fig is None:
        fig = plt.figure()
    if mode == "per batch":
        for history in scanned_history:
            plt.plot(history[key])
    elif mode == "moving avg":
        for history in scanned_history:
            plt.plot(moving_average(history[key], 200))
    elif mode == "per epoch":
        for history in scanned_history:
            steps = len(history[key])
            epochs = steps // steps_per_epoch
            per_epoch_x = np.arange(steps_per_epoch, steps, steps_per_epoch)
            history_reshaped = history[key][:epochs * steps_per_epoch].reshape(-1, steps_per_epoch)
            plt.plot(per_epoch_x, history_reshaped.mean(axis=1))
    fig = set_x_ticks_to_epochs(fig, steps_per_epoch, sparsity=2)
    plt.ylabel(get_y_label_dict()[key])

    return fig


def plot_grad_conflict_ratio(scanned_history, steps_per_epoch=2328):
    fig = plt.figure()
    for history in scanned_history:
        steps = len(history["grad_cos_angle"])
        epochs = steps // steps_per_epoch
        history_reshaped = history["grad_cos_angle"][:epochs * steps_per_epoch].reshape(-1, steps_per_epoch)
        grad_conflicts = np.round((history_reshaped < 0).mean(axis=1) * 100, decimals=2)
        plt.plot(grad_conflicts)
    plt.ylabel("Współczynnik paczek danych ze skonfliktowanymi gradientami [%]")
    plt.xlabel("Epoki")
    return fig


def plot_grad_stats(run_strings, mode="per batch", legend=None):
    keys = ["grad_vsr_norm", "grad_vc_norm", "grad_cos_angle"]
    scan_data = [scan_history(run_str, keys) for run_str in run_strings]

    for key in keys:
        fig = plot_history(scan_data, key=key, mode=mode)
        if legend is not None:
            plt.legend(legend)
    conflicts = plot_grad_conflict_ratio(scan_data)
    if legend is not None:
        plt.legend(legend)
    plt.show()


def get_stats_for_frame(eval_files, vid_idx, frame_idx):
    for eval_file in eval_files:
        eval_data = load_eval_file(eval_file)
        result = {
            "bpp": eval_data["bpp"][vid_idx][frame_idx].sum(),
            "vc_psnr": eval_data["vc_psnr"][vid_idx][frame_idx],
            "vc_ssim": eval_data["vc_ssim"][vid_idx][frame_idx],
            "vsr_psnr": eval_data["vsr_psnr"][vid_idx][frame_idx],
            "vsr_ssim": eval_data["vsr_ssim"][vid_idx][frame_idx]
        }
        print(f"frame {vid_idx}.{frame_idx}: {result}")


if __name__ == "__main__":
    # set_random_seed(777)
    # shallow_algorithms = [
    #     "camarotheboss/VSRVC/nsljta4h",  # EW
    #     "camarotheboss/VSRVC/gq3tvmdf",  # GradNorm
    #     "camarotheboss/VSRVC/oqokt5n7",  # DB_MTL
    #     "camarotheboss/VSRVC/cg4vyau5"   # GradVac
    # ]
    # legend = ["EW", "GradNorm", "DB_MTL", "GradVac"]
    dbmtl_other_tasks = [
        "camarotheboss/VSRVC/ijngp6ix",
        "camarotheboss/VSRVC/4bb8u27y",
        "camarotheboss/VSRVC/7p6saglr",
        "camarotheboss/VSRVC/6obkwi1n",
    ]
    legend = ["SegDepth DBMTL", "classif DBMTL", "classif EW", "SegDepth EW"]
    tasks = [["segmentation", "depth"], ["Real_World", "Art"], ["Real_World", "Art"], ["segmentation", "depth"]]
    # plot_grad_stats(dbmtl_other_tasks, mode="per epoch", legend=legend)
    plot_loss(dbmtl_other_tasks, legend)
    # fvc_images = r"D:\Code\ENVS\PyTorchVideoCompression\FVC\woutputs"
    # dcvc_images = r"D:\Code\DCVC\DCVC-FM\out_bin\UVG\rate_4\ShakeNDry"
    # vc_shallow_images = r"..\weights\VC shallow\256"
    # shallow_images = r"..\weights\VSRVC shallow\512"
    # hevc_images = r"..\..\VSRVC\evaluation\new_outputs\ShakeNDry_1920x1080_120fps_420_8bit_YUV"
    # mosaic("vc", [hevc_images], 5,
    #        save_root="../weights", box=(150, 100, 64, 64), frame_idx=10)
    # get_stats_for_frame([
    #     "../weights/VSRVC shallow/512/eval 128 12.json",
    #     "../weights/VC shallow/256/eval 128 12.json",
    #     "../weights/DCVC-FM_rate_4.json",
    #     "../weights/fvc-8192.json",
    # ], 5, 10)
    #
    # basicvsr_images = r"D:\Code\ENVS\BasicVSR_PlusPlus\outputs\vimeo90k_bd\YachtRide_0"
    # iart_images = r"D:\Code\ENVS\IART\results\vimeo90k_BDx4_UVG\YachtRide_0"
    # bilinear_images = r"D:\Code\Datasets\UVG_bilinear\YachtRide"
    # vsr_shallow_images = r"..\weights\VSR shallow\128"
    # mosaic("vsr", [bilinear_images, basicvsr_images, iart_images, shallow_images, vsr_shallow_images], 6,
    #        save_root="../weights", box=(525, 475, 128, 128), frame_idx=10)
    # get_stats_for_frame([
    #     "../weights/VSRVC shallow/512/eval 128 12.json",
    #     "../weights/VSR shallow/128/eval 128 12.json",
    #     "../weights/basicvsr_plusplus_trained.json",
    #     "../weights/iart_bd.json",
    # ], 6, 10)

