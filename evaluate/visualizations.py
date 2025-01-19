import copy
import glob
import json
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
from plots import _validate_task, load_eval_file, load_alg_database
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


def mosaic(task, model_roots, example, gen_mask=None, box=(0, 0, 100, 100), frame_idx=-1, save_root=".", legend=None, n=""):
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
        name = f"{task}_{legend[i]}.png" if legend is not None else f"{task}_{i}.png"
        cv2.imwrite(os.path.join(save_root, name), img)
    original_box = np.copy(crop_box(gt, box))
    original = draw_box(gt, box, thickness=2 if task == "vc" else 8)
    name = f"{task}_{n}_ORIGINAL_BOX.png"
    cv2.imwrite(os.path.join(save_root, name), original_box)
    name = f"{task}_{n}_ORIGINAL.png"
    cv2.imwrite(os.path.join(save_root, name), original)


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


def plot_loss(run_strings, legend, metric_keys, xy_labels):
    scan_data = scan_history_multiple(run_strings, metric_keys)
    for i, (data, keys, labels) in enumerate(zip(scan_data, metric_keys, xy_labels)):
        fig = plt.figure()
        vc_loss = moving_average(data[keys[0]], 10)
        vsr_loss = moving_average(data[keys[1]], 10)
        z = np.power(np.linspace(0.0005, 1.0, len(vc_loss)), 1 / 8)
        colorline(vc_loss, vsr_loss, z=z, cmap=plt.get_cmap('YlOrRd'))
        plt.xlim([np.min(vc_loss), np.max(vc_loss)])
        plt.ylim([np.min(vsr_loss), np.max(vsr_loss)])
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.title(legend[i])
    plt.show()


def scan_history_api(run_string, keys):
    api = wandb.Api()
    run = api.run(run_string)
    dataframe = run.scan_history(keys=keys)
    array = np.array([[row[key] for key in keys] for row in dataframe]).transpose()
    return {key: arr for key, arr in zip(keys, array)}


def dump_scan_history(data, path):
    for key, val in data.items():
        if isinstance(val, np.ndarray):
            data[key] = list(val)
    with open(path, "w") as f:
        json.dump(data, f)


def load_scan_history(path):
    with open(path, "r") as f:
        data = json.load(f)
    for key, val in data.items():
        data[key] = np.array(val)
    return data


def scan_history(run_string, keys):
    path = run_string.replace("/", "-") + ".json"
    if not os.path.exists(path):
        data = scan_history_api(run_string, keys)
        dump_scan_history(data, path)
        return load_scan_history(path)

    data = load_scan_history(path)
    remaining_keys = list(filter(lambda key: key not in data.keys(), copy.copy(keys)))
    if len(remaining_keys) > 0:
        new_data = scan_history_api(run_string, remaining_keys)
        for key, val in new_data.items():
            data[key] = val
        dump_scan_history(data, path)
        return load_scan_history(path)
    return data


def scan_history_multiple(run_strings, keys):
    hist = []
    for run_str, key in zip(run_strings, keys):
        hist.append(scan_history(run_str, key))
    return hist


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


def plot_gradnorms(run_strings, mode="per epoch", legend=None):
    keys = ["grad_vsr_norm", "grad_vc_norm"]
    scan_data = [scan_history(run_str, keys) for run_str in run_strings]
    steps_per_epoch = 2328
    for e, data in enumerate(scan_data):
        fig = plt.figure()
        for key in keys:
            steps = len(data[key])
            epochs = steps // steps_per_epoch
            per_epoch_x = np.arange(steps_per_epoch, steps, steps_per_epoch)
            history_reshaped = data[key][:epochs * steps_per_epoch].reshape(-1, steps_per_epoch)
            plt.plot(per_epoch_x, history_reshaped.mean(axis=1))
        fig = set_x_ticks_to_epochs(fig, steps_per_epoch, sparsity=2)
        plt.ylabel("Norma")
        plt.xlabel("Epoka")
        if legend is not None:
            plt.legend(["Norma gradientu zadania super-rozdzielczości", "Norma gradientu zadania kompresji"])
            plt.title(legend[e])
    plt.show()


def get_stats_for_frame(eval_files, vid_idx, frame_idx):
    for eval_file in eval_files:
        if eval_file.startswith("db"):
            eval_data = load_alg_database(eval_file, "hevc")
            result = {
                "bpp": eval_data["bpp"][10][vid_idx].mean(),
                "vc_psnr": eval_data["vc_psnr"][10][vid_idx][frame_idx],
                "vc_ssim": eval_data["vc_ssim"][10][vid_idx][frame_idx],
            }
            eval_data = load_alg_database(eval_file, "bilinear")
            result["vsr_psnr"] = eval_data["vsr_psnr"][vid_idx][frame_idx]
            result["vsr_ssim"] = eval_data["vsr_ssim"][vid_idx][frame_idx]

        else:
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
    shallow_algorithms = [
        "camarotheboss/VSRVC/nsljta4h",  # EW
        "camarotheboss/VSRVC/gq3tvmdf",  # GradNorm
        "camarotheboss/VSRVC/oqokt5n7",  # DB_MTL
        "camarotheboss/VSRVC/cg4vyau5"   # GradVac
    ]
    legend = ["Normy gradientów w zależności od epoki (Zrównoważone wagi)",
              "Normy gradientów w zależności od epoki (GradNorm)",
              "Normy gradientów w zależności od epoki (DB_MTL)",
              "Normy gradientów w zależności od epoki (GradVac)"]

    dbmtl_other_tasks = [
        "camarotheboss/VSRVC/nsljta4h",
        "camarotheboss/VSRVC/4xjdo1eg",
        "camarotheboss/VSRVC/mejefr7f",
        "camarotheboss/VSRVC/0dgg1ptu",
        "camarotheboss/VSRVC/rycj71j6"
    ]
    legend = ["VSRVC 128", "VSRVC 256", "VSRVC 384", "VSRVC 512", "VSRVC 640"]

    basic_shallow = [
        "camarotheboss/VSRVC/mwhyusox",
        "camarotheboss/VSRVC/8ooyd85m",
        "camarotheboss/VSRVC/2trdx646",
        "camarotheboss/VSRVC/regfhmze",
        "camarotheboss/VSRVC/wzctn0vp",
    ]
    legend = ["VSRVCv2 osobne kompensatory 128", "-.- 256", "-.- 384", "-.- 512", "-.- 640"]

    basic_shallow = [
        "camarotheboss/VSRVC/mwhyusox",
        "camarotheboss/VSRVC/8ooyd85m",
        "camarotheboss/VSRVC/2trdx646",
        "camarotheboss/VSRVC/regfhmze",
        "camarotheboss/VSRVC/wzctn0vp",
    ]
    legend = ["VSRVCv2 128", "-.- 256", "-.- 384", "-.- 512", "-.- 640"]

    basic = [
        "camarotheboss/VSRVC/q2ouchdu",
        "camarotheboss/VSRVC/n3xzo6rt",
        "camarotheboss/VSRVC/cirbpnuz",
        "camarotheboss/VSRVC/v9m3601j",
        "camarotheboss/VSRVC/8g07e560",
    ]
    legend = ["VSRVCv2 EW 128", "-.- 256", "-.- 384", "-.- 512", "-.- 640"]

    basic_dbmtl = [
        "camarotheboss/VSRVC/x938xjpt",
        "camarotheboss/VSRVC/rp5kig3n",
        "camarotheboss/VSRVC/8mijerqv",
        "camarotheboss/VSRVC/t2xvkluh",
        "camarotheboss/VSRVC/gcxvkoml",
    ]
   # legend = ["VSRVCv2 DB_MTL 128", "-.- 256", "-.- 384", "-.- 512", "-.- 640"]

    basic_gradvac = [
        "camarotheboss/VSRVC/yo8sy41o",
        "camarotheboss/VSRVC/47v49zqz",
        "camarotheboss/VSRVC/6m6d0io1",
        "camarotheboss/VSRVC/sfbzqnz6",
        "camarotheboss/VSRVC/knt2hrhr",
    ]
    #legend = ["VSRVCv2 GradVac 128", "-.- 256", "-.- 384", "-.- 512", "-.- 640"]

    basic_gradnorm = [
        "camarotheboss/VSRVC/y72b36sr",
        "camarotheboss/VSRVC/n9vfz1jk",
        "camarotheboss/VSRVC/0kw8ixj0",
        "camarotheboss/VSRVC/u3r6xm65",
        "camarotheboss/VSRVC/dssrleft"
    ]
    #legend = ["VSRVCv2 GradNorm 128", "-.- 256", "-.- 384", "-.- 512", "-.- 640"]
    # plot_gradnorms(shallow_algorithms, legend=legend)
    mix = [
        ["camarotheboss/VSRVC/q2ouchdu",
         "camarotheboss/VSRVC/x938xjpt",
         "camarotheboss/VSRVC/yo8sy41o",
         "camarotheboss/VSRVC/y72b36sr",],
        ["EW",
         "DB_MTL",
         "GradVac",
         "GradNorm"]
    ]
    plot_grad_stats(mix[0], mode="per epoch", legend=mix[1])
    # plot_loss(dbmtl_other_tasks, legend, metric_keys, xy_labels)

    names = ["Beauty", "Bosphorus", "HoneyBee", "Jockey", "ReadySteadyGo", "ShakeNDry", "YachtRide"]
    roots = [
        r"D:\Code\ENVS\PyTorchVideoCompression\FVC\woutputs",
        r"D:\Code\DCVC\DCVC-FM\out_bin\UVG\rate_4",
        r"..\weights\VC basic\384",
        r"..\weights\VSRVC basic\256",
        r"..\..\VSRVC\evaluation\new_outputs",
    ]
    legend = ["FVC", "DCVC-FM", "VC 384", "VSRVC 256", "HEVC"]
    # roots = [
    #     r"..\weights\VC shallow\128",
    #     r"..\weights\VC shallow\256",
    #     r"..\weights\VC shallow\384",
    #     r"..\weights\VC shallow\512",
    #     r"..\weights\VC shallow\640",
    # ]
    # legend = ["VSRVC 128", "VSRVC 256", "VSRVC 384", "VSRVC 512", "VSRVC 640"]

    example = 3
    name = names[example]
    image_roots = [os.path.join(root, name) for root in roots]
    # mosaic("vc", image_roots, example, save_root="../weights", box=(250, 40, 64, 64), frame_idx=10, legend=legend)
    # get_stats_for_frame([
    #     "../weights/VSRVC basic/256/eval 128 12.json",
    #     "../weights/VC basic/384/eval 128 12.json",
    #     "../weights/DCVC-FM_rate_4.json",
    #     "../weights/fvc-8192.json",
    #     "db_veryslow_uvg.json",
    # ], example, 10)
    #
    example = 0
    name = names[example]
    roots = [
        r"D:\Code\ENVS\BasicVSR_PlusPlus\outputs\vimeo90k_bd",
        r"D:\Code\ENVS\IART\results\vimeo90k_BDx4_UVG",
        r"..\weights\VSR basic\128",
        r"..\weights\VSRVC basic\256",
        r"D:\Code\Datasets\UVG_bilinear",
    ]
    legend = ["BASICVSR", "IART", "VSR", "VSRVC 256", "BILINEAR"]
    image_roots = [os.path.join(root, name + ("_0" if i < 2 else "")) for i, root in enumerate(roots) ]
    #mosaic("vsr", image_roots, example, save_root="../weights", box=(900, 600, 192, 192), frame_idx=10, legend=legend)
    # get_stats_for_frame([
    #     "../weights/VSRVC basic/256/eval 128 12.json",
    #     "../weights/VSR basic/128/eval 128 12.json",
    #     "../weights/basicvsr_plusplus_trained.json",
    #     "../weights/iart_bd.json",
    #     "db_veryslow_uvg.json",
    # ], example, 10)
