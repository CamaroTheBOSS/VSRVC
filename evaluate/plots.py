import json

import numpy as np
from matplotlib import pyplot as plt


def load_eval_file(eval_file):
    with open(eval_file) as f:
        data = json.load(f)
    for key, value in data.items():
        data[key] = np.array(value)
    return data


def load_alg_database(database_file, algorithm):
    supported_algorithms = ["bilinear", "hevc", "avc"]
    if algorithm not in supported_algorithms:
        raise ValueError(f"Unrecognized algorithm value. Supported are {supported_algorithms}")
    with open(database_file) as f:
        data = json.load(f)
    for key, value in data[algorithm].items():
        data[algorithm][key] = np.array(value)

    return data[algorithm]


def _validate_plot_mode(mode):
    supported_modes = ["normal", "per frame"]
    if mode not in supported_modes:
        raise ValueError(f"Unrecognized mode value. Supported are {supported_modes}")
    return True


def _validate_metric_type(metric):
    supported_metrics = ["psnr", "ssim"]
    if metric not in supported_metrics:
        raise ValueError(f"Unrecognized metric type. Supported are {supported_metrics}")
    return True


def _plot_vc_3d(data, metric, fig, mode, linestyle, color):
    _validate_plot_mode(mode)
    if fig is None:
        fig = plt.figure()
    metric = "vc_psnr" if metric == "psnr" else "vc_ssim"
    x = data["bpp"].mean(axis=(1, 2))
    y = data[metric].mean(axis=(1, 2) if mode == "normal" else (0, 1))
    if mode == "normal":
        plt.plot(x, y, linestyle=linestyle, color=color)
    elif mode == "per frame":
        plt.plot(y, linestyle=linestyle, color=color)
    return fig, {metric: y, "bpp": x}


def _plot_vc_2d(data, metric, fig, mode, linestyle, color):
    if fig is None:
        fig = plt.figure()
    metric = "vc_psnr" if metric == "psnr" else "vc_ssim"
    x = np.sum(data["bpp"], axis=-1).mean()
    y = data[metric].mean(axis=None if mode == "normal" else 0)
    if mode == "normal":
        plt.scatter(x, y, linestyle=linestyle, color=color)
    elif mode == "per frame":
        plt.plot(y, linestyle=linestyle, color=color)
    return fig, {metric: y, "bpp": x}


def plot_vc(data, metric="psnr", fig=None, mode="normal", linestyle="-", color="#000000"):
    _validate_plot_mode(mode)
    _validate_metric_type(metric)
    kwargs = dict(fig=fig, mode=mode, linestyle=linestyle, color=color, metric=metric)
    return _plot_vc_2d(data, **kwargs) if len(data["vc_psnr"].shape) == 2 else _plot_vc_3d(data, **kwargs)


def get_vsr(data):
    quality = {"vsr_psnr": data["vsr_psnr"].mean(), "vsr_ssim": data["vsr_ssim"].mean()}
    return quality


if __name__ == "__main__":
    eval_data = load_eval_file("../weights/isric 1024/eval.json")
    hevc_data = load_alg_database("database.json", "hevc")
    avc_data = load_alg_database("database.json", "avc")
    bilinear_data = load_alg_database("database.json", "bilinear")
    fig, _ = plot_vc(eval_data, mode="normal", metric="ssim")
    plot_vc(hevc_data, fig=fig, mode="normal", metric="ssim")
    plot_vc(avc_data, fig=fig, mode="normal", metric="ssim")
    print(get_vsr(eval_data))
    print(get_vsr(bilinear_data))
    plt.show()
