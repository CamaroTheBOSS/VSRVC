import json

import numpy as np
from matplotlib import pyplot as plt


def get_plot_colors():
    return ["#56B4E9", "#0072B2", "#009E73", "#000000", "#ad3bff", "#4b8a8c", "#9a6429", "#d07dce", "#ff7b5a"]


def get_plot_linestyles():
    return ["-", "--", "-", "--", "-", "--", "-", "--", "-"]


def load_eval_file(eval_file):
    with open(eval_file) as f:
         data = json.load(f)
    for key, value in data.items():
        if not key == "meta":
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


def _validate_task(metric):
    supported_tasks = ["vc", "vsr"]
    if metric not in supported_tasks:
        raise ValueError(f"Unrecognized task. Supported are {supported_tasks}")
    return True


def _plot_vc_3d(data, metric, fig, mode, linestyle, color):
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


def plot_vc_multiple(eval_files, database_path, linestyles=None, colors=None, legend=None, mode="normal", metric="psnr"):
    if linestyles is None:
        linestyles = get_plot_linestyles()[:len(eval_files)]
    if colors is None:
        colors = get_plot_colors()[:len(eval_files)]
    hevc_data = load_alg_database(database_path, "hevc")
    avc_data = load_alg_database(database_path, "avc")
    fig, _ = plot_vc(avc_data, mode=mode, metric=metric, linestyle="-", color="#E69F00")
    plot_vc(hevc_data, fig=fig, mode=mode, metric=metric, linestyle="-", color="#D55E00")
    for i, eval_file in enumerate(eval_files):
        eval_data = load_eval_file(eval_file)
        plot_vc(eval_data, fig=fig, mode=mode, metric=metric, linestyle=linestyles[i], color=colors[i])
    if legend is not None:
        legend = ["AVC", "HEVC"] + legend
        plt.legend(legend)
    return fig


def get_vsr(data):
    quality = {"vsr_psnr": data["vsr_psnr"].mean(), "vsr_ssim": data["vsr_ssim"].mean()}
    return quality


def get_multiple_vsr(eval_files, db_file):
    bilinear = load_alg_database(db_file, "bilinear")
    vsr_datas = [get_vsr(bilinear)]
    for eval_file in eval_files:
        eval_data = load_eval_file(eval_file)
        vsr_datas.append(get_vsr(eval_data))
    return vsr_datas


if __name__ == "__main__":
    database = "./db_veryslow_uvg.json"
    eval_files = [
        "../weights//VSRVC mv 128 EW x4 vimeo/eval 128 12 adapt.json",
        "../weights//VSRVC mv 128 GradNorm x4 vimeo/eval 128 12 adapt.json",
        "../weights//ISRIC 128 EW x4 vimeo/eval.json",
    ]
    legend = [
        "VSRVC vimeo λp=128, λi=128, 12 Equal Weighting",
        "VSRVC vimeo λp=128, λi=128, 12 GradNorm",
        "ISRIC vimeo λi=128",
        ]
    plot_vc_multiple(eval_files, database)
    plt.legend(["AVC", "HEVC"] + legend)
    plt.show()
    for i, data in enumerate(get_multiple_vsr(eval_files, database)):
        if i == 0:
            print(f"{'bilinear:':<39}{data}")
        else:
            print(f"{legend[i-1] + ':':<38} {data}")
