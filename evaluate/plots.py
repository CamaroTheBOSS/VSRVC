import json
from typing import List, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from metrics import BD_PSNR, BD_RATE


class Series:
    def __init__(self, series: Union[str, List[str]], legend: str, vc: bool = True, vsr: bool = True):
        self.series = series
        self.legend = legend
        self.vc = vc
        self.vsr = vsr


def get_plot_colors():
    return ["#56B4E9", "#0072B2", "#009E73", "#000000", "#ad3bff", "#4b8a8c", "#9a6429", "#d07dce", "#ff7b5a",
            "#587246", "#7447ea", "#829811", "#796166", "#db98fd", "#681e06", "#67ab75", "#d07abb", "#887843"]


def get_plot_linestyles():
    return ["-", "-", "-", "-", "-", "-", "-", "--", "--", "--", "--", "--", "--", "--", "--", "--", "--", "--"]


def _load_eval_file_series(eval_files):
    datas = []
    for eval_file in eval_files:
        with open(eval_file) as f:
            data = json.load(f)
        for key, value in data.items():
            if not key == "meta":
                data[key] = np.array(value)
        datas.append(data)
    return datas


def load_eval_file(eval_file):
    if isinstance(eval_file, list):
        return _load_eval_file_series(eval_file)
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
    y = None
    if mode == "normal":
        y = data[metric].mean(axis=(1, 2))
        plt.plot(x, y, linestyle=linestyle, color=color)
    elif mode == "per frame":
        y = data[metric][10].mean(axis=0)
        plt.plot(y, linestyle=linestyle, color=color)
    return fig, {metric: y, "bpp": x}


def get_bpp_2d(eval_data: Union[list, dict]):
    if isinstance(eval_data, list):
        return [np.sum(d["bpp"], axis=-1).mean() for d in eval_data]
    return np.sum(eval_data["bpp"], axis=-1).mean()


def _plot_vc_2d(data, metric, fig, mode, linestyle, color):
    if fig is None:
        fig = plt.figure()
    metric = "vc_psnr" if metric == "psnr" else "vc_ssim"
    x = get_bpp_2d(data)
    y = data[metric].mean(axis=None if mode == "normal" else 0)
    if mode == "normal":
        plt.scatter(x, y, linestyle=linestyle, color=color)
    elif mode == "per frame":
        plt.plot(y, linestyle=linestyle, color=color)
    return fig, {metric: y, "bpp": x}


def _plot_vc_2d_series(data, metric, fig, mode, linestyle, color):
    if fig is None:
        fig = plt.figure()
    metric = "vc_psnr" if metric == "psnr" else "vc_ssim"
    x, y = [], []
    for d in data:
        x.append(get_bpp_2d(d))
        y.append(d[metric].mean(axis=None if mode == "normal" else 0))
    if mode == "normal":
        plt.plot(x, y, linestyle=linestyle, color=color, marker="o")
    elif mode == "per frame":
        plt.plot(y, linestyle=linestyle, color=color)
    return fig, {metric: y, "bpp": x}


def plot_vc(data, metric="psnr", fig=None, mode="normal", linestyle="-", color="#000000"):
    _validate_plot_mode(mode)
    _validate_metric_type(metric)
    kwargs = dict(fig=fig, mode=mode, linestyle=linestyle, color=color, metric=metric)
    if isinstance(data, list):
        return _plot_vc_2d_series(data, **kwargs)
    return _plot_vc_2d(data, **kwargs) if len(data["vc_psnr"].shape) == 2 else _plot_vc_3d(data, **kwargs)


def plot_vc_multiple(eval_files, database_path, linestyles=None, colors=None, legend=None, mode="normal",
                     metric="psnr"):
    if linestyles is None:
        linestyles = get_plot_linestyles()[:len(eval_files)]
    if colors is None:
        colors = get_plot_colors()[:len(eval_files)]
    hevc_data = load_alg_database(database_path, "hevc")
    avc_data = load_alg_database(database_path, "avc")
    fig, _ = plot_vc(avc_data, mode=mode, metric=metric, linestyle="-", color="#E69F00")
    plot_vc(hevc_data, fig=fig, mode=mode, metric=metric, linestyle="-", color="#D55E00")
    max_bpp = 0
    for i, eval_file in enumerate(eval_files):
        eval_data = load_eval_file(eval_file)
        bpp_2d = get_bpp_2d(eval_data)
        max_bpp = max(max_bpp, max(bpp_2d) if isinstance(bpp_2d, list) else bpp_2d)
        plot_vc(eval_data, fig=fig, mode=mode, metric=metric, linestyle=linestyles[i], color=colors[i])
    if legend is not None:
        legend = ["AVC", "HEVC"] + legend
        plt.legend(legend, loc="lower right")
    plt.ylabel(metric.upper() + (' [dB]' if metric.upper() == "PSNR" else ''))
    plt.xlabel("BPP" if mode == "normal" else "Indeks klatki")
    if mode == "normal":
        plt.xlim([0, 1.1 * max_bpp])
    return fig


def _get_vsr(data):
    quality = {"vsr_psnr": np.round(data["vsr_psnr"].mean(), 2), "vsr_ssim": np.round(data["vsr_ssim"].mean(), 2)}
    return quality


def get_vsr_from_files(eval_files):
    vsr_datas = []
    for eval_file in eval_files:
        eval_data = load_eval_file(eval_file)
        vsr_datas.append(_get_vsr(eval_data))
    return vsr_datas


def get_vsr_from_file(eval_file):
    eval_data = load_eval_file(eval_file)
    return _get_vsr(eval_data)


def get_multiple_vsr(eval_files, db_file):
    bilinear = load_alg_database(db_file, "bilinear")
    vsr_datas = [_get_vsr(bilinear)]
    for eval_file in eval_files:
        if isinstance(eval_file, list):
            vsr_datas.append(get_vsr_from_files(eval_file))
        else:
            vsr_datas.append(get_vsr_from_file(eval_file))
    return vsr_datas


def get_mean_vc_xy_2d(data, metric="psnr"):
    metric = "vc_psnr" if metric == "psnr" else "vc_ssim"
    x = np.round(np.sum(data["bpp"], axis=-1).mean(), 2)
    y = np.round(data[metric].mean(), 2)
    return x, y


def get_mean_vc_xy_3d(data, metric="psnr"):
    metric = "vc_psnr" if metric == "psnr" else "vc_ssim"
    x = np.flip(data["bpp"].mean(axis=(1, 2)))
    y = np.flip(data[metric].mean(axis=(1, 2)))
    return x, y


def _get_interpolation_vc_series(eval_files, hx, hy, ax, ay):
    interpolations = []
    for eval_file in eval_files:
        eval_data = load_eval_file(eval_file)
        ex, ey = get_mean_vc_xy_2d(eval_data, metric)
        hy_interpolated = np.round(np.interp(np.array([ex]), hx, hy), 2).item()
        ay_interpolated = np.round(np.interp(np.array([ex]), ax, ay), 2).item()
        interpolations.append({"bpp": ex, "model": ey, "hevc": hy_interpolated, "avc": ay_interpolated})
    return interpolations


def _get_interpolation_vc(eval_file, hx, hy, ax, ay):
    eval_data = load_eval_file(eval_file)
    ex, ey = get_mean_vc_xy_2d(eval_data, metric)
    hy_interpolated = np.round(np.interp(np.array([ex]), hx, hy), 2).item()
    ay_interpolated = np.round(np.interp(np.array([ex]), ax, ay), 2).item()
    return {"bpp": ex, "model": ey, "hevc": hy_interpolated, "avc": ay_interpolated}


def get_multiple_vc(eval_files, db_file, metric="psnr"):
    hevc_data = load_alg_database(db_file, "hevc")
    avc_data = load_alg_database(db_file, "avc")
    hx, hy = get_mean_vc_xy_3d(hevc_data, metric)
    ax, ay = get_mean_vc_xy_3d(avc_data, metric)
    vc_datas = []
    for eval_file in eval_files:
        if isinstance(eval_file, list):
            vc_datas.append(_get_interpolation_vc_series(eval_file, hx, hy, ax, ay))
        else:
            vc_datas.append(_get_interpolation_vc(eval_file, hx, hy, ax, ay))
    return vc_datas


def get_bd_metrics(tested_data, reference_data):
    tx, ty, refx, refy = [], [], [], []
    for tuple_data in [(tx, ty, tested_data), (refx, refy, reference_data)]:
        x, y, data = tuple_data
        if not isinstance(data, list):
            data = [{"bpp": data["bpp"], "model": data["model"] * 0.9}, data,
                   {"bpp": data["bpp"] * 1.1, "model": data["model"]}]
        for d in data:
            x.append(d["bpp"])
            y.append(d["model"])

    bd_rate = np.round(BD_RATE(refx, refy, tx, ty), 2)
    bd_psnr = np.round(BD_PSNR(refx, refy, tx, ty), 2)
    print(f"BD_RATE related to VSRVC: {bd_rate}")
    print(f"BD_PSNR related to VSRVC: {bd_psnr}")
    return bd_rate, bd_psnr


def get_bd_metrics_trad_algs(reference_data, alg):
    alg_data = load_alg_database("db_veryslow_uvg.json", alg)
    ax = alg_data["bpp"].mean(axis=(1, 2))
    ay = alg_data["vc_psnr"].mean(axis=(1, 2))

    x, y, data = [], [], reference_data
    if not isinstance(data, list):
        data = [{"bpp": data["bpp"], "model": data["model"] * 0.9}, data,
                {"bpp": data["bpp"] * 1.1, "model": data["model"]}]
    for d in data:
        x.append(d["bpp"])
        y.append(d["model"])

    bd_rate = np.round(BD_RATE(x, y, ax, ay), 2)
    bd_psnr = np.round(BD_PSNR(x, y, ax, ay), 2)
    print(f"BD_RATE related to VSRVC: {bd_rate}")
    print(f"BD_PSNR related to VSRVC: {bd_psnr}")
    return bd_rate, bd_psnr


def write_vc_latex(vc_datas, legend):
    bd_psnrs, bd_rates = [], []
    for vc_data in vc_datas:
        bdr, bdp = get_bd_metrics(vc_data, vc_datas[-1])
        bd_psnrs.append(bdp)
        bd_rates.append(bdr)
    bd_rate_avc, bd_psnr_avc = get_bd_metrics_trad_algs(vc_datas[-1], "avc")
    bd_rate_hevc, bd_psnr_hevc = get_bd_metrics_trad_algs(vc_datas[-1], "hevc")
    df = pd.DataFrame([bd_rates, bd_psnrs])
    df.columns = legend
    df2 = pd.DataFrame({'HEVC': [bd_rate_hevc, bd_psnr_hevc], 'AVC': [bd_rate_avc, bd_psnr_avc]})
    df = df2.join(df)
    df.index = ["BD-RATE", "BD-PSNR"]
    print(df)
    df.astype(str).to_latex("vc_table.txt", caption="placeholder", label="placeholder")


def write_vsr_latex(vsr_datas, legend):
    legend = ["Interpolacja dwuliniowa"] + legend
    psnrs, ssims = [], []
    for data in vsr_datas:
        if isinstance(data, list):
            max_psnr = 0
            relevant_data = None
            for d in data:
                if max_psnr < d["vsr_psnr"]:
                    max_psnr = d["vsr_psnr"]
                    relevant_data = d
        else:
            relevant_data = data
        psnrs.append(relevant_data["vsr_psnr"])
        ssims.append(relevant_data["vsr_ssim"])
    df = pd.DataFrame([psnrs, ssims])
    df.columns = legend
    df.index = ["PSNR", "SSIM"]
    print(df)
    df.astype(str).to_latex("vsr_table.txt", caption="placeholder", label="placeholder")


if __name__ == "__main__":
    database = "./db_veryslow_uvg.json"
    isric = Series([
        "../weights//ISRIC/128/eval.json",
        "../weights//ISRIC/256/eval 128 12.json",
        "../weights//ISRIC/384/eval 128 12.json",
        "../weights//ISRIC/512/eval 128 12.json",
        "../weights//ISRIC/640/eval 128 12.json",
    ], "ISRIC")

    shallow_series = Series([
        "../weights//VSRVC shallow/128/eval 128 12 adapt.json",
        "../weights//VSRVC shallow/256/eval 128 12.json",
        "../weights//VSRVC shallow/384/eval 128 12.json",
        "../weights//VSRVC shallow/512/eval 128 12.json",
        "../weights//VSRVC shallow/640/eval 128 12.json",
    ], "VSRVC")
    shallow_series_only_vc = Series([
        "../weights//VC shallow/128/eval 128 12.json",
        "../weights//VC shallow/256/eval 128 12.json",
        "../weights//VC shallow/384/eval 128 12.json",
        "../weights//VC shallow/512/eval 128 12.json",
        "../weights//VC shallow/640/eval 128 12.json",
    ], "VC", vsr=False)
    shallow_only_vsr = Series("../weights//VSR shallow/128/eval 128 12.json", "VSR", vc=False)
    shallow_ew = Series("../weights//VSRVC shallow/128/eval 128 12 adapt.json", "VSRVC EW")
    shallow_gradnorm = Series("../weights//VSRVC shallow/GradNorm 128/eval 128 12 adapt.json", "VSRVC GradNorm")
    shallow_dbmtl = Series("../weights//VSRVC shallow/DB_MTL 128/eval 128 12 adapt.json", "VSRVC DB_MTL")
    shallow_gradvac = Series("../weights//VSRVC shallow/GradVac 128/eval 128 12.json", "VSRVC GradVac")

    mv_series = Series([
        "../weights//VSRVC mv/128/eval 128 12 adapt.json",
        "../weights//VSRVC mv/256/eval 128 12.json",
        "../weights//VSRVC mv/384/eval 128 12.json",
        "../weights//VSRVC mv/512/eval 128 12.json",
        "../weights//VSRVC mv/640/eval 128 12.json",
    ], "VSRVCv2")
    mv_series_only_vc = Series([
        "../weights//VC mv/128/eval 128 12.json",
        "../weights//VC mv/256/eval 128 12.json",
        "../weights//VC mv/384/eval 128 12.json",
        "../weights//VC mv/512/eval 128 12.json",
        "../weights//VC mv/640/eval 128 12.json",
    ], "VCv2", vsr=False)
    mv_only_vsr = Series("../weights//VSR mv/128/eval 128 12.json", "VSRv2", vc=False)
    mv_ew = Series("../weights//VSRVC mv/128/eval 128 12 adapt.json", "VSRVCv2 EW")
    mv_gradnorm = Series("../weights//VSRVC mv/GradNorm 128/eval 128 12 adapt.json", "VSRVCv2 GradNorm")
    mv_dbmtl = Series("../weights//VSRVC mv/DB_MTL 128/eval 128 12 adapt.json", "VSRVCv2 DB_MTL")
    mv_gradvac = Series("../weights//VSRVC mv/GradVac 128/eval 128 12.json", "VSRVCv2 GradVac")

    basic_series = Series([
        "../weights//VSRVC basic/128/eval 128 12.json",
        # "../weights//VSRVC basic/256/eval 128 12.json",
        # "../weights//VSRVC basic/384/eval 128 12.json",
        # "../weights//VSRVC basic/512/eval 128 12.json",
        # "../weights//VSRVC basic/640/eval 128 12.json",
    ], "VSRVCv3")
    basic_series_only_vc = Series([
        # "../weights//VC basic/128/eval 128 12.json",
        # "../weights//VC basic/256/eval 128 12.json",
        # "../weights//VC basic/384/eval 128 12.json",
        # "../weights//VC basic/512/eval 128 12.json",
        # "../weights//VC basic/640/eval 128 12.json",
    ], "VCv3", vsr=False)
    # basic_only_vsr = Series("../weights//VSR basic/128/eval 128 12.json", "VSRv3", vc=False)
    basic_gradnorm = Series("../weights//VSRVC basic/GradNorm 128/eval 128 12.json", "VSRVCv3 GradNorm")
    basic_dbmtl = Series("../weights//VSRVC basic/DB_MTL 128/eval 128 12.json", "VSRVCv3 DB_MTL")
    basic_gradvac = Series("../weights//VSRVC basic/GradVac 128/eval 128 12.json", "VSRVCv3 GradVac")
    basic_pcgrad = Series("../weights//VSRVC basic/PCGrad 128/eval 128 12.json", "VSRVCv3 PCGrad")

    fvc_series = Series(["../weights//fvc-256.json",
                         "../weights//fvc-512.json",
                         "../weights//fvc-1024.json",
                         "../weights//fvc-2048.json",
                         "../weights//fvc-4096.json",
                         "../weights//fvc-8192.json", ],
                        "FVC", vsr=False)
    dcvc_fm_series = Series(["../weights//DCVC-FM_rate_0.json",
                             "../weights//DCVC-FM_rate_1.json",
                             "../weights//DCVC-FM_rate_2.json",
                             "../weights//DCVC-FM_rate_3.json",
                             "../weights//DCVC-FM_rate_4.json", ],
                            "DCVC-FM", vsr=False)
    basic_vsr_pp = Series("../weights//basicvsr_plusplus_trained.json", "BasicVSR++", vc=False)
    iart = Series("../weights//iart_bd.json", "IART", vc=False)

    # series = [fvc_series, dcvc_fm_series, shallow_series_only_vc, shallow_series]  # VC
    series = [basic_vsr_pp, iart, shallow_only_vsr, shallow_series] # VSR
    # series = [shallow_ew, shallow_gradnorm, shallow_dbmtl, shallow_gradvac]
    # series = [mv_ew, mv_gradnorm, mv_dbmtl, mv_gradvac]
    eval_files_vc = [s.series for s in series if s.vc]
    legend_vc = [s.legend for s in series if s.vc]
    eval_files_vsr = [s.series for s in series if s.vsr]
    legend_vsr = [s.legend for s in series if s.vsr]
    metric = "psnr"
    # plot_vc_multiple(eval_files_vc, database, metric="psnr", legend=legend_vc)
    # plot_vc_multiple(eval_files_vc, database, metric="ssim", legend=legend_vc)
    plt.show()
    vcs = get_multiple_vc(eval_files_vc, database, metric)
    write_vc_latex(vcs, legend_vc)
    for i, data in enumerate(vcs):
        print(f"{legend_vc[i] + ':':<38} {data}")

    print("===========================================================================")
    vsrs = get_multiple_vsr(eval_files_vsr, database)
    write_vsr_latex(vsrs, legend_vsr)
    for i, data in enumerate(vsrs):
        if i == 0:
            print(f"{'bilinear:':<39}{data}")
        else:
            print(f"{legend_vsr[i - 1] + ':':<38} {data}")
