import torch
from scipy.interpolate import PchipInterpolator
import numpy as np

from LibMTL.loss import AbsLoss
from LibMTL.metrics import AbsMetric
from torch import nn, Tensor
from torchmetrics.functional.image import structural_similarity_index_measure, peak_signal_noise_ratio


class RateDistortionLoss(AbsLoss):
    def __init__(self, lmbda: int):
        super(RateDistortionLoss, self).__init__()
        self.distortion = nn.L1Loss()
        self.lmbda = lmbda

    def compute_loss(self, pred, gt):
        reconstruction, bits = pred
        B, _, H, W = reconstruction.shape
        num_pixels = B * H * W
        bpp_loss = sum(bits) / num_pixels
        return_value = self.lmbda * self.distortion(reconstruction, gt) + bpp_loss
        return return_value

    def _update_loss(self, pred, gt):
        loss = self.compute_loss(pred, gt)
        self.record.append(loss.item())
        self.bs.append(pred[0].size()[0])
        return loss

    def _reinit(self):
        self.record = []
        self.bs = []


class VSRLoss(AbsLoss):
    def __init__(self, lmbda: int):
        super(VSRLoss, self).__init__()
        self.distortion = nn.L1Loss()
        self.lmbda = lmbda

    def compute_loss(self, pred, gt):
        return_value = self.lmbda * self.distortion(pred, gt)
        return return_value


class DummyLoss(AbsLoss):
    def __init__(self):
        super(DummyLoss, self).__init__()

    def compute_loss(self, pred, gt):
        return torch.tensor(0)


class DummyMetrics(AbsMetric):
    def __init__(self):
        super(DummyMetrics, self).__init__()

    def update_fun(self, pred, gt):
        self.bs.append(pred.size()[0])

    def score_fun(self):
        return []

    def reinit(self):
        self.bs = []


class QualityMetrics(AbsMetric):
    def __init__(self):
        super(QualityMetrics, self).__init__()
        self.psnr_record = []
        self.ssim_record = []

    def update_fun(self, pred, gt):
        self.psnr_record.append(torch.clamp(psnr(pred, gt), 0, 255).item())
        self.ssim_record.append(torch.clamp(ssim(pred, gt), 0, 1).item())
        self.bs.append(pred.size()[0])

    def score_fun(self):
        records = torch.stack([torch.tensor(self.psnr_record), torch.tensor(self.ssim_record)])
        batch_size = torch.tensor(self.bs)
        return [(records[i] * batch_size).sum() / (torch.sum(batch_size)) for i in range(2)]

    def reinit(self):
        self.psnr_record = []
        self.ssim_record = []
        self.bs = []


class CompressionTaskMetrics(AbsMetric):
    def __init__(self):
        super(CompressionTaskMetrics, self).__init__()
        self.quality_metrics = QualityMetrics()
        self.bpp_record = []

    def update_fun(self, pred, gt):
        reconstruction, bits = pred
        B, _, H, W = reconstruction.shape
        num_pixels = B * H * W
        bpp = sum(bits) / num_pixels
        self.bpp_record.append(bpp)
        self.quality_metrics.update_fun(reconstruction, gt)
        self.bs.append(reconstruction.size()[0])

    def score_fun(self):
        metrics = self.quality_metrics.score_fun()
        bpp_records = torch.tensor(self.bpp_record)
        metrics.extend([bpp_records.mean()])
        return metrics

    def reinit(self):
        self.quality_metrics.reinit()
        self.bpp_record = []
        self.bs = []


class UVGMetrics:
    def __init__(self):
        super(UVGMetrics, self).__init__()
        self.vc_psnr_record = []
        self.vc_ssim_record = []
        self.vsr_psnr_record = []
        self.vsr_ssim_record = []
        self.bpp_record = []

    def update(self, vsr_sr_video, vc_compress_data, vc_decoded_video, gt):
        vc_gt = gt["vc"]
        vsr_gt = gt["vsr"]
        _, _, _, H, W = vc_decoded_video.size() if vc_decoded_video is not None else (1, 1, 1, 1, 1)
        bpp_const = 8.0 / (H * W)
        length = vsr_sr_video.size()[1] if vsr_sr_video is not None else vc_decoded_video.size()[1]
        for i in range(length):
            if vc_decoded_video is not None:
                vc_pred = vc_decoded_video[:, i]
                vc_psnr_value = torch.clamp(psnr(vc_pred, vc_gt[:, i]), 0, 255).item()
                vc_ssim_value = torch.clamp(ssim(vc_pred, vc_gt[:, i]), 0, 1).item()
                bpp_values = []
                for j in range(len(vc_compress_data[i])):
                    compress_data_pred = vc_compress_data[i][j][:-1]
                    bpp_values += [len(s[0]) * bpp_const for s in compress_data_pred]
            else:
                vc_psnr_value = 0.
                vc_ssim_value = 0.
                bpp_values = [0., 0., 0., 0.]

            if vsr_sr_video is not None:
                vsr_pred = vsr_sr_video[:, i]
                vsr_psnr_value = torch.clamp(psnr(vsr_pred, vsr_gt[:, i]), 0, 255).item()
                vsr_ssim_value = torch.clamp(ssim(vsr_pred, vsr_gt[:, i]), 0, 1).item()
            else:
                vsr_psnr_value = 0.
                vsr_ssim_value = 0.

            self.vc_psnr_record.append(vc_psnr_value)
            self.vc_ssim_record.append(vc_ssim_value)
            self.vsr_psnr_record.append(vsr_psnr_value)
            self.vsr_ssim_record.append(vsr_ssim_value)
            self.bpp_record.append(bpp_values)

    def get_records_dict(self):
        return {
            "vc_psnr": self.vc_psnr_record,
            "vc_ssim": self.vc_ssim_record,
            "vsr_psnr": self.vsr_psnr_record,
            "vsr_ssim": self.vsr_ssim_record,
            "bpp": self.bpp_record,
        }

    def reinit(self):
        self.vc_psnr_record = []
        self.vc_ssim_record = []
        self.vsr_psnr_record = []
        self.vsr_ssim_record = []
        self.bpp_record = []


def psnr(input_images: Tensor, target_images: Tensor, aggregate: str = "mean") -> Tensor:
    if aggregate not in ["mean", "sum", "none"]:
        raise NotImplementedError(f"Unrecognized aggregation strategy \"{aggregate}\". "
                                  f"Possible aggregation strategies: [mean, sum, none].")
    if len(input_images.shape) == 4:
        return _psnr_images(input_images, target_images, aggregate=aggregate)
    elif len(input_images.shape) == 5:
        return _psnr_videos(input_images, target_images, aggregate=aggregate)

    raise NotImplementedError("Input tensors should be 4D or 5D")


def ssim(input_images: Tensor, target_images: Tensor, aggregate: str = "mean") -> Tensor:
    if aggregate not in ["mean", "sum", "none"]:
        raise NotImplementedError(f"Unrecognized aggregation strategy \"{aggregate}\". "
                                  f"Possible aggregation strategies: [mean, sum, none].")
    if len(input_images.shape) == 4:
        return _ssim_images(input_images, target_images, aggregate=aggregate)
    elif len(input_images.shape) == 5:
        return _ssim_videos(input_images, target_images, aggregate=aggregate)

    raise NotImplementedError("Input tensors should be 4D or 5D")


def _psnr_images(input_images: Tensor, target_images: Tensor, aggregate: str = "mean") -> Tensor:
    """
    Function, which calculates PSNR for batch of images
    :param input_images: Tensor with shape B,C,H,W low resolution image
    :param target_images: Tensor with shape B,C,H,W high resolution image
    :return: sum of PSNR values in batch
    """
    if not len(input_images.shape) == 4:
        raise NotImplementedError("Input tensors should have 4D shape B,C,H,W")

    psnr_values = []
    for i in range(target_images.size()[0]):
        psnr_values.append(peak_signal_noise_ratio(input_images[i], target_images[i], 1.0))
    if aggregate == "mean":
        return torch.mean(torch.stack(psnr_values))
    if aggregate == "sum":
        return torch.sum(torch.stack(psnr_values))
    if aggregate == "none":
        return torch.stack(psnr_values)
    raise NotImplementedError(f"Unrecognized aggregation strategy \"{aggregate}\". "
                              f"Possible aggregation strategies: [mean, sum, none].")


def _psnr_videos(input_videos: Tensor, target_videos: Tensor, aggregate: str = "mean") -> Tensor:
    """
    Function, which calculates PSNR for batch of videos
    :param input_videos: Tensor with shape B,N,C,H,W low resolution video
    :param target_videos: Tensor with shape B,N,C,H,W high resolution video
    :return: sum of PSNR values in batch
    """
    if not len(input_videos.shape) == 5:
        raise NotImplementedError("Input tensors should have 5D shape B,N,C,H,W")

    psnr_values = []
    for i in range(target_videos.size()[0]):
        psnr_values.append(peak_signal_noise_ratio(input_videos[i], target_videos[i], 1.0))
    if aggregate == "mean":
        return torch.mean(torch.stack(psnr_values))
    if aggregate == "sum":
        return torch.sum(torch.stack(psnr_values))
    if aggregate == "none":
        return torch.stack(psnr_values)
    raise NotImplementedError(f"Unrecognized aggregation strategy \"{aggregate}\". "
                              f"Possible aggregation strategies: [mean, sum, none].")


def _ssim_images(input_images: Tensor, target_images: Tensor, aggregate: str = "mean") -> Tensor:
    """
       Function, which calculates SSIM for batch of images
       :param input_images: Tensor with shape B,C,H,W low resolution image
       :param target_images: Tensor with shape B,C,H,W high resolution image
       :return: sum of SSIM values in batch
       """
    if not len(input_images.shape) == 4:
        raise NotImplementedError("Input tensors should have 4D shape B,C,H,W")

    ssim_values = []
    for i in range(target_images.size()[0]):
        ssim_values.append(
            structural_similarity_index_measure(input_images[i].unsqueeze(0), target_images[i].unsqueeze(0)))
    if aggregate == "mean":
        return torch.mean(torch.stack(ssim_values))
    if aggregate == "sum":
        return torch.sum(torch.stack(ssim_values))
    if aggregate == "none":
        return torch.stack(ssim_values)
    raise NotImplementedError(f"Unrecognized aggregation strategy \"{aggregate}\". "
                              f"Possible aggregation strategies: [mean, sum, none].")


def _ssim_videos(input_videos: Tensor, target_videos: Tensor, aggregate: str = "mean") -> Tensor:
    """
       Function, which calculates SSIM for batch of images
       :param input_videos: Tensor with shape B,N,C,H,W low resolution video
       :param target_videos: Tensor with shape B,N,C,H,W high resolution video
       :return: sum of SSIM values in batch
       """
    if not len(input_videos.shape) == 5:
        raise NotImplementedError("Input tensors should have 5D shape B,N,C,H,W")

    ssim_values = []
    for i in range(target_videos.size()[0]):
        ssim_values.append(structural_similarity_index_measure(input_videos[i], target_videos[i]))
    if aggregate == "mean":
        return torch.mean(torch.stack(ssim_values))
    if aggregate == "sum":
        return torch.sum(torch.stack(ssim_values))
    if aggregate == "none":
        return torch.stack(ssim_values)
    raise NotImplementedError(f"Unrecognized aggregation strategy \"{aggregate}\". "
                              f"Possible aggregation strategies: [mean, sum, none].")

def BD_PSNR(R1, PSNR1, R2, PSNR2, piecewise=0):
    lR1 = np.log(R1)
    lR2 = np.log(R2)

    PSNR1 = np.array(PSNR1)
    PSNR2 = np.array(PSNR2)

    p1 = np.polyfit(lR1, PSNR1, 3)
    p2 = np.polyfit(lR2, PSNR2, 3)

    # integration interval
    min_int = max(min(lR1), min(lR2))
    max_int = min(max(lR1), max(lR2))

    # find integral
    if piecewise == 0:
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        # See https://chromium.googlesource.com/webm/contributor-guide/+/master/scripts/visual_metrics.py
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        f1 = PchipInterpolator(np.sort(lR1), PSNR1[np.argsort(lR1)])
        f2 = PchipInterpolator(np.sort(lR2), PSNR2[np.argsort(lR2)])
        v1 = f1(samples)
        v2 = f2(samples)
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)

    # find avg diff
    avg_diff = (int2-int1)/(max_int-min_int)

    return avg_diff


def BD_RATE(R1, PSNR1, R2, PSNR2, piecewise=0):
    lR1 = np.log(R1)
    lR2 = np.log(R2)

    # rate method
    p1 = np.polyfit(PSNR1, lR1, 3)
    p2 = np.polyfit(PSNR2, lR2, 3)

    # integration interval
    min_int = max(min(PSNR1), min(PSNR2))
    max_int = min(max(PSNR1), max(PSNR2))

    # find integral
    if piecewise == 0:
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        f1 = PchipInterpolator(np.sort(PSNR1), lR1[np.argsort(PSNR1)])
        f2 = PchipInterpolator(np.sort(PSNR2), lR1[np.argsort(PSNR2)])
        v1 = f1(samples)
        v2 = f2(samples)
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)

    # find avg diff
    avg_exp_diff = (int2-int1)/(max_int-min_int)
    avg_diff = (np.exp(avg_exp_diff)-1)*100
    return avg_diff
