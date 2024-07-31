import torch
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
        _, N, _, H, W = vc_decoded_video.size()
        bpp_const = 8.0 / (H * W)
        for i in range(len(vc_compress_data)):
            vc_pred = vc_decoded_video[:, i]
            vsr_pred = vsr_sr_video[:, i]

            vc_psnr_value = torch.clamp(psnr(vc_pred, vc_gt[:, i]), 0, 255).item()
            vc_ssim_value = torch.clamp(ssim(vc_pred, vc_gt[:, i]), 0, 1).item()
            bpp_values = []
            for j in range(len(vc_compress_data[i])):
                compress_data_pred = vc_compress_data[i][j][:-1]
                bpp_values += [sum(len(s) for s in cd) * bpp_const for cd in compress_data_pred]

            vsr_psnr_value = torch.clamp(psnr(vsr_pred, vsr_gt[:, i]), 0, 255).item()
            vsr_ssim_value = torch.clamp(ssim(vsr_pred, vsr_gt[:, i]), 0, 1).item()

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
