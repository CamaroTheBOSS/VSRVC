import cv2
import numpy as np
import torch


def show_frame(frame: torch.Tensor):
    indexes = (0,) * (len(frame.shape) - 3) + (slice(None),)
    cv2_frame = np.clip(frame[indexes].detach().cpu().permute(1, 2, 0).numpy() * 255., 0, 255).astype(np.uint8)
    cv2.imshow("", cv2.cvtColor(cv2_frame, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
