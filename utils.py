import os
import shutil
import cv2
import numpy as np
import torch


def to_cv2(img):
    return cv2.cvtColor(np.clip(img.detach().permute(1, 2, 0).cpu().numpy() * 255., 0, 255)
                        .astype(np.uint8), cv2.COLOR_RGB2BGR)


def show_frame(frame: torch.Tensor):
    indexes = (0,) * (len(frame.shape) - 3) + (slice(None),)
    cv2_frame = to_cv2(frame[indexes])
    cv2.imshow("", cv2_frame)
    cv2.waitKey(0)


def save_video(video: torch.Tensor, root: str, name: str = "vid") -> None:
    path = os.path.join(root, name)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    if len(video.size()) == 5:
        video = video[0]

    for i, frame in enumerate(video):
        numpy_frame = np.clip(frame.detach().permute(1, 2, 0).cpu().numpy() * 255., 0, 255).astype(np.uint8)
        cv_frame = cv2.cvtColor(numpy_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{path}/img{(i + 1):03d}.png", cv_frame)
