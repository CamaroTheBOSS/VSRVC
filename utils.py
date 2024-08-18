import os
import shutil
from typing import Tuple
from PIL import Image

import cv2
import numpy as np
import torch
from torch.nn.functional import interpolate


def confirm_action(prompt="Do you want to continue? (y/n): "):
    while True:
        user_input = input(prompt).strip().lower()
        if user_input == 'y':
            print("Continuing...")
            return True
        elif user_input == 'n':
            print("Action canceled.")
            return False
        else:
            print("Invalid input. Please enter 'y' or 'n'.")


def to_cv2(img):
    return cv2.cvtColor(np.clip(img.detach().permute(1, 2, 0).cpu().numpy() * 255., 0, 255)
                        .astype(np.uint8), cv2.COLOR_RGB2BGR)


def show_frame(frame: torch.Tensor):
    indexes = (0,) * (len(frame.shape) - 3) + (slice(None),)
    cv2_frame = to_cv2(frame[indexes])
    cv2.imshow("", cv2_frame)
    cv2.waitKey(0)


def save_frame(filepath: str, frame: torch.Tensor):
    indexes = (0,) * (len(frame.shape) - 3) + (slice(None),)
    saved_frame = np.clip(frame[indexes].detach().cpu().permute(1, 2, 0).numpy() * 255., 0, 255).astype(np.uint8)
    saved_format = filepath[-3:]

    if saved_format == "png":
        cv2.imwrite(filepath, cv2.cvtColor(saved_frame, cv2.COLOR_RGB2BGR))
        return
    elif saved_format == "jpg":
        saved_frame = Image.fromarray(saved_frame)
        saved_frame.save(filepath, format="JPEG", quality=95)


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


def interpolate_frame(frame: torch.Tensor, size: Tuple[int, int], mode="bilinear"):
    if len(frame.shape) == 3:
        frame = frame.unsqueeze(0)
    interpolated = interpolate(frame, size=size, mode=mode, align_corners=False)
    return interpolated
