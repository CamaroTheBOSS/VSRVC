import os
from glob import glob
from typing import Tuple

import torch
from PIL import Image

from kornia.augmentation import ColorJiggle, RandomCrop, RandomVerticalFlip, RandomHorizontalFlip, Resize
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor


class Vimeo90k(Dataset):
    def __init__(self, root: str, scale: int, test_mode: bool = False, crop_size: Tuple[int, int] = (256, 384),
                 sliding_window_size: int = 0):
        super().__init__()
        self.root = root
        self.sequences = os.path.join(root, "train")
        self.txt_file = os.path.join(root, "sep_testlist.txt" if test_mode else "sep_trainlist.txt")

        self.scale = scale
        self.crop_size = crop_size
        self.test_mode = test_mode
        self.sliding_window_size = sliding_window_size if 0 < sliding_window_size < 7 else 7
        self.videos = self.load_paths()
        self.transform = Compose([ToTensor()])
        self.augmentation = Compose([
            ColorJiggle(brightness=(0.85, 1.15), contrast=(0.75, 1.15), saturation=(0.75, 1.25), hue=(-0.02, 0.02),
                        same_on_batch=True, p=1),
            RandomCrop(size=crop_size, same_on_batch=True),
            RandomVerticalFlip(same_on_batch=True, p=0.5),
            RandomHorizontalFlip(same_on_batch=True, p=0.5),
        ])
        if self.test_mode:
            self.augmentation = self.aug
        self.resize = Resize((int(crop_size[0] / self.scale), int(crop_size[1] / self.scale)))

        assert os.path.exists(self.root)
        assert os.path.exists(self.txt_file)

    def aug(self, x):
        return x

    def load_paths(self):
        videos = []
        with open(self.txt_file, "r") as f:
            # c = 0
            for suffix in f.readlines():
                # c += 1
                # if c == 200:
                #     break
                frame_paths = glob(os.path.join(self.sequences, suffix.strip(), "*.png"))
                for i in range(7 - self.sliding_window_size + 1):
                    videos.append([path for path in frame_paths[i:i + self.sliding_window_size]])
        return videos

    def read_video(self, index):
        video = []
        for path in self.videos[index]:
            video.append(self.transform(Image.open(path).convert("RGB")))
        return torch.stack(video)

    def __getitem__(self, index: int):
        video = self.read_video(index)
        return video, {"vc": torch.tensor(0), "vsr": torch.tensor(0)}

    def __len__(self) -> int:
        return len(self.videos)


class UVGDataset(Dataset):
    def __init__(self, root: str, scale: int, max_frames: int = 100, crop_size: Tuple[int, int] = (1024, 1980)):
        super().__init__()
        self.root = root

        self.scale = scale
        self.max_frames = max_frames
        self.crop_size = crop_size
        self.videos = self.load_paths()
        self.transform = Compose([ToTensor()])
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        assert os.path.exists(self.root)

    def load_paths(self):
        videos = []
        for video in os.listdir(self.root):
            frame_paths = glob(os.path.join(self.root, video, "*.png"))
            frame_paths = frame_paths[:min(len(frame_paths), self.max_frames)]
            videos.append([path for path in frame_paths])
        return videos

    def read_video(self, index):
        video = []
        for path in self.videos[index]:
            video.append(self.transform(Image.open(path).convert("RGB")))
        return torch.stack(video).to(self.device)

    def __getitem__(self, index: int):
        video = self.read_video(index)
        hqs = video[:, :, :self.crop_size[0], :self.crop_size[1]]
        n, c, h, w = hqs.shape
        lqs = Resize((int(h / self.scale), int(w / self.scale)))(hqs)

        return lqs, {"vc": lqs, "vsr": hqs}

    def getitem_with_name(self, name: str):
        index = self.get_index_with_name(name)
        if index is None:
            return None, None
        return self.__getitem__(index)

    def get_index_with_name(self, name: str):
        for i, vid in enumerate(self.videos):
            if name in vid[0].split("\\")[-2]:
                return i
        return None

    def __len__(self) -> int:
        return len(self.videos)
