import os
from glob import glob
from typing import Tuple

import torch
from PIL import Image

from kornia.augmentation import Resize
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor


class Vimeo90k(Dataset):
    def __init__(self, root: str, scale: int, test_mode: bool = False, crop_size: Tuple[int, int] = (256, 256),
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

        assert os.path.exists(self.root)
        assert os.path.exists(self.txt_file)

    def load_paths(self):
        videos = []
        with open(self.txt_file, "r") as f:
            for suffix in f.readlines():
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
        lqs = Resize((int(h / self.scale), int(w / self.scale)))(hqs).unsqueeze(0)

        return lqs, {"vc": lqs, "vsr": hqs.unsqueeze(0)}

    def get_item_with_name(self, name: str):
        index = self.get_index_with_name(name)
        if index is None:
            return None, None
        return self.__getitem__(index)

    def get_name_with_index(self, index: int):
        return  self.videos[index][0].split("\\")[-2]

    def get_index_with_name(self, name: str):
        for i, vid in enumerate(self.videos):
            if name in vid[0].split("\\")[-2]:
                return i
        return None

    def __len__(self) -> int:
        return len(self.videos)
