import os
from glob import glob
from typing import Tuple

import torch
from PIL import Image

from kornia.augmentation import Resize, ColorJiggle, RandomCrop, RandomVerticalFlip, RandomHorizontalFlip
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor


class Vimeo90k(Dataset):
    def __init__(self, root: str, test_mode: bool = False, sliding_window_size: int = 0, multi_input=False):
        super().__init__()
        self.root = root
        self.sequences = os.path.join(root, "train")
        self.txt_file = os.path.join(root, "sep_testlist.txt" if test_mode else "sep_trainlist.txt")

        self.sliding_window_size = sliding_window_size if 0 < sliding_window_size < 7 else 7
        self.videos = self.load_paths()
        self.transform = Compose([ToTensor()])
        self.multi_input = multi_input

        assert os.path.exists(self.root)
        assert os.path.exists(self.txt_file)

    def load_paths(self):
        videos = []
        with open(self.txt_file, "r") as f:
            for suffix in f.readlines():
                frame_paths = glob(os.path.join(self.sequences, suffix.strip(), "*.png"))
                for i in range(7 - self.sliding_window_size + 1):
                    video = [path for path in frame_paths[i:i + self.sliding_window_size]]
                    if len(video) > 0:
                        videos.append(video)
                    else:
                        print(f"Skipping {suffix}. Frames not found")
        return videos

    def read_video(self, index):
        video = []
        for path in self.videos[index]:
            video.append(self.transform(Image.open(path).convert("RGB")))
        return torch.stack(video)

    def __getitem__(self, index: int):
        video = self.read_video(index)
        if self.multi_input:
            return video, torch.tensor(0)
        return video, {"vc": torch.tensor(0), "vsr": torch.tensor(0)}

    def __len__(self) -> int:
        return len(self.videos)


class Reds(Dataset):
    def __init__(self, root: str, test_mode: bool = False, sliding_window_size: int = 0, multi_input=False):
        super().__init__()
        prefix = "val" if test_mode else "train"
        self.root = root
        self.sequences = os.path.join(root, f"{prefix}_sharp")

        self.sliding_window_size = sliding_window_size if 0 < sliding_window_size < 7 else 7
        self.videos = self.load_paths()
        self.transform = Compose([ToTensor()])
        self.multi_input = multi_input

        assert os.path.exists(self.root)
        assert os.path.exists(self.sequences)

    def load_paths(self):
        videos = []
        for sequence in os.listdir(self.sequences):
            frame_paths = glob(os.path.join(self.sequences, sequence, "*.png"))
            for i in range(len(frame_paths) - self.sliding_window_size + 1):
                video = [path for path in frame_paths[i:i + self.sliding_window_size]]
                if len(video) > 0:
                    videos.append(video)
                else:
                    print(f"Skipping {sequence}. Frames not found")
        return videos

    def read_video(self, index):
        video = []
        for path in self.videos[index]:
            video.append(self.transform(Image.open(path).convert("RGB")))
        return torch.stack(video)

    def __getitem__(self, index: int):
        video = self.read_video(index)
        if self.multi_input:
            return video, torch.tensor(0)
        return video, {"vc": torch.tensor(0), "vsr": torch.tensor(0)}

    def __len__(self) -> int:
        return len(self.videos)


class Augmentation:
    def __init__(self, multi_input, scale, dataset_type="vimeo"):
        self.multi_input = multi_input
        self.scale = scale
        if dataset_type == "vimeo":
            self.prepare_vimeo_augmentation()
        else:
            self.prepare_reds_augmentation()

    def prepare_reds_augmentation(self):
        if not self.multi_input:
            self.crop_size = (512, 1024)
            self.augmentation = Compose([
                ColorJiggle(brightness=(0.85, 1.15), contrast=(0.75, 1.15), saturation=(0.75, 1.25), hue=(-0.02, 0.02),
                            same_on_batch=True, p=1),
                RandomCrop(size=self.crop_size, same_on_batch=True),
                RandomVerticalFlip(same_on_batch=True, p=0.5),
                RandomHorizontalFlip(same_on_batch=True, p=0.5),
            ])
            self.resize = Resize((self.crop_size[0] // self.scale, self.crop_size[1] // self.scale))
        else:
            self.crop_size = {"vc": (512, 1024), "vsr": (512, 1024)}
            self.augmentation = {
                "vc": Compose([
                    ColorJiggle(brightness=(0.85, 1.15), contrast=(0.75, 1.15), saturation=(0.75, 1.25),
                                hue=(-0.02, 0.02),
                                same_on_batch=True, p=1),
                    RandomCrop(size=self.crop_size["vc"], same_on_batch=True),
                    RandomVerticalFlip(same_on_batch=True, p=0.5),
                    RandomHorizontalFlip(same_on_batch=True, p=0.5),
                ]),
                "vsr": Compose([
                    ColorJiggle(brightness=(0.85, 1.15), contrast=(0.75, 1.15), saturation=(0.75, 1.25),
                                hue=(-0.02, 0.02),
                                same_on_batch=True, p=1),
                    RandomCrop(size=self.crop_size["vsr"], same_on_batch=True),
                    RandomVerticalFlip(same_on_batch=True, p=0.5),
                    RandomHorizontalFlip(same_on_batch=True, p=0.5),
                ])}
            self.resize = {
                "vc": None,
                "vsr": Resize((self.crop_size["vsr"][0] // self.scale, self.crop_size["vsr"][1] // self.scale)),
            }

    def prepare_vimeo_augmentation(self):
        if not self.multi_input:
            self.crop_size = (256, 256)
            self.augmentation = Compose([
                ColorJiggle(brightness=(0.85, 1.15), contrast=(0.75, 1.15), saturation=(0.75, 1.25), hue=(-0.02, 0.02),
                            same_on_batch=True, p=1),
                RandomCrop(size=self.crop_size, same_on_batch=True),
                RandomVerticalFlip(same_on_batch=True, p=0.5),
                RandomHorizontalFlip(same_on_batch=True, p=0.5),
            ])
            self.resize = Resize((self.crop_size[0] // self.scale, self.crop_size[1] // self.scale))
        else:
            self.crop_size = {"vc": (256, 384), "vsr": (256, 256)}
            self.augmentation = {
                "vc": Compose([
                    ColorJiggle(brightness=(0.85, 1.15), contrast=(0.75, 1.15), saturation=(0.75, 1.25),
                                hue=(-0.02, 0.02),
                                same_on_batch=True, p=1),
                    RandomCrop(size=self.crop_size["vc"], same_on_batch=True),
                    RandomVerticalFlip(same_on_batch=True, p=0.5),
                    RandomHorizontalFlip(same_on_batch=True, p=0.5),
                ]),
                "vsr": Compose([
                    ColorJiggle(brightness=(0.85, 1.15), contrast=(0.75, 1.15), saturation=(0.75, 1.25),
                                hue=(-0.02, 0.02),
                                same_on_batch=True, p=1),
                    RandomCrop(size=self.crop_size["vsr"], same_on_batch=True),
                    RandomVerticalFlip(same_on_batch=True, p=0.5),
                    RandomHorizontalFlip(same_on_batch=True, p=0.5),
                ])}
            self.resize = {
                "vc": None,
                "vsr": Resize((self.crop_size["vsr"][0] // self.scale, self.crop_size["vsr"][1] // self.scale)),
            }

    def __call__(self, data, task=None, training_mode=True):
        if not self.multi_input:
            if training_mode:
                hqs = torch.stack([self.augmentation(vid) for vid in data])
                lqs = torch.stack([self.resize(vid) for vid in hqs])
                label = {"vc": lqs[:, -1].clone(), "vsr": hqs[:, -1]}
                return lqs, label
            hqs = data[:, :, :, :self.crop_size[0], :self.crop_size[1]]
            lqs = torch.stack([self.resize(vid) for vid in hqs])
            label = {"vc": lqs[:, -1].clone(), "vsr": hqs[:, -1]}
            return lqs, label
        else:
            if training_mode:
                gt = torch.stack([self.augmentation[task](vid) for vid in data])
                if self.resize[task] is not None:
                    inp = torch.stack([self.resize[task](vid) for vid in gt])
                else:
                    inp = gt.clone()
                return inp, gt[:, -1]
            gt = data[:, :, :, :self.crop_size[task][0], :self.crop_size[task][1]]
            if self.resize[task] is not None:
                inp = torch.stack([self.resize[task](vid) for vid in gt])
            else:
                inp = gt.clone()
            return inp, gt[:, -1]


class UVGDataset(Dataset):
    def __init__(self, root: str, scale: int, max_frames: int = 100, crop_size: Tuple[int, int] = (1024, 1792)):
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
        return self.videos[index][0].split("\\")[-2]

    def get_index_with_name(self, name: str):
        for i, vid in enumerate(self.videos):
            if name in vid[0].split("\\")[-2]:
                return i
        return None

    def __len__(self) -> int:
        return len(self.videos)
