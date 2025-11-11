import os

import torch
from torch.utils.data import Dataset, DataLoader

from src.model.clip_vit import VisionEncoder


class VQADataset(Dataset):
    "Format: image - question - answer in text files."
    def __init__(
            self,
            data_file_path: str,
            images_dir
    )