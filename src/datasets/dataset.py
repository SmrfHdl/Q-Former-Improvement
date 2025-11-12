import os

import torch
from torch.utils.data import Dataset, DataLoader

from src.model.clip_vit import VisionEncoder


class VQADataset(Dataset):
    "Format: image - question - answer in text files."
    def __init__(
            self,
            data_file_path: str,
            images_dir: str,
            image_model_name: str = "openai/clip-vit-large-patch14",
            max_length: int = 14,
            device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize the dataset.

        Args:
            data_file_path: path to the file containing image paths, questions, and answers.
            images_dir: directory where images are stored.
            image_model_name: model name for image processing.
            max_length: maximum length for tokenized text.
            device: device to load the model on.
        """
        self.images_dir = images_dir
        self.max_length = max_length
        self.device = device

        self.vision_encoder = VisionEncoder(
            device=device,
            model_name=image_model_name,
            only_use_processor=True
        )

        self.samples = self._load_data_file(data_file_path)

    def _load_data_file(self, data_file_path: str) -> list[dict]:
        samples = []
        valid_answers = {"yes", "no"}

        with open(data_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                
                if not line:
                    continue
                
                image, rest = line.split("\t", 1)
                image = image.split("#")[0] 

                words = rest.strip().split()
                answer = words[-1].lower()
                if not words or answer not in valid_answers:
                    continue

                question_text = ' '.join(words[:-1]).rstrip(' ?')


