import os
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from src.model.clip_vit import VisionEncoder

from loguru import logger


class VQADataset(Dataset):
    "Format: image - question - answer in text files."
    def __init__(
            self,
            data_file_path: str,
            images_dir: str,
            image_model_name: str = "openai/clip-vit-large-patch14",
            max_length: int = 14,
            device: torch.device = torch.device("cpu"),
            use_augmentation: bool = False,
            augmentation_prob: float = 0.5,
    ):
        """
        Initialize the dataset.

        Args:
            data_file_path: path to the file containing image paths, questions, and answers.
            images_dir: directory where images are stored.
            image_model_name: model name for image processing.
            max_length: maximum length for tokenized text.
            device: device to load the model on.
            use_augmentation: whether to use data augmentation (for training).
            augmentation_prob: probability of applying augmentation.
        """
        self.images_dir = images_dir
        self.max_length = max_length
        self.device = device
        self.use_augmentation = use_augmentation
        self.augmentation_prob = augmentation_prob

        self.vision_encoder = VisionEncoder(
            device=device,
            model_name=image_model_name,
            only_use_processor=True
        )
        
        # v2 IMPROVEMENT: Data augmentation for training
        if use_augmentation:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            ])
            logger.info(f"Data augmentation enabled with prob={augmentation_prob}")
        else:
            self.augment_transform = None

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

                image_path = os.path.join(self.images_dir, image)

                if not os.path.exists(image_path):
                    logger.warning(f"Image not found: {image_path}. Skipping sample.")
                    continue

                samples.append({
                    "image_id": image,
                    "question": question_text,
                    "answer": answer,
                    "image_path": image_path
                })

        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> dict:
        """
        Get a sample from the dataset.
        
        Returns: 
            Dict with processed image features, question, and answer.
        """
        sample = self.samples[index]

        # v2 IMPROVEMENT: Apply data augmentation if enabled
        if self.use_augmentation and self.augment_transform is not None:
            if random.random() < self.augmentation_prob:
                # Load image, apply augmentation, then process
                pil_image = Image.open(sample["image_path"]).convert("RGB")
                pil_image = self.augment_transform(pil_image)
                # Process augmented image
                image = self.vision_encoder.processor(images=pil_image, return_tensors="pt")
                image = image.to(self.device)
            else:
                image = self.vision_encoder.path_to_tensor(sample["image_path"])
        else:
            image = self.vision_encoder.path_to_tensor(sample["image_path"])

        output = {
            "image": image["pixel_values"].squeeze(0),
            "question": sample["question"],
            "answer": sample["answer"],
            "image_id": sample["image_id"]
        }

        return output
    
    def collate_fn(self, batch: list[dict]) -> dict:
        batch_dict = {}

        images = torch.stack([item["image"] for item in batch]).to(self.device)
        image_input = {"pixel_values": images.to(self.device)}
        batch_dict["image_input"] = image_input

        batch_dict["question"] = [item["question"] for item in batch]
        batch_dict["answer"] = [item["answer"] for item in batch]
        batch_dict["image_id"] = [item["image_id"] for item in batch]

        return batch_dict
    
def create_dataloader(
        train_file: str,
        val_file: str,
        test_file: str,
        images_dir: str,
        image_model_name: str = "openai/clip-vit-large-patch14",
        batch_size: int = 32,
        device: torch.device = torch.device("cpu"),
        use_augmentation: bool = False,
        augmentation_prob: float = 0.5
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for training, validation, and testing datasets.

    Args:
        train_file: path to the training data file.
        val_file: path to the validation data file.
        test_file: path to the testing data file.
        images_dir: directory where images are stored.
        image_model_name: model name for image processing.
        batch_size: batch size for DataLoaders.
        device: device to load the model on.
        use_augmentation: whether to use data augmentation for training.
        augmentation_prob: probability of applying augmentation.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # v2 IMPROVEMENT: Apply augmentation only to training set
    train_dataset = VQADataset(
        data_file_path=train_file,
        images_dir=images_dir,
        image_model_name=image_model_name,
        device=device,
        use_augmentation=use_augmentation,
        augmentation_prob=augmentation_prob
    )

    # No augmentation for validation and test
    val_dataset = VQADataset(
        data_file_path=val_file,
        images_dir=images_dir,
        image_model_name=image_model_name,
        device=device,
        use_augmentation=False
    )

    test_dataset = VQADataset(
        data_file_path=test_file,
        images_dir=images_dir,
        image_model_name=image_model_name,
        device=device,
        use_augmentation=False
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_fn
    )

    return train_dataloader, val_dataloader, test_dataloader