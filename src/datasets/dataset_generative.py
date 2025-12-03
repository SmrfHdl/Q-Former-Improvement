"""
Dataset for Generative Open-Ended VQA.

Simpler than classification version - just returns answer string for generation.
No vocabulary needed since model generates text directly.
"""

import io
import base64
import random
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import polars as pl

from src.model.clip_vit import VisionEncoder
from loguru import logger


class GenerativeVQADataset(Dataset):
    """
    Dataset for Generative VQA.
    
    Returns:
        - image: processed image tensor
        - question: question string
        - answer: answer string (ground truth for generation)
    """
    def __init__(
            self,
            df: pl.DataFrame,
            image_model_name: str = "openai/clip-vit-large-patch14",
            device: torch.device = torch.device("cpu"),
            use_augmentation: bool = False,
            augmentation_prob: float = 0.5,
    ):
        self.df = df
        self.device = device
        self.use_augmentation = use_augmentation
        self.augmentation_prob = augmentation_prob
        
        self.vision_encoder = VisionEncoder(
            device=device,
            model_name=image_model_name,
            only_use_processor=True
        )
        
        if use_augmentation:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            ])
        else:
            self.augment_transform = None
            
    def _decode_image(self, image_data: str) -> Image.Image:
        """Decode base64 image string."""
        try:
            if image_data.startswith('data:'):
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            return image
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            return Image.new("RGB", (224, 224), color=(128, 128, 128))
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, index: int) -> dict:
        row = self.df.row(index, named=True)
        
        pil_image = self._decode_image(row['image'])
        
        if self.use_augmentation and self.augment_transform is not None:
            if random.random() < self.augmentation_prob:
                pil_image = self.augment_transform(pil_image)
        
        image = self.vision_encoder.processor(images=pil_image, return_tensors="pt")
        
        return {
            "image": image["pixel_values"].squeeze(0),
            "question": row['question'],
            "answer": row['gt_answer'].lower().strip(),  # Just the string
            "image_id": row['image_id'],
            "question_id": row['question_id'],
        }
    
    def collate_fn(self, batch: list[dict]) -> dict:
        images = torch.stack([item["image"] for item in batch])
        
        return {
            "image_input": {"pixel_values": images},
            "question": [item["question"] for item in batch],
            "answer": [item["answer"] for item in batch],
            "image_id": [item["image_id"] for item in batch],
            "question_id": [item["question_id"] for item in batch],
        }


def load_vqa_dataset_from_hf() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load VQA dataset from HuggingFace."""
    splits = {
        'train': 'data/train-*.parquet', 
        'validation': 'data/validation-00000-of-00001.parquet', 
        'test': 'data/test-00000-of-00001.parquet'
    }
    
    base_url = 'hf://datasets/soumyasj/vqa-dataset/'
    
    logger.info("Loading VQA dataset from HuggingFace...")
    
    df_train = pl.read_parquet(base_url + splits['train'])
    logger.info(f"Loaded train: {len(df_train)} samples")
    
    df_val = pl.read_parquet(base_url + splits['validation'])
    logger.info(f"Loaded val: {len(df_val)} samples")
    
    df_test = pl.read_parquet(base_url + splits['test'])
    logger.info(f"Loaded test: {len(df_test)} samples")
    
    return df_train, df_val, df_test


def create_generative_dataloader(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
    use_augmentation: bool = False,
    augmentation_prob: float = 0.5,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for generative VQA."""
    
    train_dataset = GenerativeVQADataset(
        df=train_df, device=device,
        use_augmentation=use_augmentation, augmentation_prob=augmentation_prob
    )
    
    val_dataset = GenerativeVQADataset(
        df=val_df, device=device, use_augmentation=False
    )
    
    test_dataset = GenerativeVQADataset(
        df=test_df, device=device, use_augmentation=False
    )
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    kwargs = {
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'persistent_workers': num_workers > 0,
        'prefetch_factor': 2 if num_workers > 0 else None,
    }
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              collate_fn=train_dataset.collate_fn, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=val_dataset.collate_fn, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=test_dataset.collate_fn, **kwargs)
    
    return train_loader, val_loader, test_loader

