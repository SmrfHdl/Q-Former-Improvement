"""
Open-Ended VQA Dataset from HuggingFace.

Dataset: soumyasj/vqa-dataset
Columns: image_id, question_id, question, gt_answer, image (base64 encoded)

This module handles:
1. Loading parquet files from HuggingFace
2. Building answer vocabulary from training data
3. Processing images from base64 format
4. Supporting both training (with answer labels) and inference modes
"""

import os
import io
import base64
import random
from collections import Counter
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import polars as pl

from src.model.clip_vit import VisionEncoder
from loguru import logger


class AnswerVocabulary:
    """
    Manages the answer vocabulary for open-ended VQA.
    
    Standard VQA approach: treat as classification over top-K most frequent answers.
    """
    def __init__(self, min_count: int = 9, max_vocab_size: int = 3129):
        """
        Args:
            min_count: Minimum occurrence count to include an answer
            max_vocab_size: Maximum vocabulary size (VQA v2 uses 3129)
        """
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.answer_to_idx = {}
        self.idx_to_answer = {}
        self.answer_counts = Counter()
        
        # Special tokens
        self.unk_token = "<UNK>"
        self.unk_idx = 0
        
    def build_from_answers(self, answers: list[str]):
        """Build vocabulary from a list of answers."""
        # Count answer frequencies
        self.answer_counts = Counter(answers)
        
        # Filter by minimum count and sort by frequency
        filtered_answers = [
            ans for ans, count in self.answer_counts.most_common()
            if count >= self.min_count
        ]
        
        # Limit to max vocab size
        filtered_answers = filtered_answers[:self.max_vocab_size - 1]  # Reserve 0 for UNK
        
        # Build mappings
        self.answer_to_idx = {self.unk_token: self.unk_idx}
        self.idx_to_answer = {self.unk_idx: self.unk_token}
        
        for idx, answer in enumerate(filtered_answers, start=1):
            self.answer_to_idx[answer] = idx
            self.idx_to_answer[idx] = answer
            
        logger.info(f"Built answer vocabulary with {len(self.answer_to_idx)} answers")
        logger.info(f"Top 10 answers: {filtered_answers[:10]}")
        
    def encode(self, answer: str) -> int:
        """Convert answer string to index."""
        return self.answer_to_idx.get(answer.lower().strip(), self.unk_idx)
    
    def decode(self, idx: int) -> str:
        """Convert index to answer string."""
        return self.idx_to_answer.get(idx, self.unk_token)
    
    def __len__(self) -> int:
        return len(self.answer_to_idx)
    
    def save(self, path: str):
        """Save vocabulary to file."""
        import json
        with open(path, 'w') as f:
            json.dump({
                'answer_to_idx': self.answer_to_idx,
                'idx_to_answer': {str(k): v for k, v in self.idx_to_answer.items()},
                'min_count': self.min_count,
                'max_vocab_size': self.max_vocab_size
            }, f)
        logger.info(f"Saved vocabulary to {path}")
            
    def load(self, path: str):
        """Load vocabulary from file."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        self.answer_to_idx = data['answer_to_idx']
        self.idx_to_answer = {int(k): v for k, v in data['idx_to_answer'].items()}
        self.min_count = data.get('min_count', 9)
        self.max_vocab_size = data.get('max_vocab_size', 3129)
        logger.info(f"Loaded vocabulary with {len(self.answer_to_idx)} answers from {path}")


class OpenEndedVQADataset(Dataset):
    """
    Open-Ended VQA Dataset from HuggingFace parquet files.
    
    Format:
        image_id: int64
        question_id: int64  
        question: str
        gt_answer: str
        image: base64 encoded string
    """
    def __init__(
            self,
            df: pl.DataFrame,
            answer_vocab: AnswerVocabulary,
            image_model_name: str = "openai/clip-vit-large-patch14",
            device: torch.device = torch.device("cpu"),
            use_augmentation: bool = False,
            augmentation_prob: float = 0.5,
    ):
        """
        Args:
            df: Polars DataFrame with VQA data
            answer_vocab: Answer vocabulary for encoding
            image_model_name: CLIP model name for image processing
            device: Device for processing
            use_augmentation: Enable data augmentation (for training)
            augmentation_prob: Probability of applying augmentation
        """
        self.df = df
        self.answer_vocab = answer_vocab
        self.device = device
        self.use_augmentation = use_augmentation
        self.augmentation_prob = augmentation_prob
        
        # Vision processor
        self.vision_encoder = VisionEncoder(
            device=device,
            model_name=image_model_name,
            only_use_processor=True
        )
        
        # Data augmentation transforms
        if use_augmentation:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
                transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
                transforms.RandomGrayscale(p=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            ])
            logger.info(f"Data augmentation enabled with prob={augmentation_prob}")
        else:
            self.augment_transform = None
            
    def _decode_image(self, image_data: str) -> Image.Image:
        """Decode base64 image string to PIL Image."""
        try:
            # Handle both raw base64 and data URL format
            if image_data.startswith('data:'):
                # Extract base64 part from data URL
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            return image
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            # Return a blank image as fallback
            return Image.new("RGB", (224, 224), color=(128, 128, 128))
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, index: int) -> dict:
        """Get a sample from the dataset."""
        row = self.df.row(index, named=True)
        
        # Decode image from base64
        pil_image = self._decode_image(row['image'])
        
        # Apply augmentation if enabled
        if self.use_augmentation and self.augment_transform is not None:
            if random.random() < self.augmentation_prob:
                pil_image = self.augment_transform(pil_image)
        
        # Process image with CLIP processor
        image = self.vision_encoder.processor(images=pil_image, return_tensors="pt")
        
        # Encode answer
        answer_str = row['gt_answer'].lower().strip()
        answer_idx = self.answer_vocab.encode(answer_str)
        
        return {
            "image": image["pixel_values"].squeeze(0),
            "question": row['question'],
            "answer": answer_str,
            "answer_idx": answer_idx,
            "image_id": row['image_id'],
            "question_id": row['question_id'],
        }
    
    def collate_fn(self, batch: list[dict]) -> dict:
        """Collate batch of samples."""
        images = torch.stack([item["image"] for item in batch])
        image_input = {"pixel_values": images}
        
        answer_indices = torch.tensor([item["answer_idx"] for item in batch], dtype=torch.long)
        
        return {
            "image_input": image_input,
            "question": [item["question"] for item in batch],
            "answer": [item["answer"] for item in batch],
            "answer_idx": answer_indices,
            "image_id": [item["image_id"] for item in batch],
            "question_id": [item["question_id"] for item in batch],
        }


def load_vqa_dataset_from_hf(
    cache_dir: Optional[str] = None,
    use_local_cache: bool = True
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Load VQA dataset from HuggingFace.
    
    Args:
        cache_dir: Directory to cache downloaded files
        use_local_cache: Whether to use local cache
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    splits = {
        'train': 'data/train-*.parquet', 
        'validation': 'data/validation-00000-of-00001.parquet', 
        'test': 'data/test-00000-of-00001.parquet'
    }
    
    base_url = 'hf://datasets/soumyasj/vqa-dataset/'
    
    logger.info("Loading VQA dataset from HuggingFace...")
    
    df_train = pl.read_parquet(base_url + splits['train'])
    logger.info(f"Loaded train split: {len(df_train)} samples")
    
    df_val = pl.read_parquet(base_url + splits['validation'])
    logger.info(f"Loaded validation split: {len(df_val)} samples")
    
    df_test = pl.read_parquet(base_url + splits['test'])
    logger.info(f"Loaded test split: {len(df_test)} samples")
    
    return df_train, df_val, df_test


def create_openended_dataloader(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    answer_vocab: Optional[AnswerVocabulary] = None,
    image_model_name: str = "openai/clip-vit-large-patch14",
    batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
    use_augmentation: bool = False,
    augmentation_prob: float = 0.5,
    num_workers: int = 0,
    pin_memory: bool = False,
    vocab_min_count: int = 9,
    vocab_max_size: int = 3129,
) -> tuple[DataLoader, DataLoader, DataLoader, AnswerVocabulary]:
    """
    Create DataLoaders for open-ended VQA.
    
    Args:
        train_df, val_df, test_df: Polars DataFrames with VQA data
        answer_vocab: Pre-built answer vocabulary (if None, will build from training data)
        image_model_name: CLIP model name
        batch_size: Batch size
        device: Device
        use_augmentation: Enable augmentation for training
        augmentation_prob: Augmentation probability
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        vocab_min_count: Minimum count for vocabulary inclusion
        vocab_max_size: Maximum vocabulary size
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, answer_vocab)
    """
    # Build answer vocabulary from training data if not provided
    if answer_vocab is None:
        answer_vocab = AnswerVocabulary(
            min_count=vocab_min_count,
            max_vocab_size=vocab_max_size
        )
        train_answers = train_df['gt_answer'].to_list()
        answer_vocab.build_from_answers(train_answers)
    
    # Create datasets
    train_dataset = OpenEndedVQADataset(
        df=train_df,
        answer_vocab=answer_vocab,
        image_model_name=image_model_name,
        device=device,
        use_augmentation=use_augmentation,
        augmentation_prob=augmentation_prob
    )
    
    val_dataset = OpenEndedVQADataset(
        df=val_df,
        answer_vocab=answer_vocab,
        image_model_name=image_model_name,
        device=device,
        use_augmentation=False
    )
    
    test_dataset = OpenEndedVQADataset(
        df=test_df,
        answer_vocab=answer_vocab,
        image_model_name=image_model_name,
        device=device,
        use_augmentation=False
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    logger.info(f"Answer vocabulary size: {len(answer_vocab)}")
    
    # DataLoader kwargs
    dataloader_kwargs = {
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'persistent_workers': num_workers > 0,
        'prefetch_factor': 2 if num_workers > 0 else None,
    }
    
    if num_workers > 0:
        logger.info(f"Using {num_workers} data loading workers with pin_memory={pin_memory}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        **dataloader_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        **dataloader_kwargs
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
        **dataloader_kwargs
    )
    
    return train_loader, val_loader, test_loader, answer_vocab


# Convenience function for quick loading
def create_openended_vqa_dataloaders(
    batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
    use_augmentation: bool = False,
    num_workers: int = 0,
    vocab_path: Optional[str] = None,
    **kwargs
) -> tuple[DataLoader, DataLoader, DataLoader, AnswerVocabulary]:
    """
    Convenience function to load open-ended VQA dataset and create dataloaders.
    
    Args:
        batch_size: Batch size
        device: Device
        use_augmentation: Enable augmentation
        num_workers: Number of workers
        vocab_path: Path to pre-built vocabulary (optional)
        **kwargs: Additional arguments for create_openended_dataloader
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, answer_vocab)
    """
    # Load data from HuggingFace
    train_df, val_df, test_df = load_vqa_dataset_from_hf()
    
    # Load or create vocabulary
    answer_vocab = None
    if vocab_path and os.path.exists(vocab_path):
        answer_vocab = AnswerVocabulary()
        answer_vocab.load(vocab_path)
    
    return create_openended_dataloader(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        answer_vocab=answer_vocab,
        batch_size=batch_size,
        device=device,
        use_augmentation=use_augmentation,
        num_workers=num_workers,
        **kwargs
    )

