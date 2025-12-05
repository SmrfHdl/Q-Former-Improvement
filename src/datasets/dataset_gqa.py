import io
import re
import random
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import polars as pl

from src.model.clip_vit import VisionEncoder
from loguru import logger


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for exact match comparison.
    
    Follows standard VQA normalization:
    - Lowercase
    - Remove articles (a, an, the)
    - Remove punctuation
    - Remove extra whitespace
    - Convert number words to digits
    """
    answer = answer.lower().strip()
    
    # Remove punctuation
    answer = re.sub(r'[^\w\s]', '', answer)
    
    # Remove articles
    articles = ['a', 'an', 'the']
    words = answer.split()
    words = [w for w in words if w not in articles]
    answer = ' '.join(words)
    
    # Number word to digit mapping
    number_map = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
        'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
        'eighteen': '18', 'nineteen': '19', 'twenty': '20'
    }
    
    words = answer.split()
    words = [number_map.get(w, w) for w in words]
    answer = ' '.join(words)
    
    # Remove extra whitespace
    answer = ' '.join(answer.split())
    
    return answer


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """
    Compute exact match after normalization.
    
    Returns 1.0 if normalized prediction == normalized ground truth, else 0.0
    """
    pred_normalized = normalize_answer(prediction)
    gt_normalized = normalize_answer(ground_truth)
    
    return 1.0 if pred_normalized == gt_normalized else 0.0


def compute_batch_exact_match(predictions: list[str], ground_truths: list[str]) -> float:
    """Compute exact match accuracy for a batch."""
    if len(predictions) == 0:
        return 0.0
    
    total_score = sum(
        compute_exact_match(pred, gt) 
        for pred, gt in zip(predictions, ground_truths)
    )
    
    return total_score / len(predictions)


class GQADataset(Dataset):
    """
    Dataset for GQA Visual Reasoning.
    
    Returns:
        - image: processed image tensor
        - question: question string
        - answer: answer string (ground truth for generation)
        - question_id: unique question identifier
        - image_id: image identifier
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
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            ])
        else:
            self.augment_transform = None
            
    def _decode_image(self, image_data) -> Image.Image:
        """Decode image from various formats."""
        try:
            # GQA images are stored as PIL Image objects or bytes
            if isinstance(image_data, Image.Image):
                return image_data.convert("RGB")
            elif isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                return image
            elif isinstance(image_data, dict) and 'bytes' in image_data:
                image = Image.open(io.BytesIO(image_data['bytes'])).convert("RGB")
                return image
            else:
                logger.warning(f"Unknown image format: {type(image_data)}")
                return Image.new("RGB", (224, 224), color=(128, 128, 128))
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
        
        # GQA uses 'id' for question_id and 'imageId' for image_id
        question_id = str(row.get('id', row.get('question_id', index)))
        image_id = str(row.get('imageId', row.get('image_id', '')))
        
        return {
            "image": image["pixel_values"].squeeze(0),
            "question": row['question'],
            "answer": row['answer'].lower().strip(),
            "image_id": image_id,
            "question_id": question_id,
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


def load_gqa_dataset_from_hf(
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Load GQA dataset from HuggingFace.
    
    Dataset: lmms-lab/GQA
    Splits: train_balanced, val_balanced, test_balanced
    
    Each split has:
        - {split}_images: Contains image data
        - {split}_instructions: Contains question/answer data
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    base_url = 'hf://datasets/lmms-lab/GQA/'
    
    logger.info("Loading GQA dataset from HuggingFace...")
    
    # Load train split
    logger.info("Loading train_balanced split...")
    try:
        df_train_images = pl.read_parquet(base_url + 'train_balanced_images/*.parquet')
        df_train_instructions = pl.read_parquet(base_url + 'train_balanced_instructions/*.parquet')
        
        # Merge on imageId
        df_train = df_train_instructions.join(
            df_train_images.select(['imageId', 'image']),
            on='imageId',
            how='left'
        )
        
        if max_train_samples and len(df_train) > max_train_samples:
            df_train = df_train.sample(n=max_train_samples, seed=42)
        
        logger.info(f"Loaded train: {len(df_train)} samples")
    except Exception as e:
        logger.error(f"Error loading train split: {e}")
        raise
    
    # Load validation split
    logger.info("Loading val_balanced split...")
    try:
        df_val_images = pl.read_parquet(base_url + 'val_balanced_images/*.parquet')
        df_val_instructions = pl.read_parquet(base_url + 'val_balanced_instructions/*.parquet')
        
        df_val = df_val_instructions.join(
            df_val_images.select(['imageId', 'image']),
            on='imageId',
            how='left'
        )
        
        if max_val_samples and len(df_val) > max_val_samples:
            df_val = df_val.sample(n=max_val_samples, seed=42)
        
        logger.info(f"Loaded val: {len(df_val)} samples")
    except Exception as e:
        logger.error(f"Error loading val split: {e}")
        raise
    
    # Load test split
    logger.info("Loading test_balanced split...")
    try:
        df_test_images = pl.read_parquet(base_url + 'test_balanced_images/*.parquet')
        df_test_instructions = pl.read_parquet(base_url + 'test_balanced_instructions/*.parquet')
        
        df_test = df_test_instructions.join(
            df_test_images.select(['imageId', 'image']),
            on='imageId',
            how='left'
        )
        
        if max_test_samples and len(df_test) > max_test_samples:
            df_test = df_test.sample(n=max_test_samples, seed=42)
        
        logger.info(f"Loaded test: {len(df_test)} samples")
    except Exception as e:
        logger.error(f"Error loading test split: {e}")
        raise
    
    logger.info(f"GQA dataset loaded successfully!")
    logger.info(f"  Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    
    return df_train, df_val, df_test


def create_gqa_dataloader(
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
    """Create DataLoaders for GQA dataset."""
    
    train_dataset = GQADataset(
        df=train_df, device=device,
        use_augmentation=use_augmentation, augmentation_prob=augmentation_prob
    )
    
    val_dataset = GQADataset(
        df=val_df, device=device, use_augmentation=False
    )
    
    test_dataset = GQADataset(
        df=test_df, device=device, use_augmentation=False
    )
    
    logger.info(f"Created datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    kwargs = {
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'persistent_workers': num_workers > 0,
        'prefetch_factor': 2 if num_workers > 0 else None,
    }
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        collate_fn=train_dataset.collate_fn, **kwargs
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=val_dataset.collate_fn, **kwargs
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=test_dataset.collate_fn, **kwargs
    )
    
    return train_loader, val_loader, test_loader

