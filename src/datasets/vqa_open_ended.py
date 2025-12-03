"""
Open-Ended VQA Dataset Information
==================================

Dataset: soumyasj/vqa-dataset (HuggingFace)
URL: https://huggingface.co/datasets/soumyasj/vqa-dataset

Columns:
    - image_id: int64
    - question_id: int64
    - question: str
    - gt_answer: str (open-ended answer, not just yes/no)
    - image: base64 encoded JPEG string

Example sample:
    image_id: 497855
    question_id: 497885003
    question: "Is the sky blue?"
    gt_answer: "yes"
    image: "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBw..."

Usage:
------
For training, use the trainer at: src/trainers/trainer_openended.py

    python src/trainers/trainer_openended.py \\
        --model_name qformer_improved_openended \\
        --use_clip_for_text True

Or use the dataset directly:

    from datasets.dataset_openended import load_vqa_dataset_from_hf, create_openended_dataloader
    
    # Load data
    train_df, val_df, test_df = load_vqa_dataset_from_hf()
    
    # Create dataloaders (vocabulary built automatically)
    train_loader, val_loader, test_loader, answer_vocab = create_openended_dataloader(
        train_df, val_df, test_df,
        batch_size=32
    )

Available Models:
-----------------
1. QFormerBaseOpenEnded - Base Q-Former for open-ended VQA
2. QFormerImprovedOpenEnded - Improved Q-Former with SGG + NSM for open-ended VQA
"""

# Quick test code to verify dataset loading
if __name__ == "__main__":
    import polars as pl
    
    splits = {
        'train': 'data/train-*.parquet', 
        'validation': 'data/validation-00000-of-00001.parquet', 
        'test': 'data/test-00000-of-00001.parquet'
    }
    
    base_url = 'hf://datasets/soumyasj/vqa-dataset/'
    
    print("Loading VQA dataset from HuggingFace...")
    df_train = pl.read_parquet(base_url + splits['train'])
    df_val = pl.read_parquet(base_url + splits['validation'])
    df_test = pl.read_parquet(base_url + splits['test'])
    
    print(f"Train samples: {len(df_train)}")
    print(f"Validation samples: {len(df_val)}")
    print(f"Test samples: {len(df_test)}")
    print(f"\nColumns: {df_train.columns}")
    print(f"\nSample row:\n{df_train.head(1)}")
