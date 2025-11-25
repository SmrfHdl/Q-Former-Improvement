import torch
import sys
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

from model.q_former_improved import QFormerImproved

def test_forward_pass():
    print("=" * 50)
    print("Testing QFormerImproved Forward Pass")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model with test hyperparameters
    model = QFormerImproved(
        sequence_size=24,
        qformer_hidden_size=768,
        blocks_num=6,
        num_heads=4,
        num_object_queries=32,
        num_relation_queries=64,
        num_global_queries=32,
        device=device,
        use_clip_for_text=True,
        dropout_rate=0.2,
        unfreeze_layers=0
    )
    
    print(f"\nModel created successfully")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create dummy input
    batch_size = 2
    dummy_samples = {
        'image_input': torch.randn(batch_size, 3, 224, 224).to(device),
        'question': ['What is in the image?', 'Is there a dog?'],
        'answer': ['yes', 'no']
    }
    
    print(f"\nDummy input created (batch_size={batch_size})")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        try:
            output = model(dummy_samples)
            print(f"\nForward pass successful!")
            
            print(f"\nOutput keys: {list(output.keys())}")
            print(f"  - answer_accuracy: {output['answer_accuracy'].item():.4f}")
            print(f"  - loss_answer: {output['loss_answer'].item():.4f}")
            print(f"  - loss_itc: {output['loss_itc'].item():.4f}")
            print(f"  - loss_itm: {output['loss_itm'].item():.4f}")
            print(f"  - loss_igt: {output['loss_igt'].item():.4f}")
            print(f"  - loss_object: {output['loss_object'].item():.4f}")
            print(f"  - loss_relation: {output['loss_relation'].item():.4f}")
            print(f"  - total_loss: {output['total_loss'].item():.4f}")
            
            print(f"\nOutput shapes:")
            print(f"  - answer_predictions: {output['answer_predictions'].shape}")
            print(f"  - answer_labels: {output['answer_labels'].shape}")
            print(f"  - object_confidence: {output['object_confidence'].shape}")
            print(f"  - spatial_info: {output['spatial_info'].shape}")
            
            print("\n" + "=" * 50)
            print("ALL TESTS PASSED!")
            print("=" * 50)
            
            return True
            
        except Exception as e:
            print(f"\n✗ Forward pass failed with error:")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = test_forward_pass()
    sys.exit(0 if success else 1)
