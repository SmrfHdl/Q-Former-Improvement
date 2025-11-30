import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from model.q_former_improved import QFormerImproved
import torch

from loguru import logger

class QFormerImprovedLightning(pl.LightningModule):
    def __init__(self,
                 hyperparams: dict,
                 device: torch.device):
        super().__init__()
        self.save_hyperparameters()

        self.hyperparams = hyperparams

        self.q_former_improved = QFormerImproved(
            sequence_size=hyperparams["sequence_size"],
            qformer_hidden_size=hyperparams["qformer_hidden_size"],
            blocks_num=self.hyperparams["blocks_num"],
            num_heads=hyperparams["num_heads"],
            num_object_queries=hyperparams.get("num_object_queries", 32),
            num_relation_queries=hyperparams.get("num_relation_queries", 64),
            num_global_queries=hyperparams.get("num_global_queries", 32),
            dropout_rate=hyperparams["dropout_rate"],
            use_clip_for_text=hyperparams["use_clip_for_text"],
            unfreeze_layers=hyperparams["unfreeze_layers"],
            device=device
        )

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Transfer nested dict batch to device - handles image_input dict."""
        if isinstance(batch, dict):
            for key, value in batch.items():
                if isinstance(value, dict):
                    batch[key] = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                  for k, v in value.items()}
                elif isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
        return batch

    def forward(self, samples: dict):
        return self.q_former_improved(samples)
    
    def _common_step(self, batch: dict, task: str):
        output = self.forward(batch)

        # Log all metrics
        self.log(f"{task}_answer_accuracy", output['answer_accuracy'], prog_bar=True, on_step=True, on_epoch=True, logger=True,
                 batch_size=self.hyperparams['batch_size'])
        self.log(f"{task}_loss_itc", output['loss_itc'], prog_bar=True, on_step=True, on_epoch=True, logger=True,
                 batch_size=self.hyperparams['batch_size'])
        self.log(f"{task}_loss_igt", output['loss_igt'], prog_bar=True, on_step=True, on_epoch=True, logger=True,
                 batch_size=self.hyperparams['batch_size'])
        self.log(f"{task}_loss_itm", output['loss_itm'], prog_bar=True, on_step=True, on_epoch=True, logger=True,
                 batch_size=self.hyperparams['batch_size'])
        self.log(f"{task}_loss_answer", output['loss_answer'], prog_bar=True, on_step=True, on_epoch=True, logger=True,
                 batch_size=self.hyperparams['batch_size'])
        
        # Log hierarchical losses
        self.log(f"{task}_loss_object", output['loss_object'], prog_bar=True, on_step=True, on_epoch=True, logger=True,
                 batch_size=self.hyperparams['batch_size'])
        self.log(f"{task}_loss_relation", output['loss_relation'], prog_bar=True, on_step=True, on_epoch=True, logger=True,
                 batch_size=self.hyperparams['batch_size'])
        
        self.log(f"{task}_total_loss", output['total_loss'], prog_bar=True, on_step=True, on_epoch=True, logger=True,
                 batch_size=self.hyperparams['batch_size'])
        
        logs = ""
        logs += f"{task}_answer_accuracy: {output['answer_accuracy'].item():.4f}, "
        logs += f"{task}_loss_itc: {output['loss_itc'].item():.4f}, "
        logs += f"{task}_loss_igt: {output['loss_igt'].item():.4f}, "
        logs += f"{task}_loss_itm: {output['loss_itm'].item():.4f}, "
        logs += f"{task}_loss_answer: {output['loss_answer'].item():.4f}, "
        logs += f"{task}_loss_object: {output['loss_object'].item():.4f}, "
        logs += f"{task}_loss_relation: {output['loss_relation'].item():.4f}, "
        logs += f"{task}_total_loss: {output['total_loss'].item():.4f}"
        
        logger.info(logs)

        return output
    
    def training_step(self, batch: dict):
        output = self._common_step(batch, task="train")
        total_loss = output['total_loss']
        
        # Skip backward if loss is NaN or Inf to prevent weight corruption
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.warning(f"NaN/Inf detected in total_loss, skipping backward pass")
            return None
        
        return total_loss
    
    def on_after_backward(self):
        """Check for NaN/Inf in weights and gradients after backward pass."""
        # Check projection layer weights for corruption
        vision_proj_weight = self.q_former_improved.vision_projection.weight
        text_proj_weight = self.q_former_improved.text_projection.weight
        
        if torch.isnan(vision_proj_weight).any() or torch.isinf(vision_proj_weight).any():
            logger.error("CRITICAL: NaN/Inf detected in vision_projection weights after backward!")
            self.q_former_improved.zero_grad()
            
        if torch.isnan(text_proj_weight).any() or torch.isinf(text_proj_weight).any():
            logger.error("CRITICAL: NaN/Inf detected in text_projection weights after backward!")
            self.q_former_improved.zero_grad()
    
    def validation_step(self, batch: dict):
        output = self._common_step(batch, task="val")
        return output
    
    def test_step(self, batch: dict):
        output = self._common_step(batch, task="test")
        return output
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hyperparams['lr'],
            betas=self.hyperparams['betas'],
            weight_decay=self.hyperparams['weight_decay'],
            eps=self.hyperparams['eps']
        )
        
        # Add warm-up scheduler: linear warm-up for first 500 steps (reasonable for typical VQA datasets)
        # With batch_size=48 and ~10k samples, 500 steps = ~2.4 epochs of warmup
        def lr_lambda(current_step: int):
            warmup_steps = self.hyperparams.get('warmup_steps', 500)  # Use config or default 500
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0
        
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    
    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val=None,
        gradient_clip_algorithm=None
    ):
        """Configure gradient clipping to prevent gradient explosion."""
        self.clip_gradients(
            optimizer,
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm"
        )
    
    def save_checkpoint(self, file_path: str):
        self.trainer.save_checkpoint(file_path)
