"""
Lightning modules for Generative VQA models.
"""

import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import torch

from loguru import logger


class QFormerBaseGenerativeLightning(pl.LightningModule):
    """Lightning module for QFormerBaseGenerative."""
    
    def __init__(self, hyperparams: dict, device: torch.device):
        super().__init__()
        self.save_hyperparameters()
        self.hyperparams = hyperparams
        
        from model.q_former_base_generative import QFormerBaseGenerative
        
        self.model = QFormerBaseGenerative(
            sequence_size=hyperparams["sequence_size"],
            qformer_hidden_size=hyperparams["qformer_hidden_size"],
            blocks_num=hyperparams["blocks_num"],
            num_heads=hyperparams["num_heads"],
            num_queries=hyperparams.get("num_queries", 32),
            max_answer_length=hyperparams.get("max_answer_length", 10),
            dropout_rate=hyperparams["dropout_rate"],
            use_clip_for_text=hyperparams["use_clip_for_text"],
            unfreeze_layers=hyperparams["unfreeze_layers"],
            device=device
        )

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if isinstance(batch, dict):
            for key, value in batch.items():
                if isinstance(value, dict):
                    batch[key] = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                  for k, v in value.items()}
                elif isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
        return batch

    def forward(self, samples: dict):
        return self.model(samples)
    
    def _common_step(self, batch: dict, task: str):
        output = self.forward(batch)

        self.log(f"{task}_answer_accuracy", output['answer_accuracy'], prog_bar=True, 
                 on_step=True, on_epoch=True, batch_size=self.hyperparams['batch_size'])
        self.log(f"{task}_loss_generation", output['loss_generation'], prog_bar=True,
                 on_step=True, on_epoch=True, batch_size=self.hyperparams['batch_size'])
        self.log(f"{task}_loss_itc", output['loss_itc'], 
                 on_step=True, on_epoch=True, batch_size=self.hyperparams['batch_size'])
        self.log(f"{task}_loss_itm", output['loss_itm'],
                 on_step=True, on_epoch=True, batch_size=self.hyperparams['batch_size'])
        self.log(f"{task}_total_loss", output['total_loss'], prog_bar=True,
                 on_step=True, on_epoch=True, batch_size=self.hyperparams['batch_size'])
        
        # Log some generated examples periodically
        if task == "val" and self.global_step % 100 == 0:
            for i in range(min(3, len(output['generated_answers']))):
                logger.info(f"Q: {batch['question'][i]}")
                logger.info(f"GT: {output['ground_truth_answers'][i]}")
                logger.info(f"Gen: {output['generated_answers'][i]}")
                logger.info("---")

        return output
    
    def training_step(self, batch: dict):
        output = self._common_step(batch, task="train")
        if torch.isnan(output['total_loss']) or torch.isinf(output['total_loss']):
            return None
        return output['total_loss']
    
    def validation_step(self, batch: dict):
        return self._common_step(batch, task="val")
    
    def test_step(self, batch: dict):
        return self._common_step(batch, task="test")
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hyperparams['lr'],
            betas=self.hyperparams['betas'],
            weight_decay=self.hyperparams['weight_decay'],
            eps=self.hyperparams['eps']
        )
        
        def lr_lambda(step: int):
            warmup = self.hyperparams.get('warmup_steps', 500)
            if step < warmup:
                return float(step) / float(max(1, warmup))
            return 1.0
        
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
    
    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        self.clip_gradients(optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm")


class QFormerImprovedGenerativeLightning(pl.LightningModule):
    """Lightning module for QFormerImprovedGenerative."""
    
    def __init__(self, hyperparams: dict, device: torch.device):
        super().__init__()
        self.save_hyperparameters()
        self.hyperparams = hyperparams
        
        from model.q_former_improved_generative import QFormerImprovedGenerative
        
        self.model = QFormerImprovedGenerative(
            sequence_size=hyperparams["sequence_size"],
            qformer_hidden_size=hyperparams["qformer_hidden_size"],
            blocks_num=hyperparams["blocks_num"],
            num_heads=hyperparams["num_heads"],
            num_object_queries=hyperparams.get("num_object_queries", 32),
            num_relation_queries=hyperparams.get("num_relation_queries", 64),
            num_global_queries=hyperparams.get("num_global_queries", 32),
            num_reasoning_hops=hyperparams.get("num_reasoning_hops", 4),
            num_relation_types=hyperparams.get("num_relation_types", 16),
            max_answer_length=hyperparams.get("max_answer_length", 10),
            dropout_rate=hyperparams["dropout_rate"],
            use_clip_for_text=hyperparams["use_clip_for_text"],
            unfreeze_layers=hyperparams["unfreeze_layers"],
            device=device
        )

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if isinstance(batch, dict):
            for key, value in batch.items():
                if isinstance(value, dict):
                    batch[key] = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                  for k, v in value.items()}
                elif isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
        return batch

    def forward(self, samples: dict):
        return self.model(samples)
    
    def _common_step(self, batch: dict, task: str):
        output = self.forward(batch)

        self.log(f"{task}_answer_accuracy", output['answer_accuracy'], prog_bar=True, 
                 on_step=True, on_epoch=True, batch_size=self.hyperparams['batch_size'])
        self.log(f"{task}_loss_generation", output['loss_generation'], prog_bar=True,
                 on_step=True, on_epoch=True, batch_size=self.hyperparams['batch_size'])
        self.log(f"{task}_loss_itc", output['loss_itc'], 
                 on_step=True, on_epoch=True, batch_size=self.hyperparams['batch_size'])
        self.log(f"{task}_loss_itm", output['loss_itm'],
                 on_step=True, on_epoch=True, batch_size=self.hyperparams['batch_size'])
        self.log(f"{task}_loss_object", output['loss_object'],
                 on_step=True, on_epoch=True, batch_size=self.hyperparams['batch_size'])
        self.log(f"{task}_loss_relation", output['loss_relation'],
                 on_step=True, on_epoch=True, batch_size=self.hyperparams['batch_size'])
        self.log(f"{task}_total_loss", output['total_loss'], prog_bar=True,
                 on_step=True, on_epoch=True, batch_size=self.hyperparams['batch_size'])
        
        # Log examples
        if task == "val" and self.global_step % 100 == 0:
            for i in range(min(3, len(output['generated_answers']))):
                logger.info(f"Q: {batch['question'][i]}")
                logger.info(f"GT: {output['ground_truth_answers'][i]}")
                logger.info(f"Gen: {output['generated_answers'][i]}")
                logger.info("---")

        return output
    
    def training_step(self, batch: dict):
        output = self._common_step(batch, task="train")
        if torch.isnan(output['total_loss']) or torch.isinf(output['total_loss']):
            return None
        return output['total_loss']
    
    def validation_step(self, batch: dict):
        return self._common_step(batch, task="val")
    
    def test_step(self, batch: dict):
        return self._common_step(batch, task="test")
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hyperparams['lr'],
            betas=self.hyperparams['betas'],
            weight_decay=self.hyperparams['weight_decay'],
            eps=self.hyperparams['eps']
        )
        
        def lr_lambda(step: int):
            warmup = self.hyperparams.get('warmup_steps', 500)
            if step < warmup:
                return float(step) / float(max(1, warmup))
            return 1.0
        
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
    
    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        self.clip_gradients(optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm")

