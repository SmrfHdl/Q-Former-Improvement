import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from model.q_former_base import QFormerBase
import torch
import wandb

from loguru import logger

class QFormerBaseLightning(pl.LightningModule):
    def __init__(self,
                 hyperparams: dict,
                 device: torch.device):
        super().__init__()
        self.save_hyperparameters()

        self.hyperparams = hyperparams

        self.q_former_base = QFormerBase(
            sequence_size=hyperparams["sequence_size"],
            qformer_hidden_size=hyperparams["qformer_hidden_size"],
            blocks_num=self.hyperparams["blocks_num"],
            num_heads=hyperparams["num_heads"],
            num_queries=hyperparams["num_queries"],
            dropout_rate=hyperparams["dropout_rate"],
            use_clip_for_text=hyperparams["use_clip_for_text"],
            unfreeze_layers=hyperparams["unfreeze_layers"],
            device=device
        )
        
        # Debug: Log trainable parameters
        self._log_trainable_params()
    
    def _log_trainable_params(self):
        """Log which parameters are trainable."""
        total_params = 0
        trainable_params = 0
        
        logger.info("=" * 60)
        logger.info("TRAINABLE PARAMETERS DEBUG:")
        
        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                if 'learned_queries' in name or 'projection' in name or 'head' in name:
                    logger.info(f"{name}: {param.shape}, requires_grad=True")
        
        logger.info(f"Total params: {total_params:,}")
        logger.info(f"Trainable params: {trainable_params:,}")
        logger.info(f"Trainable ratio: {100*trainable_params/total_params:.2f}%")
        logger.info("=" * 60)

    def forward(self, samples: dict):
        return self.q_former_base(samples)
    
    def _common_step(self, batch: dict, task: str):
        output = self.forward(batch)

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
        self.log(f"{task}_total_loss", output['total_loss'], prog_bar=True, on_step=True, on_epoch=True, logger=True,
                 batch_size=self.hyperparams['batch_size'])
        
        logs = ""
        logs += f"{task}_answer_accuracy: {output['answer_accuracy'].item():.4f}, "
        logs += f"{task}_loss_itc: {output['loss_itc'].item():.4f}, "
        logs += f"{task}_loss_igt: {output['loss_igt'].item():.4f}, "
        logs += f"{task}_loss_itm: {output['loss_itm'].item():.4f}, "
        logs += f"{task}_loss_answer: {output['loss_answer'].item():.4f}, "
        logs += f"{task}_total_loss: {output['total_loss'].item():.4f}"
        
        logger.info(logs)

        return output
    
    def training_step(self, batch: dict):
        output = self._common_step(batch, task="train")
        total_loss = output['total_loss']
        
        # Skip backward if loss is NaN or Inf to prevent weight corruption
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.warning(f"NaN/Inf detected in total_loss, skipping backward pass")
            # Return None to skip this batch completely
            return None
        
        return total_loss
    
    def on_after_backward(self):
        """Check for NaN/Inf in weights and gradients after backward pass."""
        # Debug: Log gradient statistics for key parameters
        if self.global_step % 50 == 0:  # Log every 50 steps
            grad_info = []
            
            # Check learned_queries gradient
            if self.q_former_base.learned_queries.grad is not None:
                q_grad = self.q_former_base.learned_queries.grad
                grad_info.append(f"queries_grad: mean={q_grad.mean().item():.6f}, max={q_grad.abs().max().item():.6f}")
            else:
                grad_info.append("queries_grad: None!")
            
            # Check vision_projection gradient
            if self.q_former_base.vision_projection.weight.grad is not None:
                v_grad = self.q_former_base.vision_projection.weight.grad
                grad_info.append(f"vision_proj_grad: mean={v_grad.mean().item():.6f}, max={v_grad.abs().max().item():.6f}")
            else:
                grad_info.append("vision_proj_grad: None!")
                
            # Check answer_head gradient
            if self.q_former_base.answer_head.weight.grad is not None:
                a_grad = self.q_former_base.answer_head.weight.grad
                grad_info.append(f"answer_head_grad: mean={a_grad.mean().item():.6f}, max={a_grad.abs().max().item():.6f}")
            else:
                grad_info.append("answer_head_grad: None!")
            
            # Check cross_modal_transformer gradient (first layer)
            first_layer = self.q_former_base.cross_modal_transformer.layers[0]
            if first_layer.mhca.linear_q.weight.grad is not None:
                cm_grad = first_layer.mhca.linear_q.weight.grad
                grad_info.append(f"cross_attn_grad: mean={cm_grad.mean().item():.6f}, max={cm_grad.abs().max().item():.6f}")
            else:
                grad_info.append("cross_attn_grad: None!")
                
            logger.info(f"[GRAD DEBUG Step {self.global_step}] " + " | ".join(grad_info))
        
        # Check projection layer weights for corruption
        vision_proj_weight = self.q_former_base.vision_projection.weight
        text_proj_weight = self.q_former_base.text_projection.weight
        
        if torch.isnan(vision_proj_weight).any() or torch.isinf(vision_proj_weight).any():
            logger.error("CRITICAL: NaN/Inf detected in vision_projection weights after backward!")
            self.q_former_base.zero_grad()
            
        if torch.isnan(text_proj_weight).any() or torch.isinf(text_proj_weight).any():
            logger.error("CRITICAL: NaN/Inf detected in text_projection weights after backward!")
            self.q_former_base.zero_grad()
    
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
        # Clip gradients by norm (max_norm=1.0 is quite aggressive but safe)
        self.clip_gradients(
            optimizer,
            gradient_clip_val=1.0,
            gradient_clip_algorithm="norm"
        )
    
    def save_checkpoint(self, file_path: str):
        self.trainer.save_checkpoint(file_path)