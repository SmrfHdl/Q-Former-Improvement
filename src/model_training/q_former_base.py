import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from model.q_former_base import QFormerBase
import torch
import wandb
import math

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
            device=device,
            # v2 improvements - pass new hyperparameters
            learnable_temperature=hyperparams.get("learnable_temperature", True),
            initial_temperature=hyperparams.get("initial_temperature", 0.07),
            temperature_min=hyperparams.get("temperature_min", 0.01),
            temperature_max=hyperparams.get("temperature_max", 0.5),
            label_smoothing_answer=hyperparams.get("label_smoothing_answer", 0.1),
            label_smoothing_itc=hyperparams.get("label_smoothing_itc", 0.1),
            label_smoothing_itm=hyperparams.get("label_smoothing_itm", 0.1),
            loss_weight_itc=hyperparams.get("loss_weight_itc", 0.2),
            loss_weight_itm=hyperparams.get("loss_weight_itm", 0.3),
            loss_weight_igt=hyperparams.get("loss_weight_igt", 0.0),
            loss_weight_answer=hyperparams.get("loss_weight_answer", 1.0),
            stochastic_depth_rate=hyperparams.get("stochastic_depth_rate", 0.1)
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
        return self.q_former_base(samples)
    
    def _common_step(self, batch: dict, task: str):
        output = self.forward(batch)

        self.log(f"{task}_answer_accuracy", output['answer_accuracy'], prog_bar=True, on_step=True, on_epoch=True, logger=True,
                 batch_size=self.hyperparams['batch_size'])
        self.log(f"{task}_itm_accuracy", output['itm_accuracy'], prog_bar=True, on_step=True, on_epoch=True, logger=True,
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
        
        # v2 IMPROVEMENT: Log temperature if available
        if 'temperature' in output:
            self.log(f"{task}_temperature", output['temperature'], prog_bar=False, on_step=True, on_epoch=True, logger=True,
                     batch_size=self.hyperparams['batch_size'])
        
        logs = ""
        logs += f"{task}_answer_accuracy: {output['answer_accuracy'].item():.4f}, "
        logs += f"{task}_loss_itc: {output['loss_itc'].item():.4f}, "
        logs += f"{task}_loss_igt: {output['loss_igt'].item():.4f}, "
        logs += f"{task}_loss_itm: {output['loss_itm'].item():.4f}, "
        logs += f"{task}_loss_answer: {output['loss_answer'].item():.4f}, "
        logs += f"{task}_total_loss: {output['total_loss'].item():.4f}"
        
        # Log temperature if available
        if 'temperature' in output:
            logs += f", temp: {output['temperature'].item():.4f}"
        
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
                
            # Check answer_head gradient (last linear layer in Sequential)
            answer_head_last_linear = self.q_former_base.answer_head[-1]  # Last layer is Linear
            if answer_head_last_linear.weight.grad is not None:
                a_grad = answer_head_last_linear.weight.grad
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
            betas=tuple(self.hyperparams['betas']),
            weight_decay=self.hyperparams['weight_decay'],
            eps=self.hyperparams['eps']
        )
        
        warmup_steps = self.hyperparams.get('warmup_steps', 200)
        use_cosine_scheduler = self.hyperparams.get('use_cosine_scheduler', True)
        min_lr_ratio = self.hyperparams.get('min_lr_ratio', 0.01)
        
        # v2 IMPROVEMENT: Cosine annealing scheduler with warmup
        if use_cosine_scheduler:
            # Linear warmup
            def warmup_lr_lambda(current_step: int):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return 1.0
            
            warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)
            
            # Cosine annealing after warmup
            # Estimate total steps (will be adjusted during training)
            num_epochs = self.hyperparams.get('num_epochs', 50)
            # Approximate steps per epoch (will be updated by trainer)
            estimated_steps_per_epoch = 100  # Conservative estimate
            total_steps = num_epochs * estimated_steps_per_epoch
            
            # Cosine scheduler with minimum LR
            cosine_scheduler = CosineAnnealingLR(
                optimizer, 
                T_max=max(total_steps - warmup_steps, 1),
                eta_min=self.hyperparams['lr'] * min_lr_ratio
            )
            
            # Combine warmup and cosine schedulers
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]
            )
            
            logger.info(f"Using cosine scheduler with warmup_steps={warmup_steps}, min_lr_ratio={min_lr_ratio}")
        else:
            # Fallback to warmup-only scheduler
            def lr_lambda(current_step: int):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return 1.0
            
            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
            logger.info(f"Using warmup-only scheduler with warmup_steps={warmup_steps}")
        
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