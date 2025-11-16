import pytorch_lightning as pl
from torch.optim import AdamW
from model.q_former_base import QFormerBase
import torch

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
        return output['total_loss']
    
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

        return optimizer
    
    def save_checkpoint(self, file_path: str):
        self.trainer.save_checkpoint(file_path)