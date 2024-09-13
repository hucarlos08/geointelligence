import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score, JaccardIndex, AUROC

from ..utils import get_optimizer, get_lr_scheduler

class BasicTrainer(pl.LightningModule):
    def __init__(self, model, loss, optimizer_config: dict, scheduler_config: dict):
        super().__init__()
        
        self.model = model
        
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config

        self.loss = loss

        # Metrics
        self.accuracy = Accuracy(task="binary")
        self.precision = Precision(task="binary")
        self.recall = Recall(task="binary")
        self.f1 = F1Score(task="binary")
        self.aucroc = AUROC(task="binary")

    def _compute_metrics(self, preds, masks):
        return {
            'accuracy': self.accuracy(preds, masks.int()),
            'precision': self.precision(preds, masks.int()),
            'recall': self.recall(preds, masks.int()),
            'f1': self.f1(preds, masks.int()),
            'aucroc': self.aucroc(preds, masks.int())
        }

    def _log_metrics(self, metrics, stage, batch_size):
        for name, value in metrics.items():
            self.log(f"{stage}_{name}", value, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)

    def forward(self, x):
        return self.model(x)

    def step(self, batch, stage):
        images, masks  = batch
        logits = self.forward(images)
        loss = self.loss(logits, masks)
        
        preds = torch.sigmoid(logits)
        metrics = self._compute_metrics(preds, masks)
        
        # Log the metrics using the helper function
        self._log_metrics(metrics, stage, batch_size=images.size(0))
        self.log(f'{stage}_loss', loss, prog_bar=True, on_epoch=True, logger=True, batch_size=images.size(0))
        
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self.step(batch, 'test')

    def configure_optimizers(self):
        optimizer = get_optimizer(self.parameters(), self.optimizer_config)
        scheduler = get_lr_scheduler(optimizer, self.scheduler_config)
        return [optimizer], [scheduler]