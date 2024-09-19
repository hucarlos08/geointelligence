import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score, JaccardIndex, AUROC

from ..utils import get_optimizer, get_lr_scheduler
from .basic_trainer import BasicTrainer

class FeatureAwareTrainer(BasicTrainer):
    def __init__(self, model, loss, feature_loss, optimizer_config: dict, scheduler_config: dict):
        super().__init__(model, loss, optimizer_config, scheduler_config)

        self.feature_loss = feature_loss

    # Override the step function to handle both logits and features
    def step(self, batch, stage):
        images, masks = batch

        # Forward pass now returns both logits and features
        logits, features = self.forward(images, return_features=True)

        # Compute the main classification loss (e.g., BCEWithLogitsLoss)
        loss = self.loss(logits, masks)

        # Here, you can compute the feature loss or other losses that depend on features
        feature_loss = self.feature_loss(features, masks)

        # Add the feature loss to the main loss
        loss += feature_loss

        self.log(f'{stage}_loss', loss, prog_bar=True, on_epoch=True, logger=True, batch_size=images.size(0))
        self.log(f'{stage}_feature_loss', feature_loss, prog_bar=True, on_epoch=True, logger=True, batch_size=images.size(0))

        preds = torch.sigmoid(logits)  # For binary classification
        metrics = self._compute_metrics(preds, masks)

        # Log the metrics using the helper function
        self._log_metrics(metrics, stage, batch_size=images.size(0))

        return loss

    def forward(self, x, return_features=True):
        return self.model(x, return_features=return_features)

    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self.step(batch, 'test')
