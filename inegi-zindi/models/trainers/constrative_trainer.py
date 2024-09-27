import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score, JaccardIndex, AUROC

from ..utils import get_optimizer, get_lr_scheduler
from .basic_trainer import BasicTrainer

from ..losses import ContrastiveLoss

class ContrastiveTrainer(BasicTrainer):
    """
    A PyTorch Lightning trainer class for training a model with both logits and features.

    This class extends the BasicTrainer class and overrides the step function to handle both logits and features.
    It also provides methods for training, validation, and testing steps.

    Args:
        model (nn.Module): The model to be trained.
        combined_loss (callable): A function that computes the combined loss using logits, features, and labels.
        optimizer_config (dict): Configuration for the optimizer.
        scheduler_config (dict): Configuration for the learning rate scheduler.
    """

    def __init__(self, model, loss, optimizer_config: dict, scheduler_config: dict):
        super().__init__(model, loss, optimizer_config, scheduler_config)

        self.contrastive_loss = ContrastiveLoss()

    def step(self, batch, stage):
        """
        Perform a forward pass and compute the loss and metrics for a given batch.

        Args:
            batch (tuple): A tuple containing the input images and labels.
            stage (str): The stage of training (e.g., 'train', 'val', 'test').

        Returns:
            loss (torch.Tensor): The computed loss for the batch.
        """
        anchor_images, anchor_labels, pair_images, pair_labels, labels = batch

        # Forward pass now returns both logits and features
        anchor_logits, anchor_features  = self.forward(anchor_images, anchor_labels, return_features=True)
        pair_logits, pair_features      = self.forward(pair_images, pair_labels,  return_features=True)

        # Compute the combined loss
        loss, loss_components = self.loss(anchor_logits, anchor_features, anchor_labels)

        # Compute the constrative loss
        loss_contrastive = self.contrastive_loss(anchor_features, pair_features, labels)

        # Combine the losses
        loss += loss_contrastive

        # Batch size
        batch_size = anchor_images.size(0)
        
        # Log the total loss
        self.log(f'{stage}_loss', loss, prog_bar=True, on_epoch=True, logger=True, batch_size=batch_size)

        # Log the constrative loss
        self.log(f'{stage}_contrastive_loss', loss_contrastive, prog_bar=False, on_epoch=True, logger=True, batch_size=batch_size)

        # Log individual loss components
        for loss_name, loss_value in loss_components.items():
            self.log(f'{stage}_{loss_name}_loss', loss_value, prog_bar=False, on_epoch=True, logger=True, batch_size=batch_size)

        preds = torch.sigmoid(anchor_logits)  # For binary classification
        metrics = self._compute_metrics(preds, anchor_labels)

        # Log the metrics using the helper function
        self._log_metrics(metrics, stage, batch_size=batch_size)

        return loss

    def forward(self, x, labels=None, return_features=True):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            return_features (bool): Whether to return the features along with the logits.

        Returns:
            output (tuple): A tuple containing the logits and features (if return_features is True).
        """
        return self.model(x, labels, return_features=return_features)

    def training_step(self, batch, batch_idx):
        """
        Perform a training step for a given batch.

        Args:
            batch (tuple): A tuple containing the input images and labels.
            batch_idx (int): The index of the batch.

        Returns:
            loss (torch.Tensor): The computed loss for the batch.
        """
        return self.step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        """
        Perform a validation step for a given batch.

        Args:
            batch (tuple): A tuple containing the input images and labels.
            batch_idx (int): The index of the batch.

        Returns:
            loss (torch.Tensor): The computed loss for the batch.
        """
        return self.step(batch, 'val')

    def test_step(self, batch, batch_idx):
        """
        Perform a testing step for a given batch.

        Args:
            batch (tuple): A tuple containing the input images and labels.
            batch_idx (int): The index of the batch.

        Returns:
            loss (torch.Tensor): The computed loss for the batch.
        """
        return self.step(batch, 'test')
