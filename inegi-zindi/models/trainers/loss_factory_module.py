import torch
import torch.nn as nn

from .focal_loss import FocalLoss
from .center_loss import CenterLoss
from .sigmoid_focal_loss import SigmoidFocalLoss

class LossCompose:
    def __init__(self, losses):
        """
        Initialize with a list of loss functions.
        
        Args:
            losses (list): A list of loss functions.
        """
        self.losses = losses

    def __call__(self, inputs, targets):
        """
        Compute the total loss by summing all individual losses.
        
        Args:
            inputs (torch.Tensor): The input tensor (e.g., predictions).
            targets (torch.Tensor): The target tensor (e.g., ground truth).
        
        Returns:
            torch.Tensor: The total loss.
        """
        total_loss = 0
        for loss in self.losses:
            total_loss += loss(inputs, targets)
        return total_loss

class LossFactory:
    @staticmethod
    def _create_loss(loss_config):
        """
        Create a composition of loss functions based on the configuration.
        
        Args:
            loss_config (dict): A dictionary of loss configurations.
        
        Returns:
            loss (callable): A composed loss function.
        """
        loss_list = []
        for loss_name, params in loss_config.items():
            if params is None:
                params = {}  # Use an empty dictionary if no parameters are provided
            
            if loss_name == 'FocalLoss':
                loss_list.append(FocalLoss(**params))
            elif loss_name == 'CenterLoss':
                loss_list.append(CenterLoss(**params))
            else:
                # Dynamically retrieve the loss class from torch.nn and instantiate it
                try:
                    loss_class = getattr(nn, loss_name)
                    loss_list.append(loss_class(**params))
                except AttributeError:
                    raise ValueError(f"Loss {loss_name} not found in torch.nn or is not defined.")
        
        # Use LossCompose to aggregate the losses instead of lambda
        return LossCompose(loss_list) if len(loss_list) > 1 else loss_list[0]

    @staticmethod
    def from_config(loss_config):
        """
        Create a loss function based on the configuration.
        
        Args:
            loss_config (dict): A dictionary of loss configurations.
        
        Returns:
            loss (callable): A composed loss function.
        """
        return LossFactory._create_loss(loss_config)

