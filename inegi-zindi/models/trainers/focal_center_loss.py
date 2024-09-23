import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .center_loss import CenterLoss
from .focal_loss import FocalLoss

class FocalCenterLoss(nn.Module):
    """
    FocalCenterLoss is a custom loss function that combines Center Loss and Focal Loss.
    It is designed for multi-class classification tasks.
    
    Args:
        feat_dim (int): The dimension of the input features.
        num_classes (int): The number of classes in the classification task.
        center_loss_weight (float, optional): The weight for the center loss. Default is 0.003.
        alpha (float, optional): The alpha parameter for the focal loss. Default is 0.5.
        gamma (float, optional): The gamma parameter for the focal loss. Default is 2.0.
    """
    
    def __init__(self, feat_dim, num_classes, center_loss_weight=0.003, alpha=0.5, gamma=2.0):
        super(FocalCenterLoss, self).__init__()
        self.center_loss  = CenterLoss(num_classes=2, feat_dim=feat_dim, lambda_c=center_loss_weight)
        self.focal_loss   = FocalLoss(alpha=alpha, gamma=gamma)

    def forward(self, logits, features, labels):
        """
        Forward pass of the FocalCenterLoss.
        
        Args:
            logits (torch.Tensor): The predicted logits from the model.
            features (torch.Tensor): The input features.
            labels (torch.Tensor): The ground truth labels.
        
        Returns:
            tuple: A tuple containing the total loss, center loss, and focal loss.
        """
        
        # Center Loss
        center_loss = self.center_loss(features, labels)
        
        # Focal Loss (apply on logits)
        focal_loss = self.focal_loss(logits, labels)
        
        # Combine the losses (you can adjust the weights accordingly)
        total_loss = center_loss + focal_loss
        
        return total_loss,  center_loss, focal_loss

    @classmethod
    def from_config(cls, config):
        """
        Create an instance of FocalCenterLoss from a configuration dictionary.
        
        Args:
            config (dict): A dictionary containing the configuration parameters.
        
        Returns:
            FocalCenterLoss: An instance of FocalCenterLoss.
        """
        # Extract the configuration parameters
        feat_dim = config['feat_dim']
        num_classes = config['num_classes']
        center_loss_weight = config.get('center_loss_weight', 0.003)
        alpha = config.get('alpha', 0.75)
        gamma = config.get('gamma', 2.0)
        
        return cls(**config)
