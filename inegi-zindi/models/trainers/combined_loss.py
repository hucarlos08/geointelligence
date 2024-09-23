import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .center_loss import CenterLoss
from .arcface_loss import ArcFaceLoss
from .focal_loss import FocalLoss

class CombinedLoss(nn.Module):
    """
    Combines ArcFace, Center Loss, and Focal Loss for unbalanced classification tasks.
    
    Args:
        feat_dim (int): The dimension of the input features.
        num_classes (int): The number of classes.
        arcface_margin (float, optional): Angular margin for ArcFace. Default is 0.5.
        arcface_scale (float, optional): Scaling factor for ArcFace logits. Default is 64.
        center_loss_weight (float, optional): Weight for the center loss term. Default is 0.003.
        gamma (float, optional): Focusing parameter for Focal Loss. Default is 2.
    """
    def __init__(self, feat_dim, num_classes, arcface_margin=0.5, arcface_scale=64, center_loss_weight=0.003, alpha=0.5, gamma=2.0):
        super(CombinedLoss, self).__init__()
        self.arcface_loss = ArcFaceLoss(feat_dim, margin=arcface_margin, scale=arcface_scale)
        self.center_loss  = CenterLoss(num_classes=2, feat_dim=feat_dim, lambda_c=center_loss_weight)
        self.focal_loss   = FocalLoss(alpha=alpha, gamma=gamma)

    def forward(self, logits, features, labels):

        # ArcFace Loss
        #arcface_loss = self.arcface_loss(features, labels)
        
        # Center Loss
        center_loss = self.center_loss(features, labels)
        
        # Focal Loss (apply on logits)
        focal_loss = self.focal_loss(logits, labels)
        
        # Combine the losses (you can adjust the weights accordingly)
        total_loss = center_loss + focal_loss

        # Combine the losses (you can adjust the weights accordingly)
        #total_loss = arcface_loss + center_loss + focal_loss
        
        return total_loss,  center_loss, focal_loss
        #return total_loss, arcface_loss, center_loss, focal_loss

    @classmethod
    def from_config(cls, config):
        """
        Create an instance of the CombinedLoss from a configuration dictionary.
        
        Args:
            config (dict): A dictionary containing the configuration parameters for the CombinedLoss.
        
        Returns:
            CombinedLoss: An instance of the CombinedLoss class.
        """
        # Extract the configuration parameters
        feat_dim = config['feat_dim']
        num_classes = config['num_classes']
        arcface_margin = config.get('arcface_margin', 0.5)
        arcface_scale = config.get('arcface_scale', 64)
        center_loss_weight = config.get('center_loss_weight', 0.003)
        alpha = config.get('alpha', 0.75)
        gamma = config.get('gamma', 2.0)
        
        return cls(**config)
