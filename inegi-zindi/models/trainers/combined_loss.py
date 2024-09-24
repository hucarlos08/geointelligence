import torch
import torch.nn as nn
import torch.nn.functional as F

from .center_loss import CenterLoss
from .arcface_loss import ArcFaceLoss
from .focal_loss import FocalLoss

class CombinedLoss(nn.Module):
    """
    Combines multiple loss functions for unbalanced classification tasks.
    
    Args:
        config (dict): A dictionary containing the configuration for all loss functions.
        # Example usage:
        config = {
            'arcface': {
                'params': {'feat_dim': 512, 'margin': 0.5, 'scale': 64},
                'weight': 1.0
            },
            'center': {
                'params': {'num_classes': 2, 'feat_dim': 512, 'lambda_c': 0.003},
                'weight': 0.5
            },
            'focal': {
                'params': {'alpha': 0.75, 'gamma': 2.0},
                'weight': 1.0
            }
        }

        combined_loss = CombinedLoss.from_config(config)
    """
    def __init__(self, config):
        super(CombinedLoss, self).__init__()
        self.config = config
        self.losses = nn.ModuleDict()
        self.weights = {}

        if 'arcface' in config:
            self.losses['arcface'] = ArcFaceLoss(**config['arcface']['params'])
            self.weights['arcface'] = config['arcface'].get('weight', 1.0)

        if 'center' in config:
            self.losses['center'] = CenterLoss(**config['center']['params'])
            self.weights['center'] = config['center'].get('weight', 1.0)

        if 'focal' in config:
            self.losses['focal'] = FocalLoss(**config['focal']['params'])
            self.weights['focal'] = config['focal'].get('weight', 1.0)

    def forward(self, logits, features, labels):
        total_loss = 0
        loss_components = {}

        for name, loss_fn in self.losses.items():
            if name == 'arcface':
                loss = loss_fn(features, labels)
            elif name == 'center':
                loss = loss_fn(features, labels)
            elif name == 'focal':
                loss = loss_fn(logits, labels)
            else:
                raise ValueError(f"Unknown loss function: {name}")

            weighted_loss = self.weights[name] * loss
            total_loss += weighted_loss
            loss_components[name] = loss.item()

        return total_loss, loss_components

    @classmethod
    def from_config(cls, config):
        """
        Create an instance of the CombinedLoss from a configuration dictionary.
        
        Args:
            config (dict): A dictionary containing the configuration for all loss functions.
        
        Returns:
            CombinedLoss: An instance of the CombinedLoss class.
        """
        return cls(config)

