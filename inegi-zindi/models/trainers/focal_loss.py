
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss is a variant of the Cross Entropy Loss that is designed to address the problem of class imbalance.
    It assigns higher weights to hard, misclassified examples and lower weights to easy, correctly classified examples.
    This helps in focusing the training on difficult examples and improving the model's performance.

    Args:
        alpha (float): The alpha parameter controls the balance between the positive and negative class samples.
            A value closer to 1 gives more weight to the positive class, while a value closer to 0 gives more weight to the negative class.
            Default is 0.25.
        gamma (float): The gamma parameter controls the rate at which the importance of hard examples is increased.
            A higher value of gamma puts more emphasis on hard examples.
            Default is 2.

    Attributes:
        alpha (float): The alpha parameter.
        gamma (float): The gamma parameter.
    """

    def __init__(self, alpha=0.25, gamma=2, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass of the Focal Loss.

        Args:
            inputs (torch.Tensor): The input tensor of shape (batch_size, num_classes).
                It represents the predicted logits for each class.
            targets (torch.Tensor): The target tensor of shape (batch_size, num_classes).
                It represents the ground truth labels for each class.

        Returns:
            torch.Tensor: The computed Focal Loss value.
        """
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets.float())
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        # Apply reduction
        if self.reduction == "none":
            return F_loss
        elif self.reduction == "mean":
            return F_loss.mean()
        elif self.reduction == "sum":
            return F_loss.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")

    @classmethod
    def from_config(cls, config):
        """
        Create an instance of the FocalLoss from a configuration dictionary.

        Args:
            config (dict): A dictionary containing the configuration parameters for the FocalLoss.

        Returns:
            FocalLoss: An instance of the FocalLoss created using the configuration.
        """
        return cls(alpha=config.get('alpha', 0.25), gamma=config.get('gamma', 2), reduction=config.get('reduction', 'mean'))    