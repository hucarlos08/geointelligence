import torch
import torch.nn as nn
import torch.nn.functional as F

class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2, reduction: str = "mean"):
        """
        Focal Loss used in RetinaNet for dense detection tasks.
        
        Args:
            alpha (float): Weighting factor in range (0,1) to balance positive vs negative examples.
            gamma (float): Exponent of the modulating factor to balance easy vs hard examples.
            reduction (str): 'none', 'mean', or 'sum' to specify the reduction type.
        """
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Sigmoid Focal Loss.
        
        Args:
            inputs (torch.Tensor): Predictions (logits) for each example.
            targets (torch.Tensor): Binary classification labels (0 for negative, 1 for positive).
        
        Returns:
            torch.Tensor: Computed loss value based on reduction type.
        """
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)  # Correct class probability
        loss = ce_loss * ((1 - p_t) ** self.gamma)  # Focal loss modulation

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss  # Apply alpha weighting

        # Apply reduction
        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")