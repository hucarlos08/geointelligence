import torch
import torch.nn as nn
import torch.nn.functional as F

class CenterLoss(nn.Module):
    """
    CenterLoss module calculates the center loss between features and their corresponding labels.

    Args:
        num_classes (int): The number of classes.
        feat_dim (int): The dimension of the input features.
        lambda_c (float, optional): The weight for the center loss. Defaults to 0.003.

    """

    def __init__(self, num_classes, feat_dim, lambda_c=0.003):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        # Initialize centers more carefully, e.g., with normal distribution
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim) * 0.01)  
        self.lambda_c = lambda_c

    def forward(self, features, labels):
        """
        Forward pass of the CenterLoss module.

        Args:
            features (torch.Tensor): The input features of shape (batch_size, feat_dim).
            labels (torch.Tensor): The corresponding labels of shape (batch_size, 1).

        Returns:
            torch.Tensor: The center loss value.
        """
        # Ensure labels are 1D
        labels = labels.squeeze(1)

        # Ensure self.centers are on the same device as features
        # This modifies the underlying data of the tensor without changing the parameter itself
        self.centers.data = self.centers.data.to(features.device)

        # Select the centers for the current batch of labels
        centers_batch = self.centers.index_select(0, labels.long())

        # Calculate center loss
        center_loss = (features - centers_batch).pow(2).mean() / 2.0

        return self.lambda_c * center_loss