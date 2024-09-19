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
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
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
        # Ensure labels are 1D by squeezing the second dimension
        labels = labels.squeeze(1)

        # Select the centers for the current batch of labels
        centers_batch = self.centers.index_select(0, labels.long())  # Ensure labels are long (integer type)

        # Calculate center loss
        center_loss = (features - centers_batch).pow(2).sum() / 2.0 / features.size(0)

        return self.lambda_c * center_loss
