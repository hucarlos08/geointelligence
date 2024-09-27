import torch
import torch.nn as nn

class CenterLoss(nn.Module):
    """
    CenterLoss module calculates the center loss between features and their corresponding labels.

    Args:
        num_classes (int): The number of classes.
        feat_dim (int): The dimension of the input features.
        lambda_c (float, optional): The weight for the center loss. Defaults to 0.003.
    """

    def __init__(self, num_classes, feat_dim, lambda_c=0.003, init_type="xavier"):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.lambda_c = lambda_c

        # Initialize centers more carefully, using either normal or Xavier/Glorot
        if init_type == "xavier":
            self.centers = nn.Parameter(torch.empty(num_classes, feat_dim))
            nn.init.xavier_normal_(self.centers)
        else:  # Default to normal initialization with small variance
            self.centers = nn.Parameter(torch.randn(num_classes, feat_dim) * 0.01)

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

        # Ensure centers are on the same device as features
        centers_batch = self.centers.index_select(0, labels.long()).to(features.device)

        # Calculate center loss, normalized by batch size
        center_loss = ((features - centers_batch).pow(2).sum(dim=1).mean()) / 2.0

        return self.lambda_c * center_loss

    @classmethod
    def from_config(cls, config):
        """
        Create an instance of the CenterLoss from a configuration dictionary.

        Args:
            config (dict): A dictionary containing the configuration parameters for the CenterLoss.

        Returns:
            CenterLoss: An instance of the CenterLoss class.
        """
        # Extract the configuration parameters
        num_classes = config.get('num_classes', 2)
        feat_dim = config.get('feat_dim', 256)
        lambda_c = config.get('lambda_c', 0.003)
        init_type = config.get('init_type', "normal")

        return cls(num_classes, feat_dim, lambda_c, init_type)
