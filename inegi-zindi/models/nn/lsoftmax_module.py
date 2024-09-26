import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LSoftmaxBinary(nn.Module):
    def __init__(self, in_features, out_features=1, margin=4):
        """
        Large-margin softmax for binary classification.
        Args:
            in_features (int): Size of input features (dimension of embeddings).
            out_features (int): Number of output features (1 for binary classification).
            margin (int): Angular margin to apply (default is 4).
        """
        super(LSoftmaxBinary, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, labels=None):
        """
        Forward pass with angular margin.
        Args:
            x (Tensor): Input embeddings (shape [batch_size, in_features]).
            labels (Tensor, optional): Labels for the current batch (used to apply margin).
        
        Returns:
            logits (Tensor): Logits with angular margin applied to the positive class.
        """
        # Normalize the input and weight vectors (L2 normalization)
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        
        # Compute the cosine of the angle between the feature vector and the weight vector
        cos_theta = F.linear(x_norm, w_norm)  # Shape: [batch_size, out_features]

        # If no labels are provided, return the regular logits
        if labels is None:
            return cos_theta

        # Apply the angular margin to cos_theta for the positive class (label == 1)
        with torch.no_grad():
            theta = torch.acos(cos_theta.clamp(-1, 1))  # Angle between vectors
            margin_theta = self.margin * theta  # Apply the margin to the angle

        # Convert the adjusted angles back to cosines
        cos_margin_theta = torch.cos(margin_theta)

        # Apply the margin only where the label is positive (class 1)
        logits = torch.where(labels == 1, cos_margin_theta, cos_theta)

        # Scale by a constant (optional) to prevent small values in the logits
        logits *= x_norm.norm(dim=1, keepdim=True)

        return logits
