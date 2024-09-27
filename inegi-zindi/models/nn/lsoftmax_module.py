import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LSoftmaxBinary(nn.Module):
    def __init__(self, in_features, margin=8, scale=30.0):
        """
        Large-margin softmax for binary classification with single logit output.
        Args:
            in_features (int): Size of input features (dimension of embeddings).
            margin (int): Angular margin to apply (default is 4).
            scale (float): Scaling factor for logits (default is 30.0).
        """
        super(LSoftmaxBinary, self).__init__()
        self.in_features = in_features
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(1, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, labels=None):
        """
        Forward pass with angular margin.
        Args:
            x (Tensor): Input embeddings (shape [batch_size, in_features]).
            labels (Tensor, optional): Labels for the current batch (shape [batch_size, 1]).
        
        Returns:
            logits (Tensor): Single logit output for positive class [batch_size, 1].
        """
        # Normalize the input and weight vectors (L2 normalization)
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        
        # Compute the cosine of the angle between the feature vector and the weight vector
        cos_theta = F.linear(x_norm, w_norm)  # Shape: [batch_size, 1]
        
        # Clamp for numerical stability
        cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)

        # If no labels are provided, return the scaled cosine similarity
        if labels is None:
            return self.scale * cos_theta

        # Compute theta and apply margin
        theta = torch.acos(cos_theta)
        target_theta = torch.where(labels == 1, theta * self.margin, theta)
        cos_target_theta = torch.cos(target_theta)

        # Compute the difference
        logits = self.scale * (cos_target_theta - cos_theta)
        
        # Add back the original cosine similarity
        logits += self.scale * cos_theta

        return logits
