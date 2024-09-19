import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFaceLoss(nn.Module):
    """
    ArcFace Loss (Additive Angular Margin Loss) introduces an angular margin between classes
    to enhance the inter-class separability of features.
    
    Args:
        feat_dim (int): The dimension of the input features.
        num_classes (int): The number of classes.
        margin (float, optional): The angular margin to be added between features and class centers. Default is 0.5.
        scale (float, optional): A scaling factor for the logits to adjust the impact of the margin. Default is 64.
    """
    def __init__(self, feat_dim, num_classes, margin=0.5, scale=64):
        super(ArcFaceLoss, self).__init__()
        self.feat_dim = feat_dim  # Dimensionality of the input features
        self.num_classes = num_classes  # Number of classes
        self.margin = margin  # Angular margin
        self.scale = scale  # Scaling factor to amplify the margin's effect
        
        # Weight matrix representing the class centers in the feature space
        self.weight = nn.Parameter(torch.Tensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)  # Initialize the class center weights
        
    def forward(self, features, labels):
        """
        Forward pass of ArcFace Loss.
        
        Args:
            features (torch.Tensor): The input features of shape (batch_size, feat_dim).
            labels (torch.Tensor): The ground truth labels of shape (batch_size,).
        
        Returns:
            torch.Tensor: The computed ArcFace loss.
        """
        # Normalize the features and weight to ensure they lie on the unit hypersphere
        normalized_features = F.normalize(features, p=2, dim=1)
        normalized_weight = F.normalize(self.weight, p=2, dim=1)
        
        # Compute the cosine similarity between the features and the weight (class centers)
        cosine = F.linear(normalized_features, normalized_weight)  # Shape: (batch_size, num_classes)
        
        # Retrieve the cosine similarity for the correct classes using the labels
        theta = cosine.acos()  # Get the angle for the current batch
        target_logits = torch.gather(cosine, 1, labels.view(-1, 1)).squeeze(1)  # Get cosine for the true labels
        
        # Apply the angular margin (add margin to the angle)
        target_logits = torch.cos(theta + self.margin)  # Apply margin to the correct class only
        
        # Create logits with the margin for the correct class and the original logits for the others
        one_hot = torch.zeros_like(cosine)  # Shape: (batch_size, num_classes)
        one_hot.scatter_(1, labels.view(-1, 1), 1)  # One-hot encode the labels
        logits = one_hot * target_logits.unsqueeze(1) + (1 - one_hot) * cosine  # Only apply margin to the correct class
        
        # Apply the scaling factor
        logits = logits * self.scale
        
        # Return cross-entropy loss with the modified logits
        loss = F.cross_entropy(logits, labels)
        
        return loss
