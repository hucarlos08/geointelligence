import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFaceLoss(nn.Module):
    def __init__(self, feat_dim, scale=30.0, margin=0.50):
        super(ArcFaceLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.Tensor(2, feat_dim))  # 2 classes for binary classification
        nn.init.xavier_uniform_(self.weight)  # Initialize weights

    def forward(self, features, labels):

        labels = labels.long()

        # Normalize features and weights
        features = F.normalize(features, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity between features and class weights
        cosine = torch.matmul(features, weight_norm.t())  # [batch, 2] -> cosine similarity between features and weights

        # Add the margin to the target class logits
        target_logit = cosine.gather(1, labels.view(-1, 1)).squeeze(1)
        margin_cosine = torch.cos(torch.acos(target_logit) + self.margin)
 
        # Construct the final logits with margin applied to the correct class
        cosine = cosine.scatter(1, labels.view(-1, 1), margin_cosine.view(-1, 1))

        # Apply the scaling factor
        logits = self.scale * cosine  # shape: [batch_size, 2]

        # Transform binary labels to two-class format
        labels = labels.view(-1)

        #Transform binary labels to two-class format
        two_class_labels = F.one_hot(labels.long().squeeze(), num_classes=2)

        # Compute binary cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(logits, two_class_labels.float())

        return loss
