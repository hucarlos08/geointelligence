import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        """
        Contrastive loss function.
        Args:
            margin (float): Margin for negative pairs. Default is 1.0.
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        """
        Forward pass of the contrastive loss.
        Args:
            output1 (Tensor): Embedding of the first input (shape: [batch_size, embedding_dim]).
            output2 (Tensor): Embedding of the second input (shape: [batch_size, embedding_dim]).
            label (Tensor): Binary label (shape: [batch_size]) where 0 = positive pair, 1 = negative pair.

        Returns:
            Tensor: Contrastive loss.
        """
        # Compute the Euclidean distance between the two embeddings
        euclidean_distance = F.pairwise_distance(output1, output2)
        
        # Calculate contrastive loss
        loss_contrastive = torch.mean(
            (1 - label) * 0.5 * torch.pow(euclidean_distance, 2) +  # Positive pair loss
            (label) * 0.5 * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)  # Negative pair loss
        )
        
        return loss_contrastive

if __name__ == '__main__':
    # Example usage
    output1 = torch.randn(4, 128)  # Example embeddings
    output2 = torch.randn(4, 128)  # Example embeddings
    label = torch.tensor([0, 1, 0, 1])  # Example labels (0 = positive pair, 1 = negative pair)

    contrastive_loss = ContrastiveLoss(margin=1.0)
    loss = contrastive_loss(output1, output2, label)
    print(loss)