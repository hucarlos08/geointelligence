import torch
import torch.nn as nn

class ConvEmbeddingClassifier(nn.Module):
    def __init__(self, input_channels=6, embedding_size=256, num_classes=1):
        super(ConvEmbeddingClassifier, self).__init__()
        
        # Convolutional layers to extract features
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)  # (16x16 -> 16x16)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # (16x16 -> 16x16)
        self.pool = nn.MaxPool2d(2, 2)  # (16x16 -> 8x8)
        
        # Fully connected layers to create an embedding
        self.fc1 = nn.Linear(64 * 8 * 8, embedding_size)  # Embedding of size 256
        self.fc2 = nn.Linear(embedding_size, num_classes)  # Final classification layer
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Apply convolutions
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        
        # Flatten the feature maps to create the embedding
        x = x.view(x.size(0), -1)  # Flatten
        
        # Pass through fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

    @classmethod
    def from_config(cls, config):
        """Create an instance of ConvEmbeddingClassifier from a config dictionary.
        
        Parameters:
        config (dict): Dictionary containing the parameters for the classifier.
                       Expected keys are 'input_channels', 'embedding_size', and 'num_classes'.
        
        Returns:
        ConvEmbeddingClassifier: A new instance of the classifier with the given configuration.
        """
        input_channels = config.get('input_channels', 6)  # Default 6 channels
        embedding_size = config.get('embedding_size', 256)  # Default embedding size 256
        num_classes = config.get('num_classes', 1)  # Default binary classification
        
        return cls(input_channels=input_channels, embedding_size=embedding_size, num_classes=num_classes)
