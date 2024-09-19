import torch
import torch.nn as nn

class SimpleClassifier(nn.Module):
    def __init__(self, input_channels=6, image_size=16, num_classes=10):
        super(SimpleClassifier, self).__init__()
        # El tamaño de la imagen aplanada será input_channels * image_size * image_size
        input_size = input_channels * image_size * image_size
        
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Aplanar la entrada desde (batch_size, C, H, W) a (batch_size, C*H*W)
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No aplicamos softmax aquí, lo haremos en la función de pérdida
        return x

    @classmethod
    def from_config(cls, config):
        """Create an instance of SimpleClassifier from a config dictionary.
        
        Parameters:
        config (dict): Dictionary containing the parameters for the classifier.
                       Expected keys are 'input_channels', 'image_size', and 'num_classes'.
        
        Returns:
        SimpleClassifier: A new instance of the classifier with the given configuration.
        """
        input_channels = config.get('input_channels', 6)  # Default 6 channels
        image_size = config.get('image_size', 16)         # Default image size 16x16
        num_classes = config.get('num_classes', 10)       # Default 10 classes

        return cls(input_channels=input_channels, image_size=image_size, num_classes=num_classes)
 