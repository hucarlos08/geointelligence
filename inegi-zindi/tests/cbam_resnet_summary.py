import torch
from torchinfo import summary
from models.nn import CBAMResNet  # Assuming this is the name of your module

def visualize_model_layers(input_shape=(1024, 6, 16, 16), **model_params):
    # Create an instance of the model
    model = CBAMResNet(**model_params)
    
    # Generate the summary
    model_summary = summary(model, input_size=input_shape, depth=2, col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"])


if __name__ == "__main__":
    # Example usage
    visualize_model_layers(
        input_shape=(1024, 6, 16, 16),
        input_channels=6,
        num_classes=1,
        initial_channels=32,
        num_blocks=4,
        channel_multiplier=2,
        dropout_rate=0.5,
        embedding_size=128
   )