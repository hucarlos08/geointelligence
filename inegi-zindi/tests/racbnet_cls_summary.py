import unittest
from torchinfo import summary
from models.nn import ResAttnConvNet

class TestUNetSummary(unittest.TestCase):

    def setUp(self):
        # Define the input parameters
        self.batch_size = 1024
        self.image_height = 16
        self.image_width = 16
        self.channels = 6

        # Dictionary with the model configuration
        model_config={
            'input_channels': 6,
            'initial_channels': 32,
            'embedding_size': 256,
            'num_classes': 1,
            'reduction': 16,
            'dropout_rate': 0.5,
            'depth': 2
        }
       
        self.model = ResAttnConvNet(**model_config)

    def test_model_summary(self):
        model_summary = summary(
            self.model,
            input_size=(self.batch_size, self.channels, self.image_height, self.image_width),
            depth=4,
            col_width=16,
            col_names=["kernel_size", "output_size", "num_params", "mult_adds"],
            row_settings=["var_names"]
        )
        self.assertTrue(model_summary)

if __name__ == '__main__':
    unittest.main()