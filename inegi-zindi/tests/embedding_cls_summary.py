import unittest
from torchinfo import summary
from models.nn import ConvEmbeddingClassifier

class TestUNetSummary(unittest.TestCase):

    def setUp(self):
        self.batch_size = 1024
        self.channels = 6
        self.image_height = 16
        self.image_width = 16
        self.input_shape = self.channels*self.image_height*self.image_width
        self.model = ConvEmbeddingClassifier(input_channels=self.channels, embedding_size=256, num_classes=1)

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