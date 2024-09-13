import unittest
from torchinfo import summary
from models.nn import SimpleClassifier

class TestUNetSummary(unittest.TestCase):

    def setUp(self):
        self.batch_size = 1024
        self.channels = 6
        self.image_height = 16
        self.image_width = 16
        self.input_shape = self.channels*self.image_height*self.image_width
        self.model = SimpleClassifier(input_channels=self.channels, image_size=self.image_height, num_classes=2)

    def test_model_summary(self):
        model_summary = summary(
            self.model,
            input_size=(self.batch_size, self.input_shape),
            depth=4,
            col_width=16,
            col_names=["kernel_size", "output_size", "num_params", "mult_adds"],
            row_settings=["var_names"]
        )
        self.assertTrue(model_summary)

if __name__ == '__main__':
    unittest.main()