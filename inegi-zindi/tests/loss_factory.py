import unittest
import torch
import torch.nn as nn

# Import your LossCompose and LossFactory classes
from models.trainers import LossCompose, LossFactory
from models.trainers import FocalLoss  # Assuming you have this custom loss
from models.trainers import CenterLoss  # Assuming you have this custom loss

class TestLossFunctions(unittest.TestCase):

    def setUp(self):
        # Set device to GPU if available, otherwise CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Set up some example inputs and targets, and move them to the selected device
        self.inputs = torch.randn((3, 1), requires_grad=True, device=self.device)
        self.targets = torch.empty(3, 1, device=self.device).random_(2)

    def test_single_loss_function(self):
        """Test using a single loss function, like BCEWithLogitsLoss."""
        # Configuration for BCEWithLogitsLoss
        loss_config = {'BCEWithLogitsLoss': {}}
        
        # Create the loss function
        loss_function = LossFactory._create_loss(loss_config)
        
        # Compute the loss
        loss = loss_function(self.inputs, self.targets)
        
        # Assert that the loss is a scalar (i.e., has no shape)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.ndim, 0)

    def test_multiple_loss_functions(self):
        """Test combining multiple loss functions, including custom ones."""
        # Configuration for multiple loss functions
        loss_config = {
            'BCEWithLogitsLoss': {},
            'FocalLoss': {'alpha': 0.8, 'gamma': 2},
        }

        # Create the combined loss function
        loss_function = LossFactory._create_loss(loss_config)
        
        # Compute the combined loss
        loss = loss_function(self.inputs, self.targets)
        
        # Assert that the combined loss is still a scalar
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.ndim, 0)

    def test_center_loss(self):
        """Test the custom CenterLoss function."""
        # Configuration for CenterLoss
        loss_config = {
            'FocalLoss': {'alpha': 0.8, 'gamma': 2},
            'CenterLoss': {'num_classes': 2, 'feat_dim': 3}
        }

        # Create the loss function
        loss_function = LossFactory._create_loss(loss_config)
        
        # Compute the loss
        loss = loss_function(self.inputs, self.targets)
        
        # Assert that the loss is a scalar
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.ndim, 0)

    def test_invalid_loss_function(self):
        """Test behavior when an invalid loss function is provided."""
        # Invalid configuration
        loss_config = {'NonExistentLoss': {}}

        # Ensure that an error is raised
        with self.assertRaises(ValueError):
            LossFactory._create_loss(loss_config)

# Run the tests
if __name__ == '__main__':
    unittest.main()
