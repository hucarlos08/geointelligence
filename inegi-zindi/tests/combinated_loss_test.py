import unittest
import torch
from models.trainers import CombinedLoss  # Replace 'your_module' with the actual module name

class TestCombinedLoss(unittest.TestCase):
    def setUp(self):
        self.feat_dim = 256
        self.num_classes = 1
        self.batch_size = 16
        self.combined_loss = CombinedLoss(
            feat_dim=self.feat_dim,
            num_classes=self.num_classes,
            arcface_margin=0.5,
            arcface_scale=64,
            center_loss_weight=0.003,
            alpha=0.75,
            gamma=2.0
        )

    def test_combined_loss_shape_and_values(self):
        # Create dummy inputs
        logits = torch.randn(self.batch_size, self.num_classes)
        features = torch.randn(self.batch_size, self.feat_dim)
        labels = torch.randint(0, self.num_classes, (self.batch_size, 1))

        # Compute the loss
        total_loss, center_loss, focal_loss = self.combined_loss(logits, features, labels)

        # Check that the losses are not None and have the expected shape
        self.assertIsNotNone(total_loss)
        #self.assertIsNotNone(arcface_loss)
        self.assertIsNotNone(center_loss)
        self.assertIsNotNone(focal_loss)

        # Check that the losses are scalars (0-dim tensors)
        self.assertEqual(total_loss.dim(), 0)
        #self.assertEqual(arcface_loss.dim(), 0)
        self.assertEqual(center_loss.dim(), 0)
        self.assertEqual(focal_loss.dim(), 0)

        # Check that the total loss is the sum of individual losses
        self.assertAlmostEqual(total_loss.item(), 
                               (center_loss + focal_loss).item(), 
                               places=5)

        # Check that all losses are non-negative
        self.assertGreaterEqual(total_loss.item(), 0)
        #self.assertGreaterEqual(arcface_loss.item(), 0)
        self.assertGreaterEqual(center_loss.item(), 0)
        self.assertGreaterEqual(focal_loss.item(), 0)

    def test_from_config(self):
        config = {
            'feat_dim': self.feat_dim,
            'num_classes': self.num_classes,
            'arcface_margin': 0.4,
            'arcface_scale': 32,
            'center_loss_weight': 0.005,
            'alpha': 0.6,
            'gamma': 1.5
        }

        loss_from_config = CombinedLoss.from_config(config)

        self.assertEqual(loss_from_config.arcface_loss.margin, 0.4)
        self.assertEqual(loss_from_config.arcface_loss.scale, 32)
        self.assertEqual(loss_from_config.center_loss.lambda_c, 0.005)
        self.assertEqual(loss_from_config.focal_loss.alpha, 0.6)
        self.assertEqual(loss_from_config.focal_loss.gamma, 1.5)

if __name__ == '__main__':
    unittest.main()