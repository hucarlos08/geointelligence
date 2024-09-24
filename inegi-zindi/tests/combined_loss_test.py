import pytest
import torch
import torch.nn as nn
from torch.autograd import Variable

from models.trainers import CombinedLoss, ArcFaceLoss, CenterLoss, FocalLoss

@pytest.fixture
def config():
    return {
        'arcface': {
            'params': {'feat_dim': 512, 'margin': 0.5, 'scale': 64},
            'weight': 1.0
        },
        'center': {
            'params': {'num_classes': 2, 'feat_dim': 512, 'lambda_c': 0.003},
            'weight': 0.5
        },
        'focal': {
            'params': {'alpha': 0.75, 'gamma': 2.0},
            'weight': 1.0
        }
    }

@pytest.fixture
def combined_loss(config):
    return CombinedLoss.from_config(config)

@pytest.fixture
def sample_data():
    batch_size = 32
    feat_dim = 512
    num_classes = 2

    features = torch.randn(batch_size, feat_dim)
    logits = torch.randn(batch_size, num_classes-1)
    labels = torch.randint(0, num_classes-1, (batch_size,1))

    return features, logits, labels

def test_combined_loss_initialization(combined_loss, config):
    assert isinstance(combined_loss, CombinedLoss)
    assert isinstance(combined_loss.losses['arcface'], ArcFaceLoss)
    assert isinstance(combined_loss.losses['center'], CenterLoss)
    assert isinstance(combined_loss.losses['focal'], FocalLoss)
    
    assert combined_loss.weights['arcface'] == config['arcface']['weight']
    assert combined_loss.weights['center'] == config['center']['weight']
    assert combined_loss.weights['focal'] == config['focal']['weight']

def test_combined_loss_forward(combined_loss, sample_data):
    features, logits, labels = sample_data
    
    total_loss, loss_components = combined_loss(logits, features, labels)
    
    assert isinstance(total_loss, torch.Tensor)
    assert total_loss.dim() == 0  # scalar
    assert isinstance(loss_components, dict)
    assert set(loss_components.keys()) == {'arcface','center', 'focal'}
    
    for loss_name, loss_value in loss_components.items():
        assert isinstance(loss_value, float)
        assert loss_value >= 0
    
    # Check if total_loss is approximately equal to the weighted sum of individual losses
    weighted_sum = sum(combined_loss.weights[name] * loss for name, loss in loss_components.items())
    assert torch.isclose(total_loss, torch.tensor(weighted_sum), rtol=1e-5, atol=1e-8)

def test_combined_loss_gradients(combined_loss, sample_data):
    features, logits, labels = sample_data
    
    # Make features and logits require gradients
    features = Variable(features, requires_grad=True)
    logits = Variable(logits, requires_grad=True)
    
    total_loss, _ = combined_loss(logits, features, labels)
    total_loss.backward()
    
    assert features.grad is not None
    assert logits.grad is not None
    assert not torch.isnan(features.grad).any()
    assert not torch.isnan(logits.grad).any()

def test_combined_loss_binary_classification(combined_loss, sample_data):
    features, logits, labels = sample_data

    total_loss, loss_components = combined_loss(logits, features, labels)

    assert logits.shape[1] == 1  # Binary classification
    assert labels.max().item() < 1
    assert labels.min().item() >= 0

def test_from_config(config):
    combined_loss = CombinedLoss.from_config(config)
    assert isinstance(combined_loss, CombinedLoss)
    assert set(combined_loss.losses.keys()) == set(config.keys())

if __name__ == "__main__":
    pytest.main([__file__])