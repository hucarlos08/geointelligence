import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
import copy
import torch

def get_optimizer(model_parameters, optimizer_config: dict) -> optim.Optimizer:
    """
    Create an optimizer for a model based on a configuration dictionary without modifying the original dictionary.

    Args:
    - model_parameters: The parameters of the PyTorch model to optimize.
    - optimizer_config (dict): A dictionary containing the optimizer type and its parameters.
      Example: {'type': 'Adam', 'lr': 0.001, 'weight_decay': 1e-5}

    Returns:
    - An instance of the specified optimizer.
    """
    optimizer_config_copy = copy.deepcopy(optimizer_config)
    optimizer_type = optimizer_config_copy.pop('type')  
    
    try:
        optimizer_class = getattr(optim, optimizer_type)
    except AttributeError:
        raise ValueError(f"Optimizer {optimizer_type} not found in torch.optim")

    optimizer = optimizer_class(model_parameters, **optimizer_config_copy)
    return optimizer

def get_lr_scheduler(optimizer: optim.Optimizer, scheduler_config: dict) -> lr_scheduler._LRScheduler:
    """
    Create a learning rate scheduler for an optimizer based on a configuration dictionary.

    Args:
    - optimizer: The optimizer for which the learning rate scheduler will be applied.
    - scheduler_config (dict): A dictionary containing the scheduler type and its parameters.
      Example: {'type': 'StepLR', 'step_size': 5, 'gamma': 0.1}

    Returns:
    - An instance of the specified learning rate scheduler.
    """
    scheduler_config_copy = scheduler_config.copy()
    scheduler_type = scheduler_config_copy.pop('type')  
    
    try:
        scheduler_class = getattr(lr_scheduler, scheduler_type)
    except AttributeError:
        raise ValueError(f"Scheduler {scheduler_type} not found in torch.optim.lr_scheduler")

    scheduler = scheduler_class(optimizer, **scheduler_config_copy)
    return scheduler

def get_loss_function(loss_config: dict) -> nn.Module:
    """
    Create a loss function based on a configuration dictionary.

    Args:
    - loss_config (dict): A dictionary containing the loss function type and its parameters.
      Example: {'type': 'CrossEntropyLoss', 'weight': None, 'reduction': 'mean'}

    Returns:
    - An instance of the specified loss function.
    """
    loss_config_copy = loss_config.copy()
    loss_type = loss_config_copy.pop('type')
    
    try:
        loss_class = getattr(nn, loss_type)
    except AttributeError:
        raise ValueError(f"Loss function {loss_type} not found in torch.nn")

    loss_function = loss_class(**loss_config_copy)
    return loss_function

def get_activation_function(activation_config) -> nn.Module:
    """
    Create an activation function based on a configuration.

    Args:
    - activation_config (dict or str): A configuration for the activation function.
      If it's a dictionary, it must contain the 'type' key with the name of the activation function
      and any optional parameters. If it's a string, it's the name of the activation function with default parameters.
      Example: {'type': 'ReLU', 'inplace': True} or simply 'ReLU'

    Returns:
    - An instance of the specified activation function.
    """
    if isinstance(activation_config, str):
        activation_config = {'type': activation_config}
    
    activation_type = activation_config.get('type')

    try:
        if len(activation_config) > 1:
            activation_params = {k: v for k, v in activation_config.items() if k != 'type'}
            activation_function = getattr(nn, activation_type)(**activation_params)
        else:
            activation_function = getattr(nn, activation_type)
    except AttributeError:
        raise ValueError(f"Activation function {activation_type} not found in torch.nn")

    return activation_function