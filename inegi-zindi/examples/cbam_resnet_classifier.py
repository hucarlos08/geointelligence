import numpy as np
import yaml
import logging
import traceback
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping


from models.data import LandsatDataModule
from models.nn import CBAMResNet
from models.losses import CombinedLoss
from models.trainers import FeatureAwareTrainer


def train_model():

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    CONFIG_FILE = '/teamspace/studios/this_studio/inegi/inegi-zindi/examples/configs/cbam_resnet_config.yaml'
    # Loading the config from a YAML file
    try:
        with open(CONFIG_FILE, 'r') as file:
             config = yaml.load(file, Loader=yaml.FullLoader)
        logger.info("Config loaded successfully")

        # Create the HDF5DataModule from the configuration
        data_module_config = config['data_module']
        data_module = LandsatDataModule.from_config(data_module_config)
        logger.info("DataModule created successfully")

        # Create the model from the configuration
        model_config = config['model']

        # Crear el modelo desde la configuración
        model = CBAMResNet.from_config(model_config)
        logger.info("Model created successfully")

        # Create the loss function from the configuration
        # Automatically set 'embedding_size' in the model to be equal to 'feat_dim' in the center loss function
        config['loss_functions']['center']['params']['feat_dim'] = config['model']['embedding_size']

        loss_config = config['loss_functions']
        loss = CombinedLoss.from_config(loss_config)
        logger.info("Loss function created successfully")

        # Create the training module
        optimizer_config = config['optimizer']
        scheduler_config = config['scheduler']

        trainer_module = FeatureAwareTrainer(model, loss, optimizer_config, scheduler_config)
        logger.info("Feature Aware Trainer module created successfully")

        # Initialize wandb
        run_name = f"{model.get_class_name()}_embed{config['model']['embedding_size']}"
        wandb.init(project="INEGI", entity="geo-dl", config=config, name=run_name)

        # Setup wandb logger
        wandb_logger = WandbLogger(project="INEGI", entity="geo-dl")

        model_name = f'inegi-{model.get_class_name()}'

        # Setup model checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath='checkpoints',
            filename=f'{model_name}'+'-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            monitor='val_aucroc',
            mode='max'
        )

        early_stop_callback = EarlyStopping(
            monitor='val_aucroc', # Metric to monitor
            patience=10,          # Number of epochs with no improvement before stopping training
            verbose=True,        # To display messages during training
            mode='max',          # 'min' to reduce the metric, 'max' to maximize it
            min_delta=0.0     # Minimum improvement considered significant
        )

        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        # Setup trainer
        trainer = pl.Trainer(
            max_epochs=30,
            logger=wandb_logger,
            log_every_n_steps=5,
            callbacks=[checkpoint_callback, lr_monitor, early_stop_callback],
            accumulate_grad_batches=1,
            devices=1 if torch.cuda.is_available() else None,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu'
        )
        logger.info("Lightning Trainer created successfully")

        # Train the model
        trainer.fit(trainer_module, data_module)

    except Exception as e:
        logger.info(f"An error occurred during training: {e}")
         # Print the full traceback, including the line number
        traceback.print_exc()
        return
    finally:
        # Close wandb run
        wandb.finish()
        return

if __name__ == "__main__":
    train_model()