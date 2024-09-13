import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import wandb
import traceback
import torch.nn.functional as F

from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


from models.data import LandsatDataModule
from models.nn import ResAttentionConvNet
from models.trainers import BasicTrainer
from models.trainers import FocalLoss

def train_inegi_model():
    """
    Function to train the classification models.
    """
    try:
        # Initialize wandb
        wandb.init(project="INEGI", entity="geo-dl")

        # Setup wandb logger
        wandb_logger = WandbLogger(project="INEGI", entity="geo-dl")

        # Dictionary with the configuration for the data module
        config_data_module = {
            'train_file': '/teamspace/studios/this_studio/dataset/optimized_balanced_train_data.h5',
            'test_file': '/teamspace/studios/this_studio/dataset/test_data.h5',
            'batch_size': 1024,
            'num_workers': 4,
            'seed': 50,
            'split_ratio': (0.8, 0.2),
            # 'transform': {
            #     'Normalize': {  'mean': [0.5,  0.5, 0.5], 
            #                     'std':  [0.5, 0.5, 0.5]
            #                     },
            # }
        }
        # Create the HDF5DataModule from the configuration
        data_module = LandsatDataModule.from_config(config_data_module)
        print("DataModule created successfully")

        # Loss function
        #loss = nn.BCEWithLogitsLoss()
        #loss = SigmoidFocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
        loss = FocalLoss(alpha=0.25, gamma=2.0)

        # Create model
        config_model = {
            'input_channels': 6,
            'embedding_size': 256,
            'num_classes': 1
        }

        # Crear el modelo desde la configuración
        model = ResAttentionConvNet.from_config(config_model)
        
        # Optimizer and scheduler configuration
        optimizer_config = {
            'type': 'AdamW',
            'lr': 1e-3,
            'weight_decay': 1.0e-5
        }
        scheduler_config = {
            'type': 'StepLR',
            'step_size': 10,
            'gamma': 0.1
        }

        # Create Lightning module
        trainer_module = BasicTrainer(model, loss, optimizer_config, scheduler_config)
        print("Lightning module created successfully")

         # Setup model checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath='checkpoints',
            filename='mistletoe-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            monitor='val_loss',
            mode='min'
        )

        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        # Create trainer
        trainer = pl.Trainer(
            max_epochs=20,
            logger=wandb_logger,
            log_every_n_steps=5,
            callbacks=[checkpoint_callback, lr_monitor],
            accumulate_grad_batches=1,
            devices=1 if torch.cuda.is_available() else None,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu'
        )

        # Train model
        trainer.fit(trainer_module, data_module)

        # Test model
        #trainer.test(trainer_module, datamodule=data_module)

    except Exception as e:
        print(f"An error occurred during training: {e}")
         # Print the full traceback, including the line number
        traceback.print_exc()
    finally:
        # Close wandb run
        wandb.finish()
        return None

if __name__ == "__main__":
    train_inegi_model()