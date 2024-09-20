import lightning as  L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import wandb
import argparse
from svtr.utils.config import Config
from svtr.module import SvtrModelModule, SvtrDataModule

def main():
    config = Config()   
        
    data_module = SvtrDataModule()

    # Trainer setting
    svtr = SvtrModelModule()
    epochs = config.training_config['epochs']
    num_gpus = config.training_config['num_gpus']
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator='gpu',
        devices=num_gpus,
        callbacks=[
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(save_top_k=3, monitor='val_loss', mode='min'),
        ]          
    )

    trainer.fit(model=svtr, datamodule=data_module)

if __name__ == "__main__":
    main()