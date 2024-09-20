import lightning as  L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import wandb
import argparse
from svtr.utils.config import Config
from svtr.module import SvtrModelModule, SvtrDataModule

def main(sweep_id):
    config = Config()

    # wandb setting
    wandb_entity = "youngjun-hwang"
    wandb_project = "svtr-mmocr"
    wandb_run_name = '20240920_svtr_mmocr'    
    wandb.login(key=config['wandb']['key'])
    print("login success!")

    def train():
        wandb.init()
        wandb_logger = WandbLogger(project=wandb_project, entity=wandb_entity, name=wandb_run_name, config={}, log_model=False)

        # Data setting
        data_module = SvtrDataModule()

        # Trainer setting
        svtr = SvtrModelModule()
        epochs = config['training']['epochs']
        num_gpus = config['training']['num_gpus']
        trainer = L.Trainer(
            max_epochs=epochs,
            accelerator='gpu',
            devices=num_gpus,
            logger=wandb_logger,
            callbacks=[
                LearningRateMonitor(logging_interval='epoch'),
                ModelCheckpoint(save_top_k=3, monitor='val_loss', mode='min'),
            ]          
        )

        trainer.fit(model=svtr, datamodule=data_module)
    
    wandb.agent(sweep_id=sweep_id, function=train, count=5, entity=wandb_entity, project=wandb_project)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process sweep_id.")
    parser.add_argument("--sweep_id", type=str, default="The sweep ID to process")
    args = parser.parse_args()

    main(args.sweep_id)