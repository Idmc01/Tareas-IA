import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.LightningDataModule import MVTecDataModule
from src.models.UNetAE import UNetAE
from src.LightningModuleC import LitAutoencoder
from src.callbacks.save_embeddings import SaveEmbeddingsCallback

import wandb
import hydra.utils as hy_utils


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):

    original_cwd = hy_utils.get_original_cwd()
    os.chdir(original_cwd)

    datamodule = MVTecDataModule(
        root_dir=cfg.dataset.root,
        class_name=cfg.dataset.class_name,
        img_size=cfg.dataset.img_size,
        batch_size=cfg.dataset.batch_size,
        num_workers=4,
        model_type="autoencoder",
    )

    ae_model = UNetAE(z_dim=cfg.model.latent.dim)

    lit_model = LitAutoencoder(
        model=ae_model,
        lr=cfg.train.lr,
        loss_type=cfg.model.reconstruction.loss,
    )

    wandb_logger = WandbLogger(
        project=cfg.logger.wandb.project,
        name=cfg.logger.wandb.name,
        log_model=cfg.logger.wandb.log_model,
    )

    early_stop = EarlyStopping(
        monitor="val/loss",
        patience=cfg.trainer.callbacks.early_stopping.patience,
        mode=cfg.trainer.callbacks.early_stopping.mode,
    )

    checkpoint = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        filename="modelC-best-{epoch:02d}-{val_loss:.4f}",
    )

    emb_callback = SaveEmbeddingsCallback(
        output_dir="embeddings",
        run_name=cfg.logger.wandb.name,
        split="train",
    )

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        logger=wandb_logger,
        callbacks=[early_stop, checkpoint, emb_callback],
        log_every_n_steps=cfg.trainer.log_every_n_steps,
    )

    trainer.fit(lit_model, datamodule=datamodule)

    datamodule.setup("fit")
    emb_callback.split = "train"
    trainer.predict(lit_model, dataloaders=datamodule.train_dataloader())

    emb_callback.split = "val"
    trainer.predict(lit_model, dataloaders=datamodule.val_dataloader())

    datamodule.setup("test")
    emb_callback.split = "test"
    trainer.predict(lit_model, dataloaders=datamodule.test_dataloader())

    wandb.finish()
    return lit_model


if __name__ == "__main__":
    main()
