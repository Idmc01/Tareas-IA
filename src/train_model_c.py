import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.LightningDataModule import MVTecDataModule
from src.models.UNetAE import UNetAE
from src.LightningModuleC import LitAutoencoder
from src.callbacks.save_embeddings import SaveEmbeddingsCallback

import wandb


def train_model_c(
    lr: float,
    batch_size: int,
    epochs: int,
    z_dim: int,
    run_name: str,
    loss_type: str = "l2"
):
    datamodule = MVTecDataModule(
        root_dir="data",
        class_name=[
            "bottle", "cable", "capsule", "grid", "pill",
            "screw", "tile", "toothbrush", "transistor", "zipper"
        ],
        img_size=224,          
        batch_size=batch_size,
        num_workers=4,
        model_type="autoencoder"
    )

    ae_model = UNetAE(z_dim=z_dim)

    lit_model = LitAutoencoder(
        model=ae_model,
        lr=lr,
        loss_type=loss_type
    )

    wandb_logger = WandbLogger(
        project="Proyecto-II",
        name=run_name,
        log_model=True
    )

    early_stop = EarlyStopping(
        monitor="val/loss",
        patience=10,
        mode="min"
    )

    checkpoint = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        filename=f"{run_name}-best-{{epoch:02d}}-{{val_loss:.4f}}"
    )

    emb_callback = SaveEmbeddingsCallback(
        output_dir="embeddings",
        run_name=run_name,
        split="train"
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        logger=wandb_logger,
        callbacks=[early_stop, checkpoint, emb_callback],
        log_every_n_steps=20
    )

    trainer.fit(lit_model, datamodule=datamodule)

    datamodule.setup("fit")
    emb_callback.split = "train"
    trainer.predict(lit_model, dataloaders=datamodule.train_dataloader())

    datamodule.setup("fit")
    emb_callback.split = "val"
    trainer.predict(lit_model, dataloaders=datamodule.val_dataloader())

    datamodule.setup("test")
    emb_callback.split = "test"
    trainer.predict(lit_model, dataloaders=datamodule.test_dataloader())

    wandb.finish()

    return lit_model, trainer
