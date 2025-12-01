import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.LightningDataModule import MVTecDataModule
from src.LightningModuleA import LitClassifierScratch
from src.models.resnet_partial import build_resnet18_partial
from src.callbacks.save_embeddings import SaveEmbeddingsCallback


def train_model_a(
    lr: float,
    batch_size: int,
    epochs: int,
    embedding_dim: int,
    hidden_dim: int,
    run_name: str,
    dropout: float = 0.3,
    weight_decay: float = 1e-4,
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
        model_type="classifier",
    )

    model_backbone = build_resnet18_partial(
        layers=["conv1", "conv2_x", "conv3_x"],
        embedding_dim=embedding_dim,
        num_classes=10,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )

    lit_model = LitClassifierScratch(
        model=model_backbone,
        lr=lr,
        weight_decay=weight_decay,
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
        split="train",
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        logger=wandb_logger,
        callbacks=[early_stop, checkpoint, emb_callback],
        log_every_n_steps=20,
    )

    trainer.fit(lit_model, datamodule=datamodule)

    datamodule.setup("fit")
    emb_callback.split = "train"
    trainer.predict(
        lit_model,
        dataloaders=datamodule.train_dataloader(),
    )

    datamodule.setup("fit")
    emb_callback.split = "val"
    trainer.predict(
        lit_model,
        dataloaders=datamodule.val_dataloader(),
    )

    datamodule.setup("test")
    emb_callback.split = "test"
    trainer.predict(
        lit_model,
        dataloaders=datamodule.test_dataloader(),
    )

    wandb.finish()
    return lit_model, trainer