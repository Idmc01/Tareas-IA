import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.LightningDataModule import MVTecDataModule
from src.LightningModuleB import LitDistilledClassifier
from src.models.resnet_partial import build_resnet18_partial
from src.models.teacher_loader import load_teacher_resnet18
from src.callbacks.save_embeddings import SaveEmbeddingsCallback

import wandb


def train_model_b(
    lr: float,
    batch_size: int,
    epochs: int,
    embedding_dim: int,
    hidden_dim: int,
    run_name: str,
    dropout: float = 0.3,
    weight_decay: float = 1e-4,
    temperature: float = 4.0,
    alpha: float = 0.7,
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

    student_backbone = build_resnet18_partial(
        layers=["conv1", "conv2_x", "conv3_x"],
        embedding_dim=embedding_dim,
        num_classes=10,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )

    teacher_model = load_teacher_resnet18(num_classes=10)

    lit_model = LitDistilledClassifier(
        student_model=student_backbone,
        teacher_model=teacher_model,
        lr=lr,
        weight_decay=weight_decay,
        temperature=temperature,
        alpha=alpha,
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
    trainer.predict(lit_model, dataloaders=datamodule.train_dataloader())

    datamodule.setup("fit")
    emb_callback.split = "val"
    trainer.predict(lit_model, dataloaders=datamodule.val_dataloader())

    datamodule.setup("test")
    emb_callback.split = "test"
    trainer.predict(lit_model, dataloaders=datamodule.test_dataloader())

    wandb.finish()

    return lit_model, trainer