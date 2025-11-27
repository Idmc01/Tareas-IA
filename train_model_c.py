import hydra
from omegaconf import DictConfig
from torchvision import transforms
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import sys
sys.path.append('src')

from data import MVTecDataset
from models.unet_autoencoder import UNetAE
from lightning_autoencoder import LitAE


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    
    run_name = f"ModeloC_{cfg.dataset.class_name[0]}_lr{cfg.train.lr}_bs{cfg.dataset.batch_size}_zdim{cfg.model.latent.dim}"
    
    wandb_logger = WandbLogger(
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        log_model=cfg.logger.log_model,
        name=run_name,
        tags=["modelo_c", "autoencoder", "unet"]
    )

    transform = transforms.Compose([
        transforms.Resize((cfg.dataset.img_size, cfg.dataset.img_size)),
        transforms.ToTensor()
    ])

    train_ds = MVTecDataset(
        root_dir=cfg.dataset.root,
        split="train",
        class_name=cfg.dataset.class_name,
        transform=transform
    )

    val_ds = MVTecDataset(
        root_dir=cfg.dataset.root,
        split="test",
        class_name=cfg.dataset.class_name,
        transform=transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = UNetAE(z_dim=cfg.model.latent.dim)
    
    print(f"\n>>> MODELO C | LR={cfg.train.lr} | BS={cfg.dataset.batch_size} | Z={cfg.model.latent.dim} | Loss={cfg.model.reconstruction.loss} | Epochs={cfg.train.epochs}\n")

    lit_model = LitAE(
        model=model,
        lr=cfg.train.lr,
        loss_type=cfg.model.reconstruction.loss,
        alpha=0.5
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss',
        mode='min',
        save_top_k=1,
        filename='modelo_C-{epoch:02d}-{val_loss:.4f}'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val/loss',
        patience=7,
        mode='min'
    )

    trainer = Trainer(
        max_epochs=cfg.train.epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        logger=wandb_logger,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    trainer.fit(lit_model, train_loader, val_loader)
    
    trainer.test(lit_model, val_loader)


if __name__ == "__main__":
    main()
