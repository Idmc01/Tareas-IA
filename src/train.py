import hydra
from omegaconf import DictConfig
from torchvision import transforms
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from data import MVTecDataset
from models.ae import ClassicAE
from lightning_module import LitAE


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):

    # ----------------------
    # LOGGER (WandB)
    # ----------------------
    wandb_logger = WandbLogger(
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        log_model=cfg.logger.log_model
    )

    # ----------------------
    # TRANSFORMS
    # ----------------------
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # ----------------------
    # DATASETS Y LOADERS
    # ----------------------
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
        num_workers=4
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=4
    )

    # ----------------------
    # MODELO
    # ----------------------
    model = ClassicAE(z_dim=cfg.model.z_dim)

    lit_model = LitAE(
        model=model,
        lr=cfg.train.lr,
        loss_type=cfg.loss.type,
        alpha=cfg.loss.alpha
    )

    # ----------------------
    # TRAINER
    # ----------------------
    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        logger=wandb_logger,
        log_every_n_steps=cfg.trainer.log_every_n_steps
    )

    # ----------------------
    # ENTRENAR
    # ----------------------
    trainer.fit(lit_model, train_loader, val_loader)


if __name__ == "__main__":
    main()
