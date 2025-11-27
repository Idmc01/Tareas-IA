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
from models.resnet_partial import ResNet18Partial
from lightning_classifier import LitClassifier


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    
    run_name = f"ModeloA_{cfg.dataset.class_name[0]}_lr{cfg.train.lr}_bs{cfg.dataset.batch_size}"
    
    wandb_logger = WandbLogger(
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        log_model=cfg.logger.log_model,
        name=run_name,
        tags=["modelo_a", "scratch", "resnet18_partial"]
    )

    train_transform = transforms.Compose([
        transforms.Resize((cfg.dataset.img_size, cfg.dataset.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((cfg.dataset.img_size, cfg.dataset.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = MVTecDataset(
        root_dir=cfg.dataset.root,
        split="train",
        class_name=cfg.dataset.class_name,
        transform=train_transform
    )

    val_ds = MVTecDataset(
        root_dir=cfg.dataset.root,
        split="test",
        class_name=cfg.dataset.class_name,
        transform=val_transform
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

    model = ResNet18Partial(num_classes=cfg.dataset.num_classes)
    
    print(f"\n>>> MODELO A | LR={cfg.train.lr} | BS={cfg.dataset.batch_size} | Epochs={cfg.train.epochs}\n")

    lit_model = LitClassifier(
        model=model,
        lr=cfg.train.lr,
        weight_decay=cfg.model.training.weight_decay,
        num_classes=cfg.dataset.num_classes
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val/acc',
        mode='max',
        save_top_k=1,
        filename='modelo_A-{epoch:02d}-{val_acc:.4f}'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val/loss',
        patience=5,
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
