import pytorch_lightning as pl
import torch
import torch.nn as nn


class LitClassifierScratch(pl.LightningModule):
    def __init__(self, model, lr=1e-3, weight_decay=1e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x):
        logits, features = self.model(x)
        return logits, features

    def training_step(self, batch, batch_idx):
        x, class_label, anomaly_label, defect_name, path = batch
        logits, _ = self(x)
        loss = self.criterion(logits, class_label)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, class_label, anomaly_label, defect_name, path = batch
        logits, _ = self(x)
        loss = self.criterion(logits, class_label)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, class_label, anomaly_label, defect_name, path = batch
        _, embedding = self(x)
        return {
            "embedding": embedding.cpu(),
            "class_label": class_label.cpu(),
            "anomaly_label": anomaly_label.cpu(),
            "defect_name": defect_name,
            "path": path,
        }

    def predict_step(self, batch, batch_idx):
        x, class_label, anomaly_label, defect_name, path = batch
        _, embedding = self(x)
        return {
            "embedding": embedding.cpu(),
            "class_label": class_label.cpu(),
            "anomaly_label": anomaly_label.cpu(),
            "defect_name": defect_name,
            "path": path,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }