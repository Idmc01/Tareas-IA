import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class LitAutoencoder(pl.LightningModule):
    def __init__(self, model, lr=1e-3, loss_type="l2"):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_type = loss_type

        self.save_hyperparameters(ignore=["model"])

    def compute_loss(self, x, xhat):
        if self.loss_type == "l1":
            return F.l1_loss(xhat, x)
        return F.mse_loss(xhat, x)

    def forward(self, x):
        xhat, z = self.model(x)
        return xhat, z

    def training_step(self, batch, batch_idx):
        x, class_label, anomaly_label, defect_name, path = batch
        xhat, _ = self.model(x)
        loss = self.compute_loss(x, xhat)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, class_label, anomaly_label, defect_name, path = batch
        xhat, _ = self.model(x)
        loss = self.compute_loss(x, xhat)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, class_label, anomaly_label, defect_name, path = batch
        _, z = self.model(x)

        return {
            "embedding": z.cpu(),
            "class_label": class_label.cpu(),
            "anomaly_label": anomaly_label.cpu(),
            "defect_name": defect_name,
            "path": path,
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)