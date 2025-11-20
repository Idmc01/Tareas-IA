import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim

class LitAE(pl.LightningModule):

    def __init__(self, model, lr=1e-3, loss_type="l1", alpha=0.5):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_type = loss_type
        self.alpha = alpha
        self.save_hyperparameters()

    def compute_loss(self, x, xhat):
        if self.loss_type == "l1":
            return F.l1_loss(xhat, x)
        if self.loss_type == "l2":
            return F.mse_loss(xhat, x)
        if self.loss_type == "ssim":
            return 1 - ssim(xhat, x, data_range=1.0, size_average=True, channel=3)
        if self.loss_type == "ssim_l1":
            ssim_l = 1 - ssim(xhat, x, data_range=1.0, size_average=True, channel=3)
            l1 = F.l1_loss(xhat, x)
            return self.alpha * ssim_l + (1 - self.alpha) * l1

    def training_step(self, batch, batch_idx):
        x, _, _ = batch
        xhat, _ = self.model(x)
        loss = self.compute_loss(x, xhat)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, _ = batch
        xhat, _ = self.model(x)
        loss = self.compute_loss(x, xhat)
        self.log("val/loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
