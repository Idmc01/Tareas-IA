import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import wandb

class LitAE(pl.LightningModule):

    def __init__(self, model, lr=1e-3, loss_type="l1", alpha=0.5):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_type = loss_type
        self.alpha = alpha
        self.save_hyperparameters()
        
        # Para acumular datos durante validacion
        self.validation_outputs = []

    def compute_loss(self, x, xhat):
        if self.loss_type == "l1":
            return F.l1_loss(xhat, x)
        if self.loss_type == "l2":
            return F.mse_loss(xhat, x)
        if self.loss_type == "ssim":
            return 1 - ssim(xhat, x, data_range=1.0, size_average=True)
        if self.loss_type == "ssim_l1":
            ssim_l = 1 - ssim(xhat, x, data_range=1.0, size_average=True)
            l1 = F.l1_loss(xhat, x)
            return self.alpha * ssim_l + (1 - self.alpha) * l1

    def training_step(self, batch, batch_idx):
        x, _, _ = batch
        xhat, _ = self.model(x)
        loss = self.compute_loss(x, xhat)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, labels, defects = batch
        xhat, z = self.model(x)
        loss = self.compute_loss(x, xhat)
        self.log("val/loss", loss)
        
        # Guardar datos para visualizaciones al final de la epoca
        self.validation_outputs.append({
            'images': x.detach().cpu(),
            'reconstructions': xhat.detach().cpu(),
            'latents': z.detach().cpu(),
            'labels': labels.cpu(),
            'defects': defects
        })
        
        return loss

    def on_validation_epoch_end(self):
        if self.current_epoch == self.trainer.max_epochs - 1:
            self._log_all_reconstructions()
            
            fig_tsne = self._plot_tsne()
            if fig_tsne:
                self.logger.experiment.log({"visualizations/tsne": wandb.Image(fig_tsne)})
                plt.close(fig_tsne)
        
        self.validation_outputs.clear()
    
    def _log_all_reconstructions(self):
        if len(self.validation_outputs) == 0:
            return
        
        all_images = torch.cat([out['images'] for out in self.validation_outputs], dim=0)
        all_recons = torch.cat([out['reconstructions'] for out in self.validation_outputs], dim=0)
        all_labels = torch.cat([out['labels'] for out in self.validation_outputs], dim=0)
        all_defects = [d for out in self.validation_outputs for d in out['defects']]
        
        columns = ["index", "original", "reconstruction", "label", "defect_type"]
        data = []
        
        for idx in range(len(all_images)):
            img = all_images[idx].permute(1, 2, 0).numpy()
            recon = all_recons[idx].permute(1, 2, 0).numpy()
            label = "good" if all_labels[idx].item() == 0 else "defect"
            defect = all_defects[idx]
            
            data.append([
                idx,
                wandb.Image(np.clip(img, 0, 1)),
                wandb.Image(np.clip(recon, 0, 1)),
                label,
                defect
            ])
        
        table = wandb.Table(data=data, columns=columns)
        self.logger.experiment.log({"reconstructions/all_images": table})

    def _plot_tsne(self):
        if len(self.validation_outputs) == 0:
            return None
        
        all_latents = torch.cat([out['latents'] for out in self.validation_outputs], dim=0)
        all_labels = torch.cat([out['labels'] for out in self.validation_outputs], dim=0)
        
        max_samples = min(1000, len(all_latents))
        indices = np.random.choice(len(all_latents), max_samples, replace=False)
        latents = all_latents[indices].numpy()
        labels = all_labels[indices].numpy()
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        latents_2d = tsne.fit_transform(latents)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        good_mask = labels == 0
        defect_mask = labels == 1
        
        ax.scatter(latents_2d[good_mask, 0], latents_2d[good_mask, 1],
                   c='blue', label='Good', alpha=0.6, s=20)
        ax.scatter(latents_2d[defect_mask, 0], latents_2d[defect_mask, 1],
                   c='red', label='Defect', alpha=0.6, s=20)
        
        ax.set_title('t-SNE del Espacio Latente')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
