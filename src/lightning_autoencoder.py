import pytorch_lightning as pl
import torch
import torch.nn.functional as F
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
        self.save_hyperparameters(ignore=['model'])

        self.validation_outputs = []
        self.test_outputs = []

    def compute_loss(self, x, xhat):
        if self.loss_type == "l1":
            return F.l1_loss(xhat, x)
        if self.loss_type == "l2":
            return F.mse_loss(xhat, x)
        return F.l1_loss(xhat, x)

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

        self.validation_outputs.append({
            'images': x.detach().cpu(),
            'reconstructions': xhat.detach().cpu(),
            'latents': z.detach().cpu(),
            'labels': labels.cpu(),
            'defects': defects
        })

        return loss

    def test_step(self, batch, batch_idx):
        x, labels, defects = batch
        xhat, z = self.model(x)
        loss = self.compute_loss(x, xhat)
        self.log("test/loss", loss)

        self.test_outputs.append({
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
                self.logger.experiment.log({"visualizations/val_tsne": wandb.Image(fig_tsne)})
                plt.close(fig_tsne)

        self.validation_outputs.clear()

    def on_test_epoch_end(self):
        all_latents = torch.cat([out['latents'] for out in self.test_outputs], dim=0)
        all_labels = torch.cat([out['labels'] for out in self.test_outputs], dim=0)
        all_defects = [d for out in self.test_outputs for d in out['defects']]

        self._log_all_reconstructions(split="test")

        fig_tsne = self._plot_tsne(split="test")
        if fig_tsne:
            self.logger.experiment.log({"visualizations/test_tsne": wandb.Image(fig_tsne)})
            plt.close(fig_tsne)

        self._save_embeddings(all_latents.numpy(), all_labels.numpy(), all_defects)

        self.test_outputs.clear()

    def _log_all_reconstructions(self, split="val"):
        outputs = self.validation_outputs if split == "val" else self.test_outputs
        
        if len(outputs) == 0:
            return

        all_images = torch.cat([out['images'] for out in outputs], dim=0)
        all_recons = torch.cat([out['reconstructions'] for out in outputs], dim=0)
        all_labels = torch.cat([out['labels'] for out in outputs], dim=0)
        all_defects = [d for out in outputs for d in out['defects']]

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
        self.logger.experiment.log({f"reconstructions/{split}_images": table})

    def _plot_tsne(self, split="val"):
        outputs = self.validation_outputs if split == "val" else self.test_outputs
        
        if len(outputs) == 0:
            return None

        all_latents = torch.cat([out['latents'] for out in outputs], dim=0)
        all_labels = torch.cat([out['labels'] for out in outputs], dim=0)

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

    def _save_embeddings(self, embeddings, labels, defects):
        import os
        
        os.makedirs("embeddings", exist_ok=True)
        
        model_name = "ModelC_UNet"
        save_path = f"embeddings/{model_name}_embeddings.npz"
        
        np.savez(
            save_path,
            embeddings=embeddings,
            labels=labels,
            defects=np.array(defects, dtype=object)
        )
        
        print(f"Embeddings guardados localmente en: {save_path}")
        print(f"  - Shape: {embeddings.shape}")
        print(f"  - Good samples: {(labels == 0).sum()}")
        print(f"  - Defect samples: {(labels == 1).sum()}")
        
        try:
            artifact = wandb.Artifact(
                f'{model_name}_embeddings',
                type='embeddings',
                description=f'Latent embeddings from {model_name}'
            )
            
            artifact.add_file(save_path)
            self.logger.experiment.log_artifact(artifact)
            print(f"Embeddings subidos a WandB como artefacto")
        except Exception as e:
            print(f"No se pudo subir a WandB: {e}")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
