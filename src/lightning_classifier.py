import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support
import seaborn as sns
import wandb


class LitClassifier(pl.LightningModule):
    def __init__(self, model, lr=1e-3, weight_decay=1e-4, num_classes=2):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        
        self.save_hyperparameters(ignore=['model'])
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.validation_outputs = []
        self.test_outputs = []

    def training_step(self, batch, batch_idx):
        x, labels, _ = batch
        logits, features = self.model(x)
        
        loss = self.criterion(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, labels, defects = batch
        logits, features = self.model(x)
        loss = self.criterion(logits, labels)
        
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        
        self.validation_outputs.append({
            'logits': logits.detach().cpu(),
            'probs': probs.detach().cpu(),
            'preds': preds.cpu(),
            'labels': labels.cpu(),
            'features': features.detach().cpu(),
            'defects': defects
        })
        
        return loss

    def test_step(self, batch, batch_idx):
        x, labels, defects = batch
        logits, features = self.model(x)
        loss = self.criterion(logits, labels)
        
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        
        self.test_outputs.append({
            'logits': logits.detach().cpu(),
            'probs': probs.detach().cpu(),
            'preds': preds.cpu(),
            'labels': labels.cpu(),
            'features': features.detach().cpu(),
            'defects': defects
        })
        
        return loss

    def on_validation_epoch_end(self):
        if len(self.validation_outputs) == 0:
            return
        
        all_preds = torch.cat([o['preds'] for o in self.validation_outputs])
        all_labels = torch.cat([o['labels'] for o in self.validation_outputs])
        all_probs = torch.cat([o['probs'] for o in self.validation_outputs])
        all_features = torch.cat([o['features'] for o in self.validation_outputs])
        all_defects = [d for o in self.validation_outputs for d in o['defects']]
        
        if self.num_classes == 2:
            try:
                auc = roc_auc_score(all_labels.numpy(), all_probs[:, 1].numpy())
                self.log("val/auc", auc)
            except:
                pass
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels.numpy(), all_preds.numpy(), average='binary', zero_division=0
        )
        self.log("val/precision", precision)
        self.log("val/recall", recall)
        self.log("val/f1", f1)
        
        if self.current_epoch == self.trainer.max_epochs - 1:
            fig_cm = self._plot_confusion_matrix(all_preds.numpy(), all_labels.numpy(), "Validation")
            if fig_cm:
                self.logger.experiment.log({"visualizations/val_confusion_matrix": wandb.Image(fig_cm)})
                plt.close(fig_cm)
            
            fig_tsne = self._plot_tsne(all_features.numpy(), all_labels.numpy(), all_defects, "Validation")
            if fig_tsne:
                self.logger.experiment.log({"visualizations/val_tsne": wandb.Image(fig_tsne)})
                plt.close(fig_tsne)
        
        self.validation_outputs.clear()

    def on_test_epoch_end(self):
        if len(self.test_outputs) == 0:
            return
        
        all_preds = torch.cat([o['preds'] for o in self.test_outputs])
        all_labels = torch.cat([o['labels'] for o in self.test_outputs])
        all_probs = torch.cat([o['probs'] for o in self.test_outputs])
        all_features = torch.cat([o['features'] for o in self.test_outputs])
        all_defects = [d for o in self.test_outputs for d in o['defects']]
        
        if self.num_classes == 2:
            try:
                auc = roc_auc_score(all_labels.numpy(), all_probs[:, 1].numpy())
                self.log("test/auc", auc)
            except:
                pass
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels.numpy(), all_preds.numpy(), average='binary', zero_division=0
        )
        self.log("test/precision", precision)
        self.log("test/recall", recall)
        self.log("test/f1", f1)
        
        fig_cm = self._plot_confusion_matrix(all_preds.numpy(), all_labels.numpy(), "Test")
        if fig_cm:
            self.logger.experiment.log({"visualizations/test_confusion_matrix": wandb.Image(fig_cm)})
            plt.close(fig_cm)
        
        fig_tsne = self._plot_tsne(all_features.numpy(), all_labels.numpy(), all_defects, "Test")
        if fig_tsne:
            self.logger.experiment.log({"visualizations/test_tsne": wandb.Image(fig_tsne)})
            plt.close(fig_tsne)
        
        self._log_predictions_table(all_preds, all_labels, all_probs, all_defects)
        
        self._save_embeddings(all_features.numpy(), all_labels.numpy(), all_defects)
        
        self.test_outputs.clear()

    def _plot_confusion_matrix(self, preds, labels, split_name):
        cm = confusion_matrix(labels, preds)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Good', 'Defect'],
                    yticklabels=['Good', 'Defect'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix - Modelo A ({split_name})')
        plt.tight_layout()
        return fig

    def _plot_tsne(self, features, labels, defects, split_name):
        max_samples = min(1000, len(features))
        if max_samples < 10:
            return None
            
        indices = np.random.choice(len(features), max_samples, replace=False)
        
        feat_subset = features[indices]
        labels_subset = labels[indices]
        defects_subset = [defects[i] for i in indices]
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max_samples-1))
        features_2d = tsne.fit_transform(feat_subset)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        ax1 = axes[0]
        colors = ['blue' if l == 0 else 'red' for l in labels_subset]
        ax1.scatter(features_2d[:, 0], features_2d[:, 1], c=colors, alpha=0.6, s=20)
        ax1.set_title(f't-SNE Modelo A ({split_name}): Good vs Defect')
        ax1.set_xlabel('Dim 1')
        ax1.set_ylabel('Dim 2')
        ax1.legend(['Good', 'Defect'])
        
        ax2 = axes[1]
        unique_defects = list(set(defects_subset))
        cmap = plt.cm.get_cmap('tab20', len(unique_defects))
        defect_to_idx = {d: i for i, d in enumerate(unique_defects)}
        colors2 = [cmap(defect_to_idx[d]) for d in defects_subset]
        ax2.scatter(features_2d[:, 0], features_2d[:, 1], c=colors2, alpha=0.6, s=20)
        ax2.set_title(f't-SNE by Defect Type ({split_name})')
        ax2.set_xlabel('Dim 1')
        ax2.set_ylabel('Dim 2')
        
        handles = [plt.Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=cmap(defect_to_idx[d]), markersize=8, label=d)
                   for d in unique_defects]
        ax2.legend(handles=handles, loc='best', fontsize=8, ncol=2)
        
        plt.tight_layout()
        return fig

    def _log_predictions_table(self, preds, labels, probs, defects):
        columns = ["idx", "true_label", "pred_label", "prob_good", "prob_defect", "correct", "defect_type"]
        data = []
        
        max_rows = min(500, len(preds))
        for idx in range(max_rows):
            data.append([
                idx,
                "good" if labels[idx] == 0 else "defect",
                "good" if preds[idx] == 0 else "defect",
                f"{probs[idx, 0]:.4f}",
                f"{probs[idx, 1]:.4f}",
                "correct" if preds[idx] == labels[idx] else "wrong",
                defects[idx]
            ])
        
        table = wandb.Table(data=data, columns=columns)
        self.logger.experiment.log({"predictions/test": table})

    def _save_embeddings(self, embeddings, labels, defects):
        import os
        
        os.makedirs("embeddings", exist_ok=True)
        
        model_name = self.__class__.__name__
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
                description=f'Feature embeddings from {model_name}'
            )
            
            artifact.add_file(save_path)
            self.logger.experiment.log_artifact(artifact)
            print(f"Embeddings subidos a WandB como artefacto")
        except Exception as e:
            print(f"No se pudo subir a WandB: {e}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.max_epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }
