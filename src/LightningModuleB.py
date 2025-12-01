import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class LitDistilledClassifier(pl.LightningModule):
    def __init__(
        self,
        student_model,
        teacher_model,
        lr=1e-3,
        weight_decay=1e-4,
        temperature=4.0,
        alpha=0.7,
    ):
        super().__init__()
        self.student = student_model
        self.teacher = teacher_model
        self.lr = lr
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.alpha = alpha

        self.save_hyperparameters(ignore=["student_model", "teacher_model"])

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, x):
        logits, embedding = self.student(x)
        return logits, embedding

    def distillation_loss(self, student_logits, teacher_logits, labels):
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)

        kd = self.kl_loss(soft_student, soft_teacher) * (self.temperature**2)
        ce = self.ce_loss(student_logits, labels)

        total = self.alpha * kd + (1 - self.alpha) * ce
        return total, kd, ce

    def training_step(self, batch, batch_idx):
        x, class_label, anomaly_label, defect_name, path = batch

        student_logits, _ = self.student(x)
        with torch.no_grad():
            teacher_logits = self.teacher(x)

        total_loss, kd_loss, ce_loss = self.distillation_loss(
            student_logits, teacher_logits, class_label
        )

        self.log("train/loss", total_loss, on_step=False, on_epoch=True)
        self.log("train/kd_loss", kd_loss, on_step=False, on_epoch=True)
        self.log("train/ce_loss", ce_loss, on_step=False, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, class_label, anomaly_label, defect_name, path = batch

        student_logits, _ = self.student(x)
        with torch.no_grad():
            teacher_logits = self.teacher(x)

        total_loss, kd_loss, ce_loss = self.distillation_loss(
            student_logits, teacher_logits, class_label
        )

        self.log("val/loss", total_loss, on_step=False, on_epoch=True)
        self.log("val/kd_loss", kd_loss, on_step=False, on_epoch=True)
        self.log("val/ce_loss", ce_loss, on_step=False, on_epoch=True)

        return total_loss

    def predict_step(self, batch, batch_idx):
        x, class_label, anomaly_label, defect_name, path = batch
        _, embedding = self.student(x)

        return {
            "embedding": embedding.cpu(),
            "class_label": class_label.cpu(),
            "anomaly_label": anomaly_label.cpu(),
            "defect_name": defect_name,
            "path": path,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
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