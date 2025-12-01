import os
import numpy as np
import wandb
import pytorch_lightning as pl


class SaveEmbeddingsCallback(pl.Callback):
    def __init__(self, output_dir="embeddings", run_name="run", split="train"):
        super().__init__()
        self.output_dir = output_dir
        self.run_name = run_name
        self.split = split
        self.outputs = []

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.outputs.append(outputs)

    def on_predict_end(self, trainer, pl_module):
        if len(self.outputs) == 0:
            return

        embeddings = []
        class_labels = []
        anomaly_labels = []
        defect_names = []
        paths = []

        for out in self.outputs:
            embeddings.append(out["embedding"].numpy())
            class_labels.append(out["class_label"].numpy())
            anomaly_labels.append(out["anomaly_label"].numpy())
            defect_names.extend(out["defect_name"])
            paths.extend(out["path"])

        embeddings = np.concatenate(embeddings, axis=0)
        class_labels = np.concatenate(class_labels, axis=0)
        anomaly_labels = np.concatenate(anomaly_labels, axis=0)

        os.makedirs(self.output_dir, exist_ok=True)

        filename = f"{self.run_name}_{self.split}_embeddings.npz"
        filepath = os.path.join(self.output_dir, filename)

        np.savez_compressed(
            filepath,
            embeddings=embeddings,
            class_labels=class_labels,
            anomaly_labels=anomaly_labels,
            defect_names=np.array(defect_names, dtype=object),
            paths=np.array(paths, dtype=object),
        )

        artifact = wandb.Artifact(self.run_name + "_" + self.split, type="embeddings")
        artifact.add_file(filepath)
        wandb.log_artifact(artifact)

        self.outputs = []