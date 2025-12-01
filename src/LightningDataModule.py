import os
from typing import Optional, List, Union
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


def get_all_mvtec_classes(root_dir):
    classes = []
    for cname in os.listdir(root_dir):
        full = os.path.join(root_dir, cname)
        if os.path.isdir(full):
            classes.append(cname)
    return sorted(classes)


class MVTecDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str,
        class_name: Union[str, List[str]],
        transform=None,
    ):
        self.paths = []
        self.class_labels = []
        self.anomaly_labels = []
        self.defect_names = []
        self.transform = transform

        if class_name == "all":
            class_list = get_all_mvtec_classes(root_dir)
        elif isinstance(class_name, str):
            class_list = [class_name]
        else:
            class_list = class_name

        self.classes = sorted(class_list)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for cls in self.classes:
            base = os.path.join(root_dir, cls, split)

            if split == "train":
                good_path = os.path.join(base, "good")
                if os.path.exists(good_path):
                    for f in os.listdir(good_path):
                        if f.lower().endswith((".png", ".jpg", ".jpeg")):
                            self.paths.append(os.path.join(good_path, f))
                            self.class_labels.append(self.class_to_idx[cls])
                            self.anomaly_labels.append(0)
                            self.defect_names.append("good")

            else:  # test
                if not os.path.exists(base):
                    continue

                for subclass in os.listdir(base):
                    sub_path = os.path.join(base, subclass)
                    if not os.path.isdir(sub_path):
                        continue

                    for f in os.listdir(sub_path):
                        if f.lower().endswith((".png", ".jpg", ".jpeg")):
                            self.paths.append(os.path.join(sub_path, f))
                            self.class_labels.append(self.class_to_idx[cls])
                            self.anomaly_labels.append(0 if subclass == "good" else 1)
                            self.defect_names.append(subclass)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return (
            img,
            self.class_labels[idx],
            self.anomaly_labels[idx],
            self.defect_names[idx],
            self.paths[idx],
        )


class MVTecDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        class_name: Union[str, List[str]],
        img_size: int,
        batch_size: int,
        num_workers: int = 4,
        model_type: str = "classifier",
    ):
        super().__init__()
        self.root_dir = root_dir
        self.class_name = class_name
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_type = model_type

    def _get_transforms(self, mode="train"):
        if mode == "train":
            aug = [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(0.1, 0.1, 0.1),
            ]
        else:
            aug = []
        
        if self.model_type == "autoencoder":
            resize_size = 128
        else:
            resize_size = self.img_size
        t = [transforms.Resize((resize_size, resize_size))]
        t.extend(aug)
        t.append(transforms.ToTensor())
        t.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )
        return transforms.Compose(t)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            full_train = MVTecDataset(
                root_dir=self.root_dir,
                split="train",
                class_name=self.class_name,
                transform=self._get_transforms("train"),
            )

            val_size = max(1, int(0.1 * len(full_train)))
            train_size = len(full_train) - val_size

            self.train_dataset, self.val_dataset = random_split(
                full_train,
                [train_size, val_size],
            )

        if stage == "test" or stage is None:
            self.test_dataset = MVTecDataset(
                root_dir=self.root_dir,
                split="test",
                class_name=self.class_name,
                transform=self._get_transforms("test"),
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
