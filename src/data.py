import os
from typing import Optional, List, Union
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class MVTecDataset(Dataset):
    def __init__(
        self, 
        root_dir: str, 
        split: str = 'train', 
        class_name: Union[str, List[str]] = 'cable', 
        transform=None,
        return_mask: bool = False
    ):
        self.paths = []
        self.labels = []
        self.defect_names = []
        self.return_mask = return_mask
        
        if isinstance(class_name, str):
            class_name = [class_name]
        
        for cls in class_name:
            base = os.path.join(root_dir, cls, split)
            
            if split == "train":
                good_path = os.path.join(base, "good")
                if os.path.exists(good_path):
                    for f in os.listdir(good_path):
                        if f.lower().endswith((".png", ".jpg", ".jpeg")):
                            self.paths.append(os.path.join(good_path, f))
                            self.labels.append(0)
                            self.defect_names.append("good")
            else:
                for subclass in os.listdir(base):
                    sub_path = os.path.join(base, subclass)
                    if not os.path.isdir(sub_path):
                        continue

                    for f in os.listdir(sub_path):
                        if f.lower().endswith((".png", ".jpg", ".jpeg")):
                            self.paths.append(os.path.join(sub_path, f))
                            is_good = (subclass == "good")
                            self.labels.append(0 if is_good else 1)
                            self.defect_names.append(subclass)

        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        
        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        defect = self.defect_names[idx]

        return img, label, defect


class MVTecDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        class_name: Union[str, List[str]],
        img_size: int = 128,
        batch_size: int = 32,
        num_workers: int = 4,
        use_augmentation: bool = True,
        model_type: str = "classifier"
    ):
        super().__init__()
        self.root_dir = root_dir
        self.class_name = class_name
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_augmentation = use_augmentation
        self.model_type = model_type
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            if self.use_augmentation and self.model_type == "classifier":
                train_transform = transforms.Compose([
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(10),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                train_transform = transforms.Compose([
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            
            val_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            self.train_dataset = MVTecDataset(
                root_dir=self.root_dir,
                split="train",
                class_name=self.class_name,
                transform=train_transform
            )
            
            self.val_dataset = MVTecDataset(
                root_dir=self.root_dir,
                split="test",
                class_name=self.class_name,
                transform=val_transform
            )
        
        if stage == "test" or stage is None:
            test_transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            self.test_dataset = MVTecDataset(
                root_dir=self.root_dir,
                split="test",
                class_name=self.class_name,
                transform=test_transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
