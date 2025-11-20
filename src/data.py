from torch.utils.data import Dataset
from PIL import Image
import os

class MVTecDataset(Dataset):
    def __init__(self, root_dir, split='train', class_name='cable', transform=None):
        self.paths = []
        self.labels = []
        self.defect_names = []

        base = os.path.join(root_dir, class_name, split)

        # --- ENTRENAMIENTO: solo good ---
        if split == "train":
            good_path = os.path.join(base, "good")
            for f in os.listdir(good_path):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.paths.append(os.path.join(good_path, f))
                    self.labels.append(0)  # 0 = good
                    self.defect_names.append("good")

        # --- TEST/VALIDATION: good + TODOS los defectos ---
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

        label = self.labels[idx]          # 0 = good, 1 = defect
        defect = self.defect_names[idx]   # p. ej. "good", "bent_wire", etc.

        return img, label, defect