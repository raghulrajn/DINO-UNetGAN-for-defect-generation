import os
import json
from PIL import Image
from torch.utils.data import Dataset
from config import config
from augmentations import PairedTransform

class MVTecDefectDataset(Dataset):
    def __init__(self, category_name, split="train"):
        """
        Dataset for paired clean/defect/mask images from MVTec-style preprocessed folders.
        """
        self.root = os.path.join(config.PREPARED_ROOT, category_name)
        self.clean_dir = os.path.join(self.root, "clean")
        self.def_dir = os.path.join(self.root, "defective")
        self.mask_dir = os.path.join(self.root, "masks")

        self.files = sorted(os.listdir(self.clean_dir))

        with open(os.path.join(self.root, "style_labels.json")) as f:
            self.style_labels_map = json.load(f)

        self.paired_transform = PairedTransform(config.IMAGE_SIZE)

        self.style_to_idx = config.STYLE_LABELS

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        clean_path = os.path.join(self.clean_dir, fname)
        defect_path = os.path.join(self.def_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)

        clean_img = Image.open(clean_path).convert("RGB")
        defect_img = Image.open(defect_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("L")

        clean_t, defect_t, mask_t = self.paired_transform(
            clean_img, defect_img, mask_img
        )

        style_name = self.style_labels_map[fname]
        style_idx = self.style_to_idx.get(style_name, 0)

        return {
            "clean": clean_t,
            "defective": defect_t,
            "mask": mask_t,
            "style": style_idx,
            "filename": fname,
        }
