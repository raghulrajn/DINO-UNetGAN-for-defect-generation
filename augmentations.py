import torch
from torchvision.transforms import v2 as T
from torchvision.transforms import functional as F
import random

class PairedTransform:
    def __init__(self, image_size):
        self.image_size = image_size

        # Spatial transforms
        self.resize = T.Resize((image_size, image_size))

        self.hflip = T.RandomHorizontalFlip(p=0.5)
        self.vflip = T.RandomVerticalFlip(p=0.2)
        self.rotate = T.RandomRotation(10)

        # Color transforms
        self.color_jitter = T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.02,
        )

        self.to_tensor = T.ToTensor()

    def _apply_spatial(self, clean, defect, mask):

        clean = self.resize(clean)
        defect = self.resize(defect)
        mask = self.resize(mask)

        # Random horizontal flip
        if random.random() < 0.5:
            clean = F.hflip(clean)
            defect = F.hflip(defect)
            mask = F.hflip(mask)

        # Random vertical flip
        if random.random() < 0.2:
            clean = F.vflip(clean)
            defect = F.vflip(defect)
            mask = F.vflip(mask)

        # Random rotation
        angle = random.uniform(-10, 10)
        clean = F.rotate(clean, angle)
        defect = F.rotate(defect, angle)
        mask = F.rotate(mask, angle)

        return clean, defect, mask

    def _apply_color(self, clean, defect):
        clean = self.color_jitter(clean)
        defect = self.color_jitter(defect)
        return clean, defect

    def __call__(self, clean_img, defect_img, mask_img):
        clean_img, defect_img, mask_img = self._apply_spatial(clean_img, defect_img, mask_img)

        clean_img, defect_img = self._apply_color(clean_img, defect_img)

        clean_t = self.to_tensor(clean_img)
        defect_t = self.to_tensor(defect_img)
        mask_t = F.to_tensor(mask_img)
        clean_t = torch.clamp(clean_t, 0.0, 1.0)
        defect_t = torch.clamp(defect_t, 0.0, 1.0)

        if random.random() < 0.5:
            noise = torch.randn_like(clean_t) * 0.02
            clean_t = torch.clamp(clean_t + noise, 0.0, 1.0)

        if random.random() < 0.5:
            noise = torch.randn_like(defect_t) * 0.02
            defect_t = torch.clamp(defect_t + noise, 0.0, 1.0)

        return clean_t, defect_t, mask_t
