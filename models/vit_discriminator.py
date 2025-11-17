import sys
sys.path.insert(0, "../Project_")
import torch
import torch.nn as nn
from torchvision import transforms
from config import config

try:
    import timm
except ImportError:
    timm = None


class DinoViTDiscriminator(nn.Module):
    def __init__(self, num_styles=config.NUM_STYLES):
        super().__init__()

        if timm is None:
            raise RuntimeError("Install timm: pip install timm")
        self.backbone = timm.create_model(
            "vit_small_patch16_224.dino",
            pretrained=True,
            num_classes=0,     # (B, embed_dim)
        )

        # DINO feature dimension (384 for vit_small)
        self.feature_dim = self.backbone.embed_dim

        self.style_emb = nn.Embedding(num_styles, self.feature_dim)

        self.real_fake_head = nn.Linear(self.feature_dim, 1)
        self.style_head = nn.Linear(self.feature_dim, num_styles)

        self.resize = transforms.Resize((224,224))
        self.mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
        self.std  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)

        # Partial unfreezing: only last 2 blocks + final norm
        for name, p in self.backbone.named_parameters():
            p.requires_grad = False

        for blk in self.backbone.blocks[-2:]:
            for p in blk.parameters():
                p.requires_grad = True

        for p in self.backbone.norm.parameters():
            p.requires_grad = True


    def forward(self, x, style_labels):
        """
        x: fake or real defective image in [-1,1]
        """
        device = x.device

        # [-1,1] → [0,1]
        x = (x + 1.0) / 2.0
        x = torch.clamp(x, 0.0, 1.0)

        # Resize to DINO input size
        x = self.resize(x)
        mean = self.mean.to(device)
        std = self.std.to(device)
        x = (x - mean) / std

        # Pass through DINO (pixel → patchify → transformer)
        feat = self.backbone(x)  # (B, feature_dim)
        style_vec = self.style_emb(style_labels)
        fused = feat + style_vec

        real_fake_logits = self.real_fake_head(fused)
        style_logits = self.style_head(feat)

        return real_fake_logits, style_logits
