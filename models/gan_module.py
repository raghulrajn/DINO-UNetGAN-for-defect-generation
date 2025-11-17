import sys
sys.path.insert(0, "../Project_")
import torch
import torch.nn as nn
from config import config
from .unet import UNetGenerator
from .vit_discriminator import DinoViTDiscriminator

class GANModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = UNetGenerator(
            in_channels=config.IN_CHANNELS,
            out_channels=config.OUT_CHANNELS,
            base_channels=config.BASE_CHANNELS,
        )
        self.discriminator = DinoViTDiscriminator()

        self.adv_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss(reduction="none")
        self.ce_loss = nn.CrossEntropyLoss()

    def _masked_l1(self, fake, target, mask):
        if mask is None or not config.USE_MASKED_L1:
            return torch.mean(torch.abs(fake - target))

        mask = (mask > 0.5).float()

        if mask.shape[1] == 1 and fake.shape[1] == 3:
            mask = mask.repeat(1, fake.shape[1], 1, 1)

        diff = torch.abs(fake - target) * mask
        denom = mask.sum()
        if denom.item() == 0:
            return torch.mean(torch.abs(fake - target))
        return diff.sum() / denom

    def generator_step(self, batch, device):
        clean = batch["clean"].to(device)
        target_defect = batch["defective"].to(device)
        style = batch["style"].to(device)
        mask = batch.get("mask", None)
        if mask is not None:
            mask = mask.to(device)

        fake_defect = self.generator(clean, style)

        fake_l1 = (fake_defect + 1.0) / 2.0
        fake_l1 = torch.clamp(fake_l1, 0.0, 1.0)

        rf_logits_fake, style_logits_fake = self.discriminator(fake_defect, style)

        valid = torch.ones_like(rf_logits_fake, device=device)
        adv = self.adv_loss(rf_logits_fake, valid)

        l1 = self._masked_l1(fake_l1, target_defect, mask)

        style_loss = self.ce_loss(style_logits_fake, style)

        total_g = (
            config.LAMBDA_ADV * adv +
            config.LAMBDA_L1 * l1 +
            config.LAMBDA_STYLE * style_loss
        )

        losses = {
            "g_total": total_g,
            "g_adv": adv,
            "g_l1": l1,
            "g_style": style_loss,
        }
        return total_g, losses, fake_defect.detach()

    def discriminator_step(self, batch, fake_detached, device):
        real_defect = batch["defective"].to(device)
        style = batch["style"].to(device)

        real_in = real_defect * 2.0 - 1.0  # Convert real to [-1,1] for discriminator
        fake_in = fake_detached

        rf_logits_real, style_logits_real = self.discriminator(real_in, style)
        valid = torch.ones_like(rf_logits_real, device=device)

        real_loss = self.adv_loss(rf_logits_real, valid)
        style_loss_real = self.ce_loss(style_logits_real, style)

        rf_logits_fake, _ = self.discriminator(fake_in, style)
        fake_label = torch.zeros_like(rf_logits_fake, device=device)
        fake_loss = self.adv_loss(rf_logits_fake, fake_label)

        d_adv = 0.5 * (real_loss + fake_loss)
        total_d = d_adv + config.LAMBDA_STYLE * style_loss_real

        losses = {
            "d_total": total_d,
            "d_adv": d_adv,
            "d_style": style_loss_real,
        }
        return total_d, losses
