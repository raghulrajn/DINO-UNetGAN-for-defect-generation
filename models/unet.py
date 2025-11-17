import sys
sys.path.insert(0, "../Project_")
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config

class FiLM(nn.Module):
    """
    Feature-wise linear modulation from style embedding.
    gamma, beta are predicted from style vector.
    """
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.fc_gamma = nn.Linear(style_dim, num_features)
        self.fc_beta = nn.Linear(style_dim, num_features)

    def forward(self, x, style_emb):
        # x: (B, C, H, W), style_emb: (B, style_dim)
        gamma = self.fc_gamma(style_emb).unsqueeze(-1).unsqueeze(-1)
        beta = self.fc_beta(style_emb).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, style_dim=None, use_film=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.use_film = use_film
        if use_film and style_dim is not None:
            self.film = FiLM(style_dim, out_ch)
        else:
            self.film = None

    def forward(self, x, style_emb=None):
        x = self.conv(x)
        if self.use_film and self.film is not None and style_emb is not None:
            x = self.film(x, style_emb)
        return x

class UNetGenerator(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 base_channels=64,
                 num_styles=config.NUM_STYLES,
                 style_dim=config.STYLE_EMB_DIM):
        super().__init__()

        self.style_emb = nn.Embedding(num_styles, style_dim)

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels, style_dim, use_film=True)
        self.enc2 = ConvBlock(base_channels, base_channels*2, style_dim, use_film=True)
        self.enc3 = ConvBlock(base_channels*2, base_channels*4, style_dim, use_film=True)
        self.enc4 = ConvBlock(base_channels*4, base_channels*8, style_dim, use_film=True)

        # Bottleneck
        self.bottleneck = ConvBlock(base_channels*8, base_channels*16, style_dim, use_film=True)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_channels*16, base_channels*8, 2, stride=2)
        self.dec4 = ConvBlock(base_channels*16, base_channels*8, style_dim, use_film=True)

        self.up3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, stride=2)
        self.dec3 = ConvBlock(base_channels*8, base_channels*4, style_dim, use_film=True)

        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2)
        self.dec2 = ConvBlock(base_channels*4, base_channels*2, style_dim, use_film=True)

        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)
        self.dec1 = ConvBlock(base_channels*2, base_channels, style_dim, use_film=True)

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x, style_labels):
        """
        x: clean image (B,3,H,W)
        style_labels: long tensor (B,) with style indices
        """
        style_emb = self.style_emb(style_labels)  # (B, style_dim)

        # Encoder
        e1 = self.enc1(x, style_emb)
        p1 = F.max_pool2d(e1, 2)

        e2 = self.enc2(p1, style_emb)
        p2 = F.max_pool2d(e2, 2)

        e3 = self.enc3(p2, style_emb)
        p3 = F.max_pool2d(e3, 2)

        e4 = self.enc4(p3, style_emb)
        p4 = F.max_pool2d(e4, 2)

        # Bottleneck
        b = self.bottleneck(p4, style_emb)

        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4, style_emb)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3, style_emb)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2, style_emb)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1, style_emb)

        out = self.out_conv(d1)
        out = torch.tanh(out)
        return out
