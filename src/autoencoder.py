import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """
    This model block performs residual facial refinement
    """
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.block(x)


class DownBlock(nn.Module):
    """
    This model block performs downsampling while extracting features using ResBlock
    """
    def __init__(self, in_ch, out_ch, n_res=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False)
        self.norm = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU(inplace=True)
        self.res = nn.Sequential(*[ResBlock(out_ch) for _ in range(n_res)])

    def forward(self, x):
        return self.res(self.act(self.norm(self.conv(x))))


class UpBlock(nn.Module):
    """
    This model block performs upsampling to recover features using ResBlock
    """
    def __init__(self, in_ch, out_ch, n_res=2):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(inplace=True),
        )
        self.res = nn.Sequential(*[ResBlock(out_ch) for _ in range(n_res)])

    def forward(self, x):
        return self.res(self.up(x))


class SelfAttention(nn.Module):
    """
    This model block incorporates attention using a MultiheadAttention block.
    """
    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.attn = nn.MultiheadAttention(ch, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.view(B, C, H * W).permute(0, 2, 1)    # (B, HW, C)
        h, _ = self.attn(h, h, h)
        h = h.permute(0, 2, 1).view(B, C, H, W)
        return x + h


class SpatialVAE(nn.Module):
    """
    Spatial Variational Autoencoder
    """
    def __init__(self, in_ch=1, latent_ch=2, latent_spatial=12):
        super().__init__()
        self.latent_ch = latent_ch
        self.latent_spatial = latent_spatial

        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1, bias=False),         # first conv layer
            nn.SiLU(inplace=True),                                  # nonlinear activation

            # these blocks downsample the image, while increasing the number of channels
            DownBlock(64, 128, n_res=2),
            DownBlock(128, 256, n_res=2),
            DownBlock(256, 256, n_res=2),

            SelfAttention(256),
            ResBlock(256),
            ResBlock(256),
        )

        self.to_mean = nn.Conv2d(256, latent_ch, 1)
        self.to_logvar = nn.Conv2d(256, latent_ch, 1)

        self.from_latent = nn.Sequential(
            nn.Conv2d(latent_ch, 256, 3, padding=1, bias=False),
            nn.GroupNorm(8, 256),
            nn.SiLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            ResBlock(256),
            ResBlock(256),
            SelfAttention(256),
            ResBlock(256),
            ResBlock(256),

            UpBlock(256, 256, n_res=3),      # 12 → 24
            SelfAttention(256),               # attention at 24×24

            UpBlock(256, 128, n_res=3),      # 24 → 48

            UpBlock(128, 64, n_res=3),       # 48 → 96

            # Final refinement
            ResBlock(64),
            ResBlock(64),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, in_ch, 3, padding=1),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.to_mean(h), self.to_logvar(h)

    def reparameterize(self, mean, logvar):
        if self.training:
            return mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        return mean

    def decode(self, z):
        return self.decoder(self.from_latent(z))

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return recon, mean, logvar, z
