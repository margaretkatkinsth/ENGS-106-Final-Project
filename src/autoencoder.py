import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    Residual refinement block
    """

    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.block(x)


class DownBlock(nn.Module):
    """
    Downsampling + residual refinement
    """

    def __init__(self, in_ch, out_ch, n_res=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False)
        self.norm = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()

        self.res = nn.Sequential(*[ResBlock(out_ch) for _ in range(n_res)])

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.res(x)
        return x


class UpBlock(nn.Module):
    """
    Upsampling + residual refinement
    """

    def __init__(self, in_ch, out_ch, n_res=2):
        super().__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )

        self.res = nn.Sequential(*[ResBlock(out_ch) for _ in range(n_res)])

    def forward(self, x):
        x = self.up(x)
        x = self.res(x)
        return x


class SelfAttention(nn.Module):
    """
    Spatial self-attention block
    """

    def __init__(self, ch, num_heads=4):
        super().__init__()

        self.norm = nn.GroupNorm(8, ch)
        self.attn = nn.MultiheadAttention(ch, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape

        h = self.norm(x)
        h = h.view(B, C, H * W).permute(0, 2, 1)

        h, _ = self.attn(h, h, h)

        h = h.permute(0, 2, 1).view(B, C, H, W)

        return x + h


class SpatialVAE(nn.Module):
    """
    Spatial Variational Autoencoder
    """

    def __init__(self, in_ch=1, latent_ch=4, latent_spatial=12):
        super().__init__()

        self.latent_ch = latent_ch
        self.latent_spatial = latent_spatial

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False),
            nn.SiLU(),
            DownBlock(32, 64, n_res=2),
            DownBlock(64, 128, n_res=2),
            SelfAttention(128),
            DownBlock(128, 256, n_res=2),
            SelfAttention(256),
            ResBlock(256),
            ResBlock(256),
        )

        self.to_mean = nn.Conv2d(256, latent_ch, 1)
        self.to_logvar = nn.Conv2d(256, latent_ch, 1)

        # Latent -> decoder
        self.from_latent = nn.Sequential(
            nn.Conv2d(latent_ch, 256, 3, padding=1, bias=False),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            ResBlock(256),
            ResBlock(256),
            SelfAttention(256),
            UpBlock(256, 128, n_res=2),
            SelfAttention(128),
            UpBlock(128, 64, n_res=2),
            UpBlock(64, 32, n_res=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, in_ch, 3, padding=1),
        )

    def encode(self, x):
        h = self.encoder(x)

        mean = self.to_mean(h)
        logvar = self.to_logvar(h)

        return mean, logvar

    def reparameterize(self, mean, logvar):
        if self.training:

            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)

            return mean + eps * std

        return mean

    def decode(self, z):
        h = self.from_latent(z)
        recon = self.decoder(h)

        return recon

    def forward(self, x):
        mean, logvar = self.encode(x)

        z = self.reparameterize(mean, logvar)

        recon = self.decode(z)

        return recon, mean, logvar, z


class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator
    """

    def __init__(self, in_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, 256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class SharpReconLoss(nn.Module):
    lap_kernel: torch.Tensor
    sobel_x: torch.Tensor
    sobel_y: torch.Tensor

    def __init__(self):
        super().__init__()

        lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)

        self.register_buffer("lap_kernel", lap.view(1, 1, 3, 3))

        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)

        sy = sx.T

        self.register_buffer("sobel_x", sx.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sy.view(1, 1, 3, 3))

    def edge_loss(self, pred, target):
        pred_edges = F.conv2d(pred, self.lap_kernel, padding=1)
        target_edges = F.conv2d(target, self.lap_kernel, padding=1)

        return F.l1_loss(pred_edges, target_edges)

    def gradient_loss(self, pred, target):
        pred_gx = F.conv2d(pred, self.sobel_x, padding=1)
        pred_gy = F.conv2d(pred, self.sobel_y, padding=1)

        target_gx = F.conv2d(target, self.sobel_x, padding=1)
        target_gy = F.conv2d(target, self.sobel_y, padding=1)

        return F.l1_loss(pred_gx, target_gx) + F.l1_loss(pred_gy, target_gy)

    def ssim_loss(self, pred, target, window_size=11):
        C1 = 0.01**2
        C2 = 0.03**2

        pad = window_size // 2

        mu_p = F.avg_pool2d(pred, window_size, 1, padding=pad)
        mu_t = F.avg_pool2d(target, window_size, 1, padding=pad)

        sigma_pp = F.avg_pool2d(pred**2, window_size, 1, padding=pad) - mu_p**2
        sigma_tt = F.avg_pool2d(target**2, window_size, 1, padding=pad) - mu_t**2

        sigma_pt = (
            F.avg_pool2d(pred * target, window_size, 1, padding=pad) - mu_p * mu_t
        )

        ssim = ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / (
            (mu_p**2 + mu_t**2 + C1) * (sigma_pp + sigma_tt + C2)
        )

        return 1 - ssim.mean()

    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        edge = self.edge_loss(pred, target)
        grad = self.gradient_loss(pred, target)

        total = l1 + 0.5 * ssim + 0.3 * edge + 0.2 * grad

        return total, {
            "l1": l1.item(),
            "ssim": ssim.item(),
            "edge": edge.item(),
            "grad": grad.item(),
        }
