import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import cast


class ActNorm(nn.Module):
    """
    Activation Normalization — learnable per-channel scale and bias.
    Data-dependent initialization on first batch.
    Replaces BatchNorm in flows (BatchNorm breaks exact log-det).
    """

    def __init__(self, channels):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.initialized = False

    @torch.no_grad()
    def initialize(self, x):
        """Set scale/bias so first batch has zero mean, unit variance."""
        mean = x.mean(dim=[0, 2, 3], keepdim=True)
        std = x.std(dim=[0, 2, 3], keepdim=True) + 1e-6
        self.bias.data.copy_(-mean)
        self.log_scale.data.copy_(-torch.log(std))
        self.initialized = True

    def forward(self, x):
        if not self.initialized:
            self.initialize(x)

        y = x * torch.exp(self.log_scale) + self.bias
        H, W = x.shape[2], x.shape[3]
        log_det = self.log_scale.sum() * H * W
        return y, log_det

    def inverse(self, y):
        x = (y - self.bias) * torch.exp(-self.log_scale)
        return x


class InvertibleConv1x1(nn.Module):
    """
    Learnable channel permutation via 1×1 convolution.
    Uses LU decomposition for efficient log-det computation.
    """

    def __init__(self, channels):
        super().__init__()
        # Initialize as random rotation
        W = torch.linalg.qr(torch.randn(channels, channels))[0]
        # LU decomposition for O(c) log-det instead of O(c³)
        P, L, U = torch.linalg.lu(W)

        self.P: torch.Tensor
        self.sign_S: torch.Tensor
        self.log_S: torch.Tensor

        self.register_buffer("P", P)
        self.L = nn.Parameter(L)
        self.U = nn.Parameter(U)
        # Store sign of diagonal separately for log computation
        S = torch.diag(U)
        self.register_buffer("sign_S", torch.sign(S))
        self.log_S = nn.Parameter(torch.log(torch.abs(S)))

    def _get_weight(self):
        C = self.L.shape[0]

        L = torch.tril(self.L, -1) + torch.eye(C, device=self.L.device)

        U = torch.triu(self.U, 1)
        diag = self.sign_S * torch.exp(self.log_S)
        U = U + torch.diag(diag)

        return self.P @ L @ U

    def forward(self, x):
        W = self._get_weight()
        y = F.conv2d(x, W.unsqueeze(-1).unsqueeze(-1))
        _, _, H, W_spatial = x.shape
        log_det = torch.sum(self.log_S) * H * W_spatial
        return y, log_det

    def inverse(self, y):
        W = self._get_weight()
        W_inv = torch.inverse(W)
        return F.conv2d(y, W_inv.unsqueeze(-1).unsqueeze(-1))


class AffineCoupling(nn.Module):
    """
    Split channels in half. Use first half to predict
    scale and translation for second half.
    """

    def __init__(self, channels, hidden_channels=256):
        super().__init__()
        half = channels // 2

        self.net = nn.Sequential(
            nn.Conv2d(half, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 3, padding=1),
        )

        last = cast(nn.Conv2d, self.net[-1])
        nn.init.zeros_(cast(torch.Tensor, last.weight))
        nn.init.zeros_(cast(torch.Tensor, last.bias))

    def forward(self, x):
        xa, xb = x.chunk(2, dim=1)

        st = self.net(xa)
        log_s, t = st.chunk(2, dim=1)
        log_s = torch.tanh(log_s) * 3.0

        yb = xb * torch.exp(log_s) + t
        y = torch.cat([xa, yb], dim=1)

        log_det = log_s.sum(dim=[1, 2, 3])
        return y, log_det

    def inverse(self, y):
        ya, yb = y.chunk(2, dim=1)

        st = self.net(ya)
        log_s, t = st.chunk(2, dim=1)
        log_s = torch.tanh(log_s) * 3.0

        xb = (yb - t) * torch.exp(-log_s)
        return torch.cat([ya, xb], dim=1)


class Squeeze2d(nn.Module):
    """(B, C, H, W) → (B, 4C, H/2, W/2). Invertible, volume-preserving."""

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C * 4, H // 2, W // 2)
        return x

    def inverse(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C // 4, 2, 2, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(B, C // 4, H * 2, W * 2)
        return x


class FlowStep(nn.Module):
    def __init__(self, channels, hidden_channels=256):
        super().__init__()
        self.actnorm = ActNorm(channels)
        self.invconv = InvertibleConv1x1(channels)
        self.coupling = AffineCoupling(channels, hidden_channels)

    def forward(self, x):
        x, ld1 = self.actnorm(x)
        x, ld2 = self.invconv(x)
        x, ld3 = self.coupling(x)
        return x, ld1 + ld2 + ld3

    def inverse(self, y):
        y = self.coupling.inverse(y)
        y = self.invconv.inverse(y)
        y = self.actnorm.inverse(y)
        return y


class FlowLevel(nn.Module):
    def __init__(self, channels, hidden_channels, K):
        super().__init__()
        self.steps = nn.ModuleList(
            [FlowStep(channels, hidden_channels) for _ in range(K)]
        )

    def forward(self, x):
        log_det = 0
        for step in self.steps:
            x, ld = step(x)
            log_det += ld
        return x, log_det

    def inverse(self, x):
        for step in reversed(self.steps):
            x = cast(FlowStep, step).inverse(x)
        return x


class Glow(nn.Module):
    """
    Glow flow for spatial latents
    """

    def __init__(self, in_channels=3, hidden_channels=256, K=12, n_scales=2):
        super().__init__()
        self.n_scales = n_scales
        self.squeeze = Squeeze2d()

        self.flow_blocks = nn.ModuleList()
        self.split_channels = []

        channels = in_channels
        for s in range(n_scales):
            channels = channels * 4  # after squeeze

            self.flow_blocks.append(FlowLevel(channels, hidden_channels, K))

            if s < n_scales - 1:
                # Split: factor out half the channels
                self.split_channels.append(channels // 2)
                channels = channels // 2
            else:
                self.split_channels.append(channels)

    def forward(self, x):
        """
        Forward: latent → noise
        Returns list of z's and total log_det
        """
        total_log_det = 0
        zs = []

        for s in range(self.n_scales):
            x = self.squeeze(x)

            x, ld = self.flow_blocks[s](x)
            total_log_det += ld

            if s < self.n_scales - 1:
                z, x = x.chunk(2, dim=1)
                zs.append(z)
            else:
                zs.append(x)

        return zs, total_log_det

    def inverse(self, zs):
        x = zs[-1]

        for s in reversed(range(self.n_scales)):
            if s < self.n_scales - 1:
                x = torch.cat([zs[s], x], dim=1)

            x = cast(FlowLevel, self.flow_blocks[s]).inverse(x)
            x = self.squeeze.inverse(x)

        return x

    def log_prob(self, x):
        """Compute log probability of latent x under the flow model."""
        zs, log_det = self.forward(x)

        log_pz = 0
        for z in zs:
            log_pz += -0.5 * (z.pow(2) + np.log(2 * np.pi)).sum(dim=[1, 2, 3])

        return log_pz + log_det

    def sample(self, n_samples, device, temperature=0.7):
        zs = []

        zs.append(torch.randn(n_samples, 6, 6, 6, device=device) * temperature)

        zs.append(torch.randn(n_samples, 24, 3, 3, device=device) * temperature)

        return self.inverse(zs)
