"""Denoising Diffusion Probabilistic Model (DDPM) for Fashion MNIST.

Implements the DDPM from Ho et al., 2020 (https://arxiv.org/abs/2006.11239)
using only PyTorch. A small U-Net with sinusoidal time embeddings predicts
the noise added at each timestep, and the model is trained with the simple
MSE objective on epsilon.
"""

import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# ---------------------------------------------------------------------------
# U-Net building blocks
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.op(x)


class UNet(nn.Module):
    """Small U-Net for 28x28 single-channel images.

    Pads the input to 32x32 so the two downsampling stages yield clean
    8x8 feature maps, then crops back on output.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults=(1, 2, 2),
        time_dim: int = 128,
    ):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        channels = [base_channels * m for m in channel_mults]
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        prev_ch = base_channels
        for i, ch in enumerate(channels):
            self.down_blocks.append(
                nn.ModuleList(
                    [
                        ResidualBlock(prev_ch, ch, time_dim),
                        ResidualBlock(ch, ch, time_dim),
                    ]
                )
            )
            if i < len(channels) - 1:
                self.downsamples.append(Downsample(ch))
            else:
                self.downsamples.append(nn.Identity())
            prev_ch = ch

        self.mid1 = ResidualBlock(prev_ch, prev_ch, time_dim)
        self.mid2 = ResidualBlock(prev_ch, prev_ch, time_dim)

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, ch in enumerate(reversed(channels)):
            self.up_blocks.append(
                nn.ModuleList(
                    [
                        ResidualBlock(prev_ch + ch, ch, time_dim),
                        ResidualBlock(ch + ch, ch, time_dim),
                    ]
                )
            )
            if i < len(channels) - 1:
                self.upsamples.append(Upsample(ch))
            else:
                self.upsamples.append(nn.Identity())
            prev_ch = ch

        self.out_norm = nn.GroupNorm(8, prev_ch)
        self.out_conv = nn.Conv2d(prev_ch, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (2, 2, 2, 2))  # 28 -> 32
        t_emb = self.time_embed(t)

        h = self.init_conv(x)
        skips = []
        for (block1, block2), down in zip(self.down_blocks, self.downsamples):
            h = block1(h, t_emb)
            skips.append(h)
            h = block2(h, t_emb)
            skips.append(h)
            h = down(h)

        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        for (block1, block2), up in zip(self.up_blocks, self.upsamples):
            h = torch.cat([h, skips.pop()], dim=1)
            h = block1(h, t_emb)
            h = torch.cat([h, skips.pop()], dim=1)
            h = block2(h, t_emb)
            h = up(h)

        h = self.out_conv(F.silu(self.out_norm(h)))
        return h[:, :, 2:-2, 2:-2]  # crop 32 -> 28


# ---------------------------------------------------------------------------
# Diffusion process
# ---------------------------------------------------------------------------
def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    return torch.linspace(beta_start, beta_end, T)


class DDPM(nn.Module):
    def __init__(self, model: nn.Module, T: int = 1000, img_shape=(1, 28, 28)):
        super().__init__()
        self.model = model
        self.T = T
        self.img_shape = img_shape

        betas = linear_beta_schedule(T)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = F.pad(alpha_bars[:-1], (1, 0), value=1.0)
        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("posterior_variance", posterior_variance)

    @staticmethod
    def _extract(buf: torch.Tensor, t: torch.Tensor, shape) -> torch.Tensor:
        out = buf.gather(0, t)
        return out.view(-1, *([1] * (len(shape) - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self._extract(self.sqrt_alpha_bars, t, x0.shape)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alpha_bars, t, x0.shape)
        return sqrt_ab * x0 + sqrt_1mab * noise

    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        b = x0.size(0)
        t = torch.randint(0, self.T, (b,), device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred = self.model(xt, t)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def p_sample(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        betas_t = self._extract(self.betas, t, xt.shape)
        sqrt_1mab_t = self._extract(self.sqrt_one_minus_alpha_bars, t, xt.shape)
        sqrt_recip_a_t = self._extract(self.sqrt_recip_alphas, t, xt.shape)

        eps = self.model(xt, t)
        mean = sqrt_recip_a_t * (xt - betas_t / sqrt_1mab_t * eps)

        nonzero = (t != 0).float().view(-1, *([1] * (xt.dim() - 1)))
        var = self._extract(self.posterior_variance, t, xt.shape)
        noise = torch.randn_like(xt)
        return mean + nonzero * torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        x = torch.randn(n, *self.img_shape, device=device)
        for step in reversed(range(self.T)):
            t = torch.full((n,), step, device=device, dtype=torch.long)
            x = self.p_sample(x, t)
        return x


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def get_dataloader(batch_size: int, data_dir: str) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1),  # [0,1] -> [-1,1]
        ]
    )
    dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )


def save_samples(ddpm: DDPM, device: torch.device, path: Path, n: int = 64) -> None:
    ddpm.eval()
    samples = ddpm.sample(n, device)
    samples = (samples.clamp(-1, 1) + 1) / 2
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(samples, path, nrow=int(math.sqrt(n)))
    ddpm.train()


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loader = get_dataloader(args.batch_size, args.data_dir)
    unet = UNet(in_channels=1, base_channels=args.base_channels)
    ddpm = DDPM(unet, T=args.timesteps).to(device)

    n_params = sum(p.numel() for p in ddpm.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.Adam(ddpm.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(args.epochs):
        running = 0.0
        for i, (x, _) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            loss = ddpm.loss(x)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
            optimizer.step()

            running += loss.item()
            step += 1
            if step % args.log_every == 0:
                avg = running / args.log_every
                print(f"epoch {epoch} step {step} loss {avg:.4f}")
                running = 0.0

        if (epoch + 1) % args.sample_every == 0 or epoch == args.epochs - 1:
            save_samples(ddpm, device, out_dir / f"samples_epoch_{epoch + 1:03d}.png")
            torch.save(
                {"model": ddpm.state_dict(), "args": vars(args)},
                out_dir / "ddpm_fashion_mnist.pt",
            )
            print(f"Saved samples and checkpoint at epoch {epoch + 1}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DDPM for Fashion MNIST")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--out-dir", type=str, default="./outputs")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--sample-every", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
