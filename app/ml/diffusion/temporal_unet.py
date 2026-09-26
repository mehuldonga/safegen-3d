"""
Temporal U-Net for trajectory denoising.

Adapted from the official SafeDiffuser repository (Weixy21/SafeDiffuser),
which itself builds on Janner et al.'s Diffuser (https://github.com/jannerm/diffuser).

Classification: [SPECIFIED] — Architecture described in the paper and implemented in the official repo.
Paper reference: SafeDiffuser Section 3 — "Diffusion-based Planning" 
Official repo: diffuser/models/temporal.py

Local adaptations:
  - [ADAPTATION FOR LOCAL HARDWARE] Reduced default dim from 32→32 (kept same for correctness)
  - [ADAPTATION FOR LOCAL HARDWARE] Default dim_mults reduced to (1,2,4) for 6GB VRAM
  - [ENGINEERING DECISION] Removed pdb imports and print statements
  - [ENGINEERING DECISION] Added device-agnostic support
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps.
    
    Classification: [SPECIFIED] — Standard diffusion timestep embedding.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """Conv1d → GroupNorm → Mish activation."""
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ResidualTemporalBlock(nn.Module):
    """Residual block with temporal convolutions and time embedding injection.
    
    Classification: [SPECIFIED] — Standard residual block in Diffuser/SafeDiffuser.
    """
    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])
        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )
        self.residual_conv = (
            nn.Conv1d(inp_channels, out_channels, 1)
            if inp_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t):
        """
        x : [batch_size, inp_channels, horizon]
        t : [batch_size, embed_dim]
        """
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class TemporalUnet(nn.Module):
    """Temporal U-Net for denoising trajectory sequences.
    
    Takes a noisy trajectory of shape [batch, horizon, transition_dim] and 
    predicts the noise (or the clean trajectory) conditioned on a diffusion timestep.
    
    Classification: [SPECIFIED] — Core architecture from Diffuser/SafeDiffuser.
    Paper reference: SafeDiffuser Section 3, building on Janner et al. (2022).
    Official repo: diffuser/models/temporal.py, class TemporalUnet

    Args:
        horizon: trajectory length (number of timesteps in the plan)
        transition_dim: observation_dim + action_dim
        cond_dim: unused in this version (conditioning is done via apply_conditioning)
        dim: base channel dimension
        dim_mults: channel dimension multipliers for each U-Net level
    """

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim=None,
        dim=32,
        dim_mults=(1, 2, 4),  # [ADAPTATION FOR LOCAL HARDWARE] reduced from (1,2,4,8)
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        _horizon = horizon
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim, horizon=_horizon),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, horizon=_horizon),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))
            if not is_last:
                _horizon = _horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=_horizon)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=_horizon)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, horizon=_horizon),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, horizon=_horizon),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
            if not is_last:
                _horizon = _horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x, cond=None, time=None):
        """
        x    : [batch, horizon, transition_dim]
        cond : dict (unused directly — conditioning applied externally) or time tensor
        time : [batch] diffusion timestep indices
        
        Returns: [batch, horizon, transition_dim] predicted noise / x0
        """
        if time is None and isinstance(cond, torch.Tensor):
            time = cond
            cond = None
            
        x = einops.rearrange(x, 'b h t -> b t h')
        t = self.time_mlp(time)
        h = []

        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        x = self.final_conv(x)
        x = einops.rearrange(x, 'b t h -> b h t')
        return x
