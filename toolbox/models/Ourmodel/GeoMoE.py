"""Geometry-guided mixture-of-experts (GeoMoE), equations (2)-(9)."""
from __future__ import annotations

import contextlib
import io
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F

from .AGE import AGE
from .TPE import TPE


def orthogonal_loss(F_tex: torch.Tensor, F_geo: torch.Tensor) -> torch.Tensor:
    b, c, _, _ = F_tex.shape
    tex = F.normalize(F_tex.view(b, c, -1), p=2, dim=1)
    geo = F.normalize(F_geo.view(b, c, -1), p=2, dim=1)
    return torch.mean(torch.abs(torch.sum(tex * geo, dim=1)))


def normalize_map(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_min = x.amin(dim=(2, 3), keepdim=True)
    x_max = x.amax(dim=(2, 3), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


def _load_gmr_conv2d():
    from .GMRC import GMR_Conv2d
    return GMR_Conv2d


def _load_profile_texture_block():
    # Legacy Potsdam checkpoint only; never substitute this for the paper TPE.
    try:
        from .legacy_texture import Block
    except ImportError as exc:
        raise ImportError(
            "Legacy profile_a requires ShiftWise CUDA extensions. Install with "
            "`cd toolbox/models/shift_wiseConv_master/shiftadd && python setup.py install`."
        ) from exc
    return Block


class GeoMoE(nn.Module):
    """Height-guided routing and residual fusion of TPE and AGE responses."""
    def __init__(
        self,
        dim: int,
        reduction_dim: int = 256,
        use_gmr_prior: bool = True,
        texture_mode: str = "profile_b",
        architecture_version: str = "paper_v1",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.reduction_dim = reduction_dim
        self.use_gmr_prior = use_gmr_prior
        self.texture_mode = texture_mode
        self.architecture_version = architecture_version

        self.down_proj = nn.Sequential(
            nn.Conv2d(dim, reduction_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduction_dim),
            nn.GELU(),
        )

        self.register_buffer(
            "sobel_x",
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3),
        )
        self.spatial_router = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=1),
        )

        if self.texture_mode == "profile_a":
            with contextlib.redirect_stdout(io.StringIO()):
                self.tex_expert = _load_profile_texture_block()(dim=reduction_dim)
        elif self.texture_mode == "profile_b":
            self.tex_expert = TPE(in_channels=reduction_dim, architecture_version=architecture_version)
        else:
            raise ValueError(f"Unsupported teacher profile: {self.texture_mode}")
        self.geo_expert = AGE(inc=reduction_dim, outc=reduction_dim)
        if self.use_gmr_prior:
            self._inject_gmr_prior(self.geo_expert, reduction_dim)

        self.up_proj = nn.Conv2d(reduction_dim, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)
        if self.texture_mode == "profile_b":
            self.tex_response = nn.Conv2d(reduction_dim, 1, kernel_size=1)
            self.geo_response = nn.Conv2d(reduction_dim, 1, kernel_size=1)

    @staticmethod
    def _inject_gmr_prior(age_module: nn.Module, dim: int) -> None:
        # Configure the GMRC modulation (m_conv) and bias (b_conv) branches.
        GMR_Conv2d = _load_gmr_conv2d()
        age_module.m_conv[0] = GMR_Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=1,
            stride=1,
            dw_fwd=False,
        )
        age_module.b_conv[0] = GMR_Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=1,
            stride=1,
            dw_fwd=False,
        )
        age_module.hook_handles.append(age_module.m_conv[0].register_full_backward_hook(age_module._set_lr))
        age_module.hook_handles.append(age_module.b_conv[0].register_full_backward_hook(age_module._set_lr))

    def forward(
        self,
        F_rgb_i: torch.Tensor,
        G_z: torch.Tensor,
        epoch: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        b, c, h, w = F_rgb_i.shape
        F_down_i = self.down_proj(F_rgb_i)

        geo_map = G_z.mean(dim=1, keepdim=True)
        grad_x = F.conv2d(geo_map, self.sobel_x, padding=1)
        grad_y = F.conv2d(geo_map, self.sobel_y, padding=1)
        edge_intensity = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)

        route_logits = self.spatial_router(edge_intensity)
        route_weights = F.softmax(route_logits, dim=1)
        w_tex = route_weights[:, 0:1]
        w_geo = route_weights[:, 1:2]

        if self.texture_mode == "profile_a":
            F_tex_i = self.tex_expert(F_down_i.float()).to(F_down_i.dtype)
        else:
            F_tex_i = self.tex_expert(F_down_i)
        if isinstance(F_tex_i, tuple):
            F_tex_i = F_tex_i[0]

        F_geo_i = self.geo_expert(F_down_i, epoch=epoch, hw_range=[F_down_i.shape[2], F_down_i.shape[3]])
        if isinstance(F_geo_i, tuple):
            F_geo_i = F_geo_i[0]

        F_fused_i = F_tex_i * w_tex + F_geo_i * w_geo

        if self.texture_mode == "profile_b":
            r_tex_i = self.tex_response(F_tex_i)
            r_geo_i = self.geo_response(F_geo_i)
        else:
            r_tex_i = F_tex_i.mean(dim=1, keepdim=True)
            r_geo_i = F_geo_i.mean(dim=1, keepdim=True)
        p_tex_i = torch.sigmoid(r_tex_i)
        p_geo_i = torch.sigmoid(r_geo_i)

        A_i = torch.clamp(torch.exp(-4.0 * torch.abs(p_tex_i - p_geo_i)), 0.0, 1.0)
        C_i = torch.clamp(1.0 - A_i, 0.0, 1.0)

        S_tex_i = normalize_map(torch.mean(torch.abs(F_tex_i), dim=1, keepdim=True))
        S_geo_i = normalize_map(torch.mean(torch.abs(F_geo_i), dim=1, keepdim=True))

        K_sh_i = 0.5 * (F_tex_i + F_geo_i) * A_i
        K_tex_i = F_tex_i * torch.clamp(C_i * S_tex_i, 0.0, 1.0)
        K_geo_i = F_geo_i * torch.clamp(C_i * S_geo_i, 0.0, 1.0)
        r_sh_i = 0.5 * (r_tex_i + r_geo_i)

        router_mean = route_weights.mean(dim=(0, 2, 3))
        router_balance_loss = torch.sum((router_mean - 0.5) ** 2)
        kd_dict = {
            "shared": K_sh_i,
            "tex_private": K_tex_i,
            "geo_private": K_geo_i,
            "tex_feature": F_tex_i,
            "geo_feature": F_geo_i,
            "tex_response": r_tex_i,
            "geo_response": r_geo_i,
            "response": r_sh_i,
            "agreement": A_i,
            "complementarity": C_i,
            "shared_mask": A_i.detach(),
            "tex_private_mask": torch.clamp(C_i * S_tex_i, 0.0, 1.0).detach(),
            "geo_private_mask": torch.clamp(C_i * S_geo_i, 0.0, 1.0).detach(),
            "router_tex": w_tex,
            "router_geo": w_geo,
            "tex_strength": S_tex_i,
            "geo_strength": S_geo_i,
            "router_balance_loss": router_balance_loss,
        }

        out = self.up_proj(F_fused_i)
        out_flat = out.flatten(2).transpose(1, 2)
        residual_flat = F_rgb_i.flatten(2).transpose(1, 2)
        out = (residual_flat + self.norm(out_flat)
               if self.architecture_version == "paper_v1"
               else self.norm(out_flat + residual_flat))
        out = out.transpose(1, 2).view(b, c, h, w)
        return out, F_tex_i, F_geo_i, kd_dict
