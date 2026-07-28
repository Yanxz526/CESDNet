from __future__ import annotations

import contextlib
import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
from torch import nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

from CESDNet.config import CESDNetConfig, REPO_ROOT


_MODEL_DIRS = (
    REPO_ROOT / "toolbox" / "models" / "Ourmodel",
    REPO_ROOT / "toolbox" / "models" / "BEFORE" / "Ourmodel",
)
for model_dir in _MODEL_DIRS:
    if model_dir.exists() and str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
try:
    import toolbox.models as _toolbox_models

    before_models = REPO_ROOT / "toolbox" / "models" / "BEFORE"
    if before_models.exists() and str(before_models) not in _toolbox_models.__path__:
        _toolbox_models.__path__.append(str(before_models))
except Exception:
    pass

try:
    from ARConv import ARConv
except ImportError as exc:
    raise ImportError(f"Missing ARConv.py under {_MODEL_DIRS}") from exc


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
    try:
        from GMR_Conv.gmr_conv import GMR_Conv2d

        return GMR_Conv2d
    except ImportError:
        for candidate in (
            _MODEL_DIRS[0] / "GMR-Conv" / "src",
            _MODEL_DIRS[0] / "src",
            _MODEL_DIRS[0],
            _MODEL_DIRS[1] / "GMR_Conv_main" / "src",
            _MODEL_DIRS[1] / "GMR-Conv" / "src",
            _MODEL_DIRS[1] / "src",
            _MODEL_DIRS[1],
        ):
            if candidate.exists() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
        try:
            from GMR_Conv.gmr_conv import GMR_Conv2d

            return GMR_Conv2d
        except ImportError as exc:
            raise ImportError("Missing GMR_Conv.gmr_conv.GMR_Conv2d in the local repository") from exc


def _load_profile_texture_block():
    try:
        from SW_v2_unirep import Block

        return Block
    except ImportError as exc:
        raise ImportError(f"Missing SW_v2_unirep.py under {_MODEL_DIRS}") from exc


def _unwrap_checkpoint_state(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        return checkpoint
    for key in ("teacher_state_dict", "state_dict", "model", "net", "module"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value
    return checkpoint


def infer_teacher_use_gmr_prior(checkpoint_path: str | Path | None, default: bool = True) -> bool:
    if not checkpoint_path:
        return default
    path = Path(checkpoint_path)
    if not path.is_file():
        return default
    checkpoint = torch.load(str(path), map_location="cpu")
    state_dict = _unwrap_checkpoint_state(checkpoint)
    if not isinstance(state_dict, Mapping):
        return default
    for key, value in state_dict.items():
        if key.endswith("geo_expert.m_conv.0.weight") and hasattr(value, "ndim"):
            if value.ndim == 3:
                return True
            if value.ndim == 4:
                return False
    return default


def infer_teacher_texture_mode(checkpoint_path: str | Path | None, default: str = "profile_b") -> str:
    if not checkpoint_path:
        return default
    path = Path(checkpoint_path)
    if not path.is_file():
        return default
    checkpoint = torch.load(str(path), map_location="cpu")
    state_dict = _unwrap_checkpoint_state(checkpoint)
    if not isinstance(state_dict, Mapping):
        return default
    for key in state_dict:
        if "tex_expert.texture_backbone" in key:
            return "profile_b"
    for key in state_dict:
        if key.endswith("tex_expert.gamma") or "tex_expert.large_kernel.LoRAs" in key:
            return "profile_a"
    return default


def infer_teacher_head_profile(checkpoint_path: str | Path | None, default: str = "profile_b") -> str:
    if not checkpoint_path:
        return default
    path = Path(checkpoint_path)
    if not path.is_file():
        return default
    checkpoint = torch.load(str(path), map_location="cpu")
    state_dict = _unwrap_checkpoint_state(checkpoint)
    if not isinstance(state_dict, Mapping):
        return default
    if any(key.startswith("decoder.out_head.0.fc.") for key in state_dict):
        return "profile_a"
    if any(key.startswith("decoder.out_head.0.fc1.") for key in state_dict):
        return "profile_b"
    return default


class ConvDWT(nn.Module):
    def __init__(self, wave: str = "haar", mode: str = "zero") -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        x01 = x[:, :, 0::2, :] / 2.0
        x02 = x[:, :, 1::2, :] / 2.0
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        Y_l = x1 + x2 + x3 + x4
        H_l = -x1 - x2 + x3 + x4
        L_h = -x1 + x2 - x3 + x4
        H_h = x1 - x2 - x3 + x4
        return torch.cat((Y_l, torch.cat([H_l, L_h, H_h], dim=1)), dim=1)


class ConvIDWT(nn.Module):
    def __init__(self, wave: str = "haar", mode: str = "zero") -> None:
        super().__init__()

    def forward(self, low_freqs: torch.Tensor, high_freqs: torch.Tensor) -> torch.Tensor:
        H_l, L_h, H_h = torch.chunk(high_freqs, 3, dim=1)
        x1 = low_freqs - H_l - L_h + H_h
        x2 = low_freqs - H_l + L_h - H_h
        x3 = low_freqs + H_l - L_h - H_h
        x4 = low_freqs + H_l + L_h + H_h
        b, c, h, w = low_freqs.shape
        out = torch.zeros(b, c, h * 2, w * 2, device=low_freqs.device, dtype=low_freqs.dtype)
        out[:, :, 0::2, 0::2] = x1
        out[:, :, 1::2, 0::2] = x2
        out[:, :, 0::2, 1::2] = x3
        out[:, :, 1::2, 1::2] = x4
        return out


class GRNwithNHWC(nn.Module):
    def __init__(self, dim: int, use_bias: bool = True) -> None:
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        if use_bias:
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        out = (self.gamma * nx + 1) * x
        if self.use_bias:
            out = out + self.beta
        return out


class NCHWtoNHWC(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 1)


class NHWCtoNCHW(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2)


class SEBlock(nn.Module):
    def __init__(self, input_channels: int, internal_neurons: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(input_channels, internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(internal_neurons, input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels
        self.nonlinear = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x = self.down(x)
        x = self.nonlinear(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        return inputs * x.view(-1, self.input_channels, 1, 1)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


def get_texture_bn(channels: int) -> nn.Module:
    return nn.SyncBatchNorm(channels)


class SWBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        kernel_size: tuple[int, int] = (7, 7),
        ghost_ratio: float = 0.23,
        N_path: int = 2,
        N_rep: int = 4,
        bn: bool = True,
        deploy: bool = False,
        use_lk_impl: bool = False,
    ) -> None:
        super().__init__()
        del ghost_ratio, N_path, N_rep, bn
        if use_lk_impl:
            raise ValueError("CESDNet teacher only supports SWBlock(use_lk_impl=False).")

        self.large_kernel = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size[1],
            stride=1,
            padding=kernel_size[1] // 2,
            groups=dim,
            bias=False,
        )
        self.norm = get_texture_bn(dim)
        self.se = SEBlock(dim, dim // 4)
        self.pwconv1 = nn.Sequential(NCHWtoNHWC(), nn.Linear(dim, 4 * dim))
        self.act = nn.Sequential(nn.GELU(), GRNwithNHWC(4 * dim, use_bias=not deploy))
        self.pwconv2 = nn.Sequential(nn.Linear(4 * dim, dim, bias=False), NHWCtoNCHW(), get_texture_bn(dim))
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.large_kernel(x)
        x = self.se(self.norm(x))
        x = self.pwconv2(self.act(self.pwconv1(x)))
        if self.gamma is not None:
            x = self.gamma.view(1, -1, 1, 1) * x
        return residual + self.drop_path(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7, bn_before_sigmoid: bool = False) -> None:
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.bn_before_sigmoid = bn_before_sigmoid
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        if bn_before_sigmoid:
            self.bn = nn.BatchNorm2d(1)
            self.bn.bias.data.fill_(0)
            self.bn.bias.requires_grad = False
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = self.conv1(torch.cat([avg_out, max_out], dim=1))
        if self.bn_before_sigmoid:
            x = self.bn(x)
        return self.sigmoid(x)


class LearnableGaussianFilterBank(nn.Module):
    def __init__(self, kernel_size: int, num_filters: int, num_channels: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.C = num_channels
        self.padding = kernel_size // 2
        self.sigmas = nn.ParameterList([nn.Parameter(torch.tensor([1.0])) for _ in range(num_filters)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = [
            self._gaussian_kernel(self.kernel_size, sigma).repeat(self.C, 1, 1, 1)
            for sigma in self.sigmas
        ]
        filtered_outputs = [
            F.conv2d(
                F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode="replicate"),
                weight.to(device=x.device, dtype=x.dtype),
                groups=self.C,
            )
            for weight in weights
        ]
        return torch.cat(filtered_outputs, dim=1)

    @staticmethod
    def _gaussian_kernel(kernel_size: int, sigma: torch.Tensor) -> torch.Tensor:
        device, dtype = sigma.device, sigma.dtype
        center = kernel_size // 2
        ar = torch.arange(kernel_size, device=device, dtype=dtype) - center
        yy = ar.view(-1, 1).expand(kernel_size, kernel_size)
        xx = ar.view(1, -1).expand(kernel_size, kernel_size)
        kernel = torch.exp(-(yy**2 + xx**2) / (2 * sigma**2 + 1e-12))
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        return kernel / (kernel.sum() + 1e-12)


class LFP_TextureExpert(nn.Module):
    def __init__(
        self,
        in_channels: int,
        wave: str = "haar",
        mode: str = "symmetric",
        with_gauss: bool = True,
        gauss_gate: float = 0.5,
    ) -> None:
        super().__init__()
        self.dwt = ConvDWT(wave=wave, mode=mode)
        self.idwt = ConvIDWT(wave=wave, mode=mode)
        self.with_gauss = with_gauss
        self.gauss_gate = gauss_gate
        self.texture_backbone = SWBlock(dim=in_channels, use_lk_impl=False)
        self.attention = SpatialAttention()
        if self.with_gauss:
            self.gaussian_filter = LearnableGaussianFilterBank(
                kernel_size=3,
                num_filters=1,
                num_channels=3 * in_channels,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, c, _, _ = x.shape
        dwt_out = self.dwt(x)
        low_freq = dwt_out[:, :c]
        high_freq = dwt_out[:, c:]

        clean_low = self.texture_backbone(low_freq)
        if isinstance(clean_low, tuple):
            clean_low = clean_low[0]

        high_freq = high_freq * self.attention(clean_low)
        if self.with_gauss:
            blurred = self.gaussian_filter(high_freq)
            mask = (high_freq.abs() < self.gauss_gate).float()
            high_freq = high_freq * (1 - mask) + blurred * mask
        return self.idwt(clean_low, high_freq)


class OrthogonalDualExpertAdapter(nn.Module):
    def __init__(
        self,
        dim: int,
        reduction_dim: int = 256,
        use_gmr_prior: bool = True,
        texture_mode: str = "profile_b",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.reduction_dim = reduction_dim
        self.use_gmr_prior = use_gmr_prior
        self.texture_mode = texture_mode

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
            self.tex_expert = LFP_TextureExpert(in_channels=reduction_dim)
        else:
            raise ValueError(f"Unsupported teacher profile: {self.texture_mode}")
        self.geo_expert = ARConv(inc=reduction_dim, outc=reduction_dim)
        if self.use_gmr_prior:
            self._inject_gmr_prior(self.geo_expert, reduction_dim)

        self.up_proj = nn.Conv2d(reduction_dim, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)
        if self.texture_mode == "profile_b":
            self.tex_response = nn.Conv2d(reduction_dim, 1, kernel_size=1)
            self.geo_response = nn.Conv2d(reduction_dim, 1, kernel_size=1)

    @staticmethod
    def _inject_gmr_prior(arconv_module: nn.Module, dim: int) -> None:
        GMR_Conv2d = _load_gmr_conv2d()
        arconv_module.m_conv[0] = GMR_Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=1,
            stride=1,
            dw_fwd=False,
        )
        arconv_module.b_conv[0] = GMR_Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=1,
            stride=1,
            dw_fwd=False,
        )
        arconv_module.hook_handles.append(arconv_module.m_conv[0].register_full_backward_hook(arconv_module._set_lr))
        arconv_module.hook_handles.append(arconv_module.b_conv[0].register_full_backward_hook(arconv_module._set_lr))

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
        out = self.norm(out_flat + residual_flat)
        out = out.transpose(1, 2).view(b, c, h, w)
        return out, F_tex_i, F_geo_i, kd_dict


class GeometricPriorEncoder(nn.Module):
    def __init__(self, in_chans: int = 1, embed_dim: int = 1024, patch_size: int = 16) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, embed_dim, kernel_size=patch_size // 2, stride=patch_size // 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj(self.stem(z))


class DualPathPatchEmbed(nn.Module):
    def __init__(self, original_patch_embed: nn.Module, embed_dim: int = 1024, ndsm_in_chans: int = 1) -> None:
        super().__init__()
        self.rgb_embed = original_patch_embed
        self.embed_dim = embed_dim
        self.ndsm_in_chans = ndsm_in_chans
        if hasattr(original_patch_embed, "proj"):
            kernel_size = original_patch_embed.proj.kernel_size
            stride = original_patch_embed.proj.stride
            padding = original_patch_embed.proj.padding
            dim = original_patch_embed.proj.out_channels
        else:
            kernel_size, stride, padding, dim = (14, 14), (14, 14), (0, 0), embed_dim
        self.ndsm_embed = nn.Conv2d(ndsm_in_chans, dim, kernel_size=kernel_size, stride=stride, padding=padding)
        nn.init.constant_(self.ndsm_embed.weight, 0.0)
        nn.init.constant_(self.ndsm_embed.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_rgb, x_ndsm = x[:, :3], x[:, 3:]
        rgb_token = self.rgb_embed(x_rgb)
        ndsm_token = self.ndsm_embed(x_ndsm)
        if rgb_token.ndim == 4 and rgb_token.shape[-1] == self.embed_dim:
            ndsm_token = ndsm_token.permute(0, 2, 3, 1)
        elif rgb_token.ndim == 4 and rgb_token.shape[1] == self.embed_dim:
            pass
        elif rgb_token.ndim == 3:
            ndsm_token = ndsm_token.flatten(2).transpose(1, 2)
            if rgb_token.shape[1] > ndsm_token.shape[1]:
                diff = rgb_token.shape[1] - ndsm_token.shape[1]
                zeros = torch.zeros(
                    (ndsm_token.shape[0], diff, ndsm_token.shape[2]),
                    device=ndsm_token.device,
                    dtype=ndsm_token.dtype,
                )
                ndsm_token = torch.cat([zeros, ndsm_token], dim=1)
        return rgb_token + ndsm_token


class ChannelAttention(nn.Module):
    def __init__(self, dim: int, ratio: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(dim, dim // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(dim // ratio, dim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out) * x


class ChannelAttentionProfileA(nn.Module):
    def __init__(self, dim: int, reduction: int = 8) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim // reduction, dim, kernel_size=1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(torch.mean(x, dim=(2, 3), keepdim=True))
        max_out = self.fc(torch.amax(x, dim=(2, 3), keepdim=True))
        return x * self.sigmoid(avg_out + max_out)


class TSMD_Decoder(nn.Module):
    def __init__(
        self,
        in_dims: list[int],
        embed_dim: int = 256,
        num_classes: int = 6,
        head_profile: str = "profile_b",
    ) -> None:
        super().__init__()
        self.lat_layers = nn.ModuleList([nn.Conv2d(dim, embed_dim, 1) for dim in in_dims])
        self.aggr = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
        )
        if head_profile == "profile_a":
            attention = ChannelAttentionProfileA(embed_dim)
        elif head_profile == "profile_b":
            attention = ChannelAttention(embed_dim)
        else:
            raise ValueError(f"Unsupported teacher head profile: {head_profile}")
        self.out_head = nn.Sequential(
            attention,
            nn.ConvTranspose2d(embed_dim, embed_dim, 2, stride=2),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, num_classes, 2, stride=2),
        )

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        target_size = features[0].shape[-2:]
        projs = [layer(feat) for layer, feat in zip(self.lat_layers, features)]
        concat_feat = torch.cat(
            [F.interpolate(proj, size=target_size, mode="bilinear") for proj in projs],
            dim=1,
        )
        return self.out_head(self.aggr(concat_feat))


class MultiClassMorphology(nn.Module):
    def __init__(self, num_classes: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.k = kernel_size
        self.pad = (kernel_size - 1) // 2
        self.alpha_raw = nn.Parameter(torch.tensor(0.8))
        self.alpha_open = nn.Parameter(torch.tensor(0.1))
        self.alpha_close = nn.Parameter(torch.tensor(0.1))
        self.class_gate = nn.Parameter(torch.ones(1, num_classes, 1, 1))

    def dilation(self, x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(x, kernel_size=self.k, stride=1, padding=self.pad)

    def erosion(self, x: torch.Tensor) -> torch.Tensor:
        return -F.max_pool2d(-x, kernel_size=self.k, stride=1, padding=self.pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dilated = self.dilation(x)
        closed = self.erosion(dilated)
        eroded = self.erosion(x)
        opened = self.dilation(eroded)
        weights = torch.softmax(torch.stack([self.alpha_raw, self.alpha_open, self.alpha_close]), dim=0)
        out = weights[0] * x + weights[1] * opened + weights[2] * closed
        return x + self.class_gate * (out - x)


class ImprovedExpertNet(nn.Module):
    def __init__(
        self,
        repo_dir: str | Path,
        weights_path: str | Path,
        model_name: str = "dinov3_vitl16",
        num_classes: int = 6,
        lora_r: int = 4,
        sar_in_chans: int = 1,
        use_gmr_prior: bool = True,
        texture_mode: str = "profile_b",
        head_profile: str = "profile_b",
    ) -> None:
        super().__init__()
        if isinstance(repo_dir, tuple):
            repo_dir = repo_dir[0]
        if isinstance(weights_path, tuple):
            weights_path = weights_path[0]
        self.sar_in_chans = sar_in_chans
        self.patch_size = 16 if "16" in model_name else 14
        self.backbone = self._load_backbone(repo_dir, weights_path, model_name, lora_r)

        if hasattr(self.backbone, "base_model") and hasattr(self.backbone.base_model, "model"):
            embed_dim = self.backbone.base_model.model.embed_dim
        elif hasattr(self.backbone, "embed_dim"):
            embed_dim = self.backbone.embed_dim
        else:
            embed_dim = 1024

        self.geo_encoder = GeometricPriorEncoder(
            in_chans=self.sar_in_chans,
            embed_dim=embed_dim,
            patch_size=self.patch_size,
        )
        self.extract_indices = [5, 11, 17, 23]
        self.adapters = nn.ModuleList(
            [
                OrthogonalDualExpertAdapter(
                    dim=embed_dim,
                    reduction_dim=256,
                    use_gmr_prior=use_gmr_prior,
                    texture_mode=texture_mode,
                )
                for _ in range(4)
            ]
        )
        self.decoder = TSMD_Decoder(
            in_dims=[embed_dim] * 4,
            embed_dim=256,
            num_classes=num_classes,
            head_profile=head_profile,
        )
        self.lmm = MultiClassMorphology(num_classes=num_classes, kernel_size=3)
        self.ortho_loss = torch.tensor(0.0)
        self.comp_loss = torch.tensor(0.0)

    def _load_backbone(
        self,
        repo_dir: str | Path,
        weights_path: str | Path,
        model_name: str,
        lora_r: int,
    ) -> nn.Module:
        print(f"Loading Backbone from local repo: {repo_dir}")
        try:
            model = torch.hub.load(str(repo_dir), model_name, source="local", pretrained=False)
        except Exception as exc:
            print(f"Warning: failed to load local DINOv3 ({exc}). Using timm ViT fallback.")
            import timm

            model = timm.create_model("vit_large_patch16_256", pretrained=False)

        if os.path.exists(weights_path):
            try:
                state_dict = torch.load(weights_path, map_location="cpu")
                if "model" in state_dict:
                    state_dict = state_dict["model"]
                if "teacher" in state_dict:
                    state_dict = state_dict["teacher"]
                model.load_state_dict(state_dict, strict=False)
            except Exception:
                pass

        embed_dim = getattr(model, "embed_dim", 1024)
        model.patch_embed = DualPathPatchEmbed(
            model.patch_embed,
            embed_dim=embed_dim,
            ndsm_in_chans=self.sar_in_chans,
        )
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_r * 2,
            target_modules=["qkv", "proj", "fc1", "fc2"],
            lora_dropout=0.1,
            bias="lora_only",
            modules_to_save=["patch_embed"],
        )
        return get_peft_model(model, peft_config)

    def forward(
        self,
        rgb: torch.Tensor,
        ndsm: torch.Tensor,
        gt_mask: Optional[torch.Tensor] = None,
        return_feats: bool = False,
        return_kd: bool = False,
        epoch: int = 1,
    ):
        device = rgb.device
        x_input = torch.cat([rgb, ndsm], dim=1)
        G_z = self.geo_encoder(ndsm)

        if hasattr(self.backbone, "base_model") and hasattr(self.backbone.base_model, "model"):
            vit_model = self.backbone.base_model.model
        else:
            vit_model = self.backbone

        if hasattr(vit_model, "get_intermediate_layers"):
            features_seq = vit_model.get_intermediate_layers(
                x_input,
                n=self.extract_indices,
                reshape=False,
                return_class_token=False,
                norm=True,
            )
        else:
            features_seq = [vit_model.forward_features(x_input)] * 4

        h_out = rgb.shape[-2] // self.patch_size
        w_out = rgb.shape[-1] // self.patch_size
        enhanced_feats = []
        kd_features = []
        self.ortho_loss = torch.tensor(0.0, device=device)
        self.comp_loss = torch.tensor(0.0, device=device)

        for feat_seq, adapter in zip(features_seq, self.adapters):
            b, n, c = feat_seq.shape
            if n == h_out * w_out + 1:
                feat_seq = feat_seq[:, 1:]
            F_rgb_i = feat_seq.transpose(1, 2).view(b, c, h_out, w_out)
            F_enh_i, F_tex_i, F_geo_i, kd_dict = adapter(F_rgb_i, G_z, epoch=epoch)
            self.ortho_loss = self.ortho_loss + orthogonal_loss(F_tex_i, F_geo_i)
            self.comp_loss = self.comp_loss + kd_dict.get("router_balance_loss", torch.tensor(0.0, device=device))
            enhanced_feats.append(F_enh_i)
            kd_features.append(kd_dict)

        logits = self.decoder(enhanced_feats)
        if logits.shape[-2:] != rgb.shape[-2:]:
            logits = F.interpolate(logits, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        O_t = self.lmm(logits)

        if return_kd:
            stage_names = [f"b{idx + 1}" for idx in self.extract_indices]
            stage_kd = {name: kd for name, kd in zip(stage_names, kd_features)}
            kd_out = dict(kd_features[-1])
            kd_out.update(
                {
                    "kd_schema_version": "netv14_original_position_shared_private_v1",
                    "stages": stage_kd,
                    "all": kd_features,
                    "stage_names": stage_names,
                    "adapter_block_indices": list(self.extract_indices),
                    "c3": kd_features[-2],
                    "c4": kd_features[-1],
                }
            )
            return O_t, kd_out
        if return_feats:
            return O_t, enhanced_feats[-1]
        return O_t


TPE = LFP_TextureExpert
GeoMoE = OrthogonalDualExpertAdapter


def _stage_aliases(stage_kd: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = dict(stage_kd)
    if "shared" in stage_kd:
        out.setdefault("K_sh", stage_kd["shared"])
    if "tex_private" in stage_kd:
        out.setdefault("K_tex", stage_kd["tex_private"])
    if "geo_private" in stage_kd:
        out.setdefault("K_geo", stage_kd["geo_private"])
    if "agreement" in stage_kd:
        out.setdefault("A", stage_kd["agreement"])
    if "complementarity" in stage_kd:
        out.setdefault("C", stage_kd["complementarity"])
    if "tex_feature" in stage_kd:
        out.setdefault("F_tex", stage_kd["tex_feature"])
    if "geo_feature" in stage_kd:
        out.setdefault("F_geo", stage_kd["geo_feature"])
    if "response" in stage_kd:
        out.setdefault("r_sh", stage_kd["response"])
    return out


def normalize_teacher_kd_tree(kd_tree: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = dict(kd_tree)
    for stage in ("c3", "c4"):
        if isinstance(normalized.get(stage), Mapping):
            normalized[stage] = _stage_aliases(normalized[stage])

    stages = normalized.get("stages")
    if isinstance(stages, Mapping):
        normalized["stages"] = {
            key: _stage_aliases(value) if isinstance(value, Mapping) else value
            for key, value in stages.items()
        }

    all_stages = normalized.get("all")
    if isinstance(all_stages, list):
        normalized["all"] = [
            _stage_aliases(value) if isinstance(value, Mapping) else value
            for value in all_stages
        ]
    normalized.setdefault("kd_schema_version", "cesdnet_t_geomoE_v1")
    return normalized


class CESDNetT(ImprovedExpertNet):
    def __init__(
        self,
        repo_dir: str | Path,
        weights_path: str | Path,
        num_classes: int = 6,
        model_name: str = "dinov3_vitl16",
        lora_r: int = 4,
        ndsm_channels: int = 1,
        use_gmr_prior: bool = True,
        texture_mode: str = "profile_b",
        head_profile: str = "profile_b",
    ) -> None:
        super().__init__(
            str(repo_dir),
            str(weights_path),
            model_name=model_name,
            num_classes=num_classes,
            lora_r=lora_r,
            sar_in_chans=ndsm_channels,
            use_gmr_prior=use_gmr_prior,
            texture_mode=texture_mode,
            head_profile=head_profile,
        )

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        return_kd: bool = False,
        return_features: bool = False,
        epoch: int = 1,
    ):
        if return_kd:
            O_t, kd_tree = super().forward(x, z, gt_mask=y, return_kd=True, epoch=epoch)
            return O_t, normalize_teacher_kd_tree(kd_tree)
        if return_features:
            return super().forward(x, z, gt_mask=y, return_feats=True, epoch=epoch)
        return super().forward(x, z, gt_mask=y, epoch=epoch)


def build_cesdnet_t(
    config: CESDNetConfig,
    use_gmr_prior: bool = True,
    texture_mode: str = "profile_b",
    head_profile: str = "profile_b",
) -> CESDNetT:
    return CESDNetT(
        repo_dir=config.teacher_repo,
        weights_path=config.teacher_backbone,
        num_classes=config.num_classes,
        use_gmr_prior=use_gmr_prior,
        texture_mode=texture_mode,
        head_profile=head_profile,
    )
