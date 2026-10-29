"""Texture-purification expert (TPE), equations (10)-(13)."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


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
    def __init__(self, wave: str = "haar", mode: str = "zero", architecture_version: str = "paper_v1") -> None:
        super().__init__()
        self.architecture_version = architecture_version

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
        return out * (0.5 if self.architecture_version == "paper_v1" else 1.0)


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


class LowFrequencyRefinement(nn.Module):
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
            raise ValueError("CESDNet teacher only supports LowFrequencyRefinement(use_lk_impl=False).")

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


class TPE(nn.Module):
    """Texture-purification expert: DWT guidance, soft smoothing and IDWT."""
    def __init__(
        self,
        in_channels: int,
        wave: str = "haar",
        mode: str = "symmetric",
        with_gauss: bool = True,
        gauss_gate: float = 0.5,
        architecture_version: str = "paper_v1",
        gate_slope: float = 10.0,
    ) -> None:
        super().__init__()
        self.dwt = ConvDWT(wave=wave, mode=mode)
        self.architecture_version = architecture_version
        self.gate_slope = gate_slope
        self.idwt = ConvIDWT(wave=wave, mode=mode, architecture_version=architecture_version)
        self.with_gauss = with_gauss
        self.gauss_gate = gauss_gate
        self.texture_backbone = LowFrequencyRefinement(dim=in_channels, use_lk_impl=False)
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
            if self.architecture_version == "paper_v1":
                # Eq. (11): delta=10, nu=0.5 are explicit implementation settings.
                mask = torch.sigmoid(self.gate_slope * (self.gauss_gate - high_freq.abs()))
            else:
                mask = (high_freq.abs() < self.gauss_gate).float()
            if self.architecture_version == "paper_v1":
                high_freq = high_freq + mask * (blurred - high_freq)
            else:
                high_freq = high_freq * (1 - mask) + blurred * mask
        low_for_reconstruction = low_freq if self.architecture_version == "paper_v1" else clean_low
        return self.idwt(low_for_reconstruction, high_freq)
