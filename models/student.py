from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import torch
from torch import nn
import torch.nn.functional as F

from CESDNet.config import CESDNetConfig
from toolbox.backbone.DFormer1.DFormer import DFormer_Tiny


def _unwrap_checkpoint_state(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Student checkpoint must contain a tensor state dictionary")
    for key in ("student_state_dict", "state_dict", "model", "net", "module"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value
    return checkpoint


def infer_student_architecture(checkpoint_path: str | Path | None, default: str = "paper") -> str:
    if not checkpoint_path:
        return default
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Student checkpoint not found: {path}")
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    state_dict = _unwrap_checkpoint_state(checkpoint)
    state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    if any(key.startswith("hfip3.") or key.startswith("hfip4.") for key in state_dict):
        return "hfip"
    if any(key.startswith("cafe3.") or key.startswith("cafe4.") for key in state_dict):
        return "cafe"
    if any(key.startswith("enhance_c3.") or key.startswith("enhance_c4.") for key in state_dict):
        return "paper"
    raise ValueError(f"Unrecognized CESDNet student architecture in {path}")


def load_dformer_weight(backbone: nn.Module, path: Optional[str | Path]) -> Dict[str, Any]:
    if not path:
        raise ValueError("DFormer pretrained initialization requires an explicit weight file")
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"DFormer pretrained weights not found: {path}")
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(checkpoint, Mapping):
        state_dict = checkpoint.get("state_dict_ema", checkpoint.get("state_dict", checkpoint))
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError(f"DFormer checkpoint contains no state dictionary: {path}")
    stripped = {}
    for key, value in state_dict.items():
        while key.startswith(("module.", "backbone.")):
            key = key.split(".", 1)[1]
        if key in stripped:
            raise ValueError(f"Duplicate DFormer parameter after prefix normalization: {key}")
        stripped[key] = value
    expected = backbone.state_dict()
    missing = sorted(set(expected) - set(stripped))
    mismatched = [key for key in expected if key in stripped and
                  (not isinstance(stripped[key], torch.Tensor) or
                   expected[key].shape != stripped[key].shape)]
    # The upstream ImageNet checkpoint includes a classification head and
    # auxiliary-stream tensors unused by the last (optical-output-only) block.
    unused_prefixes = (
        "pred.", "stages.3.1.attn.proj_e.", "stages.3.1.mlp_e2.",
    )
    unused_names = {"stages.3.1.layer_scale_1_e", "stages.3.1.layer_scale_2_e"}
    unexpected = sorted(key for key in set(stripped) - set(expected)
                        if key not in unused_names and not key.startswith(unused_prefixes))
    if missing or mismatched or unexpected:
        raise RuntimeError(
            f"DFormer-Tiny checkpoint mismatch: missing={missing[:8]}, "
            f"shape_mismatch={mismatched[:8]}, unexpected={unexpected[:8]}"
        )
    backbone.load_state_dict({key: stripped[key] for key in expected}, strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(block)
    report = {
        "pretrained": True, "path": str(path.resolve()), "sha256": digest.hexdigest(),
        "loaded_tensors": len(expected),
        "ignored_source_keys": sorted(set(stripped) - set(expected)),
    }
    print(f"Loaded DFormer-Tiny: {len(expected)}/{len(expected)} tensors; sha256={report['sha256']}")
    return report


class StudentFeatureEnhancer(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, F_s_i: torch.Tensor) -> torch.Tensor:
        A_c_i = self.channel_attn(F_s_i) * F_s_i
        A_sp_i = self.spatial_attn(F_s_i) * F_s_i
        return F_s_i + self.gamma * self.out_proj(torch.cat([A_c_i, A_sp_i], dim=1))


class MultiLevelDecoder(nn.Module):
    def __init__(self, in_channels: List[int], embed_dim: int = 64, num_classes: int = 6) -> None:
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(ch, embed_dim, kernel_size=1) for ch in in_channels])
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels), embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, num_classes, kernel_size=1),
        )

    def forward(self, features: List[torch.Tensor], target_size: tuple[int, int]) -> torch.Tensor:
        fuse_size = features[0].shape[-2:]
        projected = [layer(feat) for layer, feat in zip(self.proj, features)]
        projected = [
            F.interpolate(feat, size=fuse_size, mode="bilinear", align_corners=False)
            for feat in projected
        ]
        logits = self.fuse(torch.cat(projected, dim=1))
        return F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)


class HighFreqImplicitPrompt(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        kernel = torch.tensor([[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]])
        self.register_buffer("laplacian", kernel.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1))
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge = F.conv2d(x, self.laplacian, padding=1, groups=x.size(1))
        weight = self.attn(edge)
        return x + self.gamma * (edge * weight)


class HFIPDecoder(nn.Module):
    def __init__(self, in_dims: List[int], embed_dim: int = 256, num_classes: int = 6) -> None:
        super().__init__()
        self.lat_layers = nn.ModuleList([nn.Conv2d(dim, embed_dim, 1) for dim in in_dims])
        self.aggr = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.attention = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 8, embed_dim, 1),
            nn.Sigmoid(),
        )
        self.out_head = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, 2, stride=2),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, num_classes, 2, stride=2),
        )

    def forward(self, features: List[torch.Tensor], target_size: tuple[int, int]) -> torch.Tensor:
        stage_size = features[0].shape[-2:]
        projs = [layer(feat) for layer, feat in zip(self.lat_layers, features)]
        concat_feat = torch.cat(
            [F.interpolate(proj, size=stage_size, mode="bilinear", align_corners=False) for proj in projs],
            dim=1,
        )
        feat = self.aggr(concat_feat)
        attn = self.attention(F.adaptive_avg_pool2d(feat, 1)) + self.attention(F.adaptive_max_pool2d(feat, 1))
        logits = self.out_head(feat * attn)
        if logits.shape[-2:] != target_size:
            logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, in_channels: int = 2, N_freqs: int = 64, logscale: bool = True) -> None:
        super().__init__()
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)
        if logscale:
            freq_bands = 2 ** torch.linspace(0, N_freqs, N_freqs)
        else:
            freq_bands = torch.linspace(1, 2**N_freqs, N_freqs)
        self.register_buffer("freq_bands", freq_bands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        for freq in self.freq_bands:
            for func in self.funcs:
                out.append(func(freq * x))
        return torch.cat(out, dim=-1)


class CAFEBlock(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, num_branches: int = 3) -> None:
        super().__init__()
        self.num_branches = num_branches
        self.branches = nn.ModuleList(
            [nn.Linear(in_features, hidden_features, bias=True) for _ in range(num_branches)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.branches[0](x)
        for idx in range(1, self.num_branches):
            out = out * self.branches[idx](x)
        return out


class BackboneMLP(nn.Module):
    def __init__(self, hidden_features: int, hidden_layers: int = 1) -> None:
        super().__init__()
        layers = []
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_features, hidden_features, bias=True))
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers) if hidden_layers > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PureCAFEEnhancer(nn.Module):
    def __init__(self, in_channels: int, N_freqs: int = 4, hidden_dim: int = 128, num_branches: int = 3) -> None:
        super().__init__()
        self.pe = PositionalEncoding(in_channels=2, N_freqs=N_freqs, logscale=True)
        encoded_dim = 2 * 2 * N_freqs
        total_in_features = in_channels + encoded_dim
        self.mult_block = CAFEBlock(
            in_features=total_in_features,
            hidden_features=hidden_dim,
            num_branches=num_branches,
        )
        self.backbone = BackboneMLP(hidden_features=hidden_dim, hidden_layers=1)
        self.out_linear = nn.Linear(hidden_dim, in_channels)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.hf_residual = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        y_coord = torch.linspace(-1, 1, h, device=x.device)
        x_coord = torch.linspace(-1, 1, w, device=x.device)
        yy, xx = torch.meshgrid(y_coord, x_coord, indexing="ij")
        coords = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
        fourier_feats = self.pe(coords)
        x_permuted = x.permute(0, 2, 3, 1)
        implicit_input = torch.cat([x_permuted, fourier_feats], dim=-1)
        feat = self.mult_block(implicit_input)
        feat = self.backbone(feat)
        out_feat = self.out_linear(feat).permute(0, 3, 1, 2)
        self.hf_residual = self.gamma * out_feat
        return x + self.hf_residual


class CAFEDecoder(nn.Module):
    def __init__(self, in_dims: List[int], embed_dim: int = 256, num_classes: int = 6) -> None:
        super().__init__()
        self.lat_layers = nn.ModuleList([nn.Conv2d(dim, embed_dim, 1) for dim in in_dims])
        self.aggr = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_dims), embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, num_classes, kernel_size=1),
        )

    def forward(self, features: List[torch.Tensor], target_size: tuple[int, int]) -> torch.Tensor:
        fuse_size = features[0].shape[-2:]
        projs = [layer(feat) for layer, feat in zip(self.lat_layers, features)]
        concat_feat = torch.cat(
            [F.interpolate(proj, size=fuse_size, mode="bilinear", align_corners=False) for proj in projs],
            dim=1,
        )
        logits = self.aggr(concat_feat)
        return F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)


class CESDNetS(nn.Module):
    def __init__(
        self,
        num_classes: int = 6,
        dformer_weight: Optional[str | Path] = None,
        use_pretrained: bool = True,
        student_arch: str = "paper",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.student_arch = student_arch
        self.stage_channels = {"c1": 32, "c2": 64, "c3": 128, "c4": 256}
        self.backbone = DFormer_Tiny(pretrained=False)
        self.initialization_report = {"pretrained": False}
        if use_pretrained:
            self.initialization_report = load_dformer_weight(self.backbone, dformer_weight)
        channels = list(self.stage_channels.values())
        if student_arch == "paper":
            self.enhance_c3 = StudentFeatureEnhancer(self.stage_channels["c3"])
            self.enhance_c4 = StudentFeatureEnhancer(self.stage_channels["c4"])
            self.decoder = MultiLevelDecoder(channels, embed_dim=64, num_classes=num_classes)
        elif student_arch == "hfip":
            self.hfip3 = HighFreqImplicitPrompt(self.stage_channels["c3"])
            self.hfip4 = HighFreqImplicitPrompt(self.stage_channels["c4"])
            self.decoder = HFIPDecoder(channels, embed_dim=256, num_classes=num_classes)
        elif student_arch == "cafe":
            self.cafe3 = PureCAFEEnhancer(self.stage_channels["c3"])
            self.cafe4 = PureCAFEEnhancer(self.stage_channels["c4"])
            self.decoder = CAFEDecoder(channels, embed_dim=64, num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported student architecture: {student_arch}")

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_feats: bool = False,
    ):
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"CESDNet-S requires Bx3xHxW optical input, got {tuple(x.shape)}")
        target_size = x.shape[-2:]
        c1, c2, c3, c4 = self.backbone(x, x_e=None)
        if self.student_arch == "paper":
            c3_enh = self.enhance_c3(c3)
            c4_enh = self.enhance_c4(c4)
            O_s = self.decoder([c1, c2, c3_enh, c4_enh], target_size=target_size)
        elif self.student_arch == "hfip":
            c3_enh = self.hfip3(c3)
            c4_enh = self.hfip4(c4)
            O_s = self.decoder([c1, c2, c3_enh, c4_enh], target_size=target_size)
        else:
            c3_enh = self.cafe3(c3)
            c4_enh = self.cafe4(c4)
            O_s = self.decoder([c1, c2, c3_enh, c4_enh], target_size=target_size)
        if return_features or return_feats:
            F_s = {"c1": c1, "c2": c2, "c3": c3, "c4": c4}
            F_s_enh = {"c3": c3_enh, "c4": c4_enh}
            result: Dict[str, object] = {"O_s": O_s, "logits": O_s, "F_s": F_s, "F_s_enh": F_s_enh}
            if return_feats and not return_features:
                return O_s, c4_enh
            return result
        return O_s


CESDNetKD = CESDNetS


def build_cesdnet_s(
    config: CESDNetConfig,
    use_pretrained: bool = True,
    student_arch: Optional[str] = None,
) -> CESDNetS:
    return CESDNetS(
        num_classes=config.num_classes,
        dformer_weight=config.dformer_weight,
        use_pretrained=use_pretrained,
        student_arch=student_arch or config.student_arch,
    )
