from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
from torch import nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

from CESDNet.config import CESDNetConfig, REPO_ROOT


from toolbox.models.Ourmodel.GeoMoE import GeoMoE, normalize_map, orthogonal_loss
from toolbox.models.Ourmodel.TPE import (
    ConvDWT, ConvIDWT, DropPath, GRNwithNHWC, LearnableGaussianFilterBank,
    LowFrequencyRefinement as SWBlock, NCHWtoNHWC, NHWCtoNCHW,
    SEBlock, SpatialAttention, TPE, get_texture_bn,
)


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
        raise FileNotFoundError(f"Teacher checkpoint not found: {path}")
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    state_dict = _unwrap_checkpoint_state(checkpoint)
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"Teacher checkpoint must contain a state dictionary: {path}")
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
        raise FileNotFoundError(f"Teacher checkpoint not found: {path}")
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    state_dict = _unwrap_checkpoint_state(checkpoint)
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"Teacher checkpoint must contain a state dictionary: {path}")
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
        raise FileNotFoundError(f"Teacher checkpoint not found: {path}")
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    state_dict = _unwrap_checkpoint_state(checkpoint)
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"Teacher checkpoint must contain a state dictionary: {path}")
    if any(key.startswith("decoder.out_head.0.fc.") for key in state_dict):
        return "profile_a"
    if any(key.startswith("decoder.out_head.0.fc1.") for key in state_dict):
        return "profile_b"
    return default


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
        architecture_version: str = "paper_v1",
        checkpoint_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        if isinstance(repo_dir, tuple):
            repo_dir = repo_dir[0]
        if isinstance(weights_path, tuple):
            weights_path = weights_path[0]
        if architecture_version not in ("paper_v1", "legacy_v0"):
            raise ValueError(f"Unsupported architecture version: {architecture_version}")
        if architecture_version == "paper_v1" and texture_mode != "profile_b":
            raise ValueError("paper_v1 requires the wavelet TPE (texture_mode='profile_b').")
        self.architecture_version = architecture_version
        self._checkpoint_verified = checkpoint_path is None
        self.sar_in_chans = sar_in_chans
        self.patch_size = 16
        self.backbone = self._load_backbone(repo_dir, weights_path, model_name, lora_r, checkpoint_path)

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
                GeoMoE(
                    dim=embed_dim,
                    reduction_dim=256,
                    use_gmr_prior=use_gmr_prior,
                    texture_mode=texture_mode,
                    architecture_version=architecture_version,
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
        weights_path: str | Path | None,
        model_name: str,
        lora_r: int,
        checkpoint_path: str | Path | None = None,
    ) -> nn.Module:
        if model_name != "dinov3_vitl16":
            raise ValueError(f"CESDNet requires dinov3_vitl16, got {model_name!r}.")
        repo_dir = Path(repo_dir)
        if not (repo_dir / "hubconf.py").is_file():
            raise FileNotFoundError(f"DINOv3 local source is incomplete: {repo_dir / 'hubconf.py'}")
        if checkpoint_path is not None and not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")
        pretrained_path = Path(weights_path) if weights_path else None
        if checkpoint_path is None and (pretrained_path is None or not pretrained_path.is_file()):
            raise FileNotFoundError(
                f"DINOv3 pretrained weights are required for teacher training: {weights_path!r}. "
                "Pass --teacher-backbone, or provide a complete teacher checkpoint for evaluation."
            )
        try:
            model = torch.hub.load(str(repo_dir), model_name, source="local", pretrained=False)
        except Exception as exc:
            raise RuntimeError(f"Cannot construct required DINOv3 ViT-L/16 from {repo_dir}") from exc
        if (type(model).__name__ != "DinoVisionTransformer" or
            getattr(model, "embed_dim", None) != 1024 or len(getattr(model, "blocks", [])) != 24 or
            tuple(model.patch_embed.proj.kernel_size) != (16, 16) or
            not callable(getattr(model, "get_intermediate_layers", None))):
            raise TypeError("Backbone must be DINOv3 ViT-L/16 (24 blocks, 1024 channels, patch size 16).")
        if checkpoint_path is None:
            from CESDNet.utils.checkpoint import load_checkpoint_state, strip_module_prefix
            state = strip_module_prefix(load_checkpoint_state(pretrained_path, preferred_keys=("teacher", "model")))
            # strict=True rejects corrupt, incomplete and alternative-backbone states.
            try:
                model.load_state_dict(state, strict=True)
            except Exception as exc:
                raise RuntimeError(f"DINOv3 pretrained weights do not match ViT-L/16: {pretrained_path}") from exc
            print(f"Strictly loaded DINOv3 ViT-L/16 pretrained weights: {pretrained_path}")
        model.patch_embed = DualPathPatchEmbed(model.patch_embed, embed_dim=1024, ndsm_in_chans=self.sar_in_chans)
        peft_config = LoraConfig(
            r=lora_r, lora_alpha=lora_r * 2,
            # Restrict LoRA to ViT blocks; suffix "proj" also matches the optical patch conv.
            target_modules=r"blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(fc1|fc2))",
            lora_dropout=0.1, bias="lora_only", modules_to_save=["patch_embed"],
        )
        model = get_peft_model(model, peft_config)
        if self.architecture_version == "paper_v1":
            # The original ViT, optical embedding and legacy LoRA matrices stay frozen.
            # Only the additional nDSM embedding belongs to the trainable adaptation path.
            for name, parameter in model.named_parameters():
                parameter.requires_grad = ("patch_embed.modules_to_save.default.ndsm_embed." in name)
        return model

    def train(self, mode: bool = True):
        super().train(mode)
        if self.architecture_version == "paper_v1":
            # Frozen DINOv3 must not activate stochastic positional augmentation/dropout.
            self.backbone.eval()
        return self

    def forward(
        self,
        rgb: torch.Tensor,
        ndsm: torch.Tensor,
        gt_mask: Optional[torch.Tensor] = None,
        return_feats: bool = False,
        return_kd: bool = False,
        epoch: int = 1,
    ):
        if not self._checkpoint_verified:
            raise RuntimeError("Complete teacher checkpoint has not been strictly loaded. Use build_cesdnet_t(..., checkpoint_path=...).")
        if rgb.ndim != 4 or rgb.shape[1] != 3 or ndsm.ndim != 4 or ndsm.shape[1] != self.sar_in_chans:
            raise ValueError("Teacher inputs must be aligned [B,3,H,W] IRRG and [B,1,H,W] nDSM.")
        if rgb.shape[0] != ndsm.shape[0] or rgb.shape[-2:] != ndsm.shape[-2:]:
            raise ValueError("IRRG and nDSM batch and spatial dimensions must match.")
        if rgb.shape[-2] % 16 or rgb.shape[-1] % 16:
            raise ValueError("Teacher spatial input dimensions must be divisible by 16.")
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
            raise TypeError("DINOv3 must expose get_intermediate_layers; feature substitution is prohibited.")

        if len(features_seq) != 4:
            raise RuntimeError(f"DINOv3 returned {len(features_seq)} stages; expected four at blocks 6,12,18,24.")
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

        if self.architecture_version == "paper_v1":
            self.ortho_loss = self.ortho_loss / len(self.adapters)
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


LFP_TextureExpert = TPE
OrthogonalDualExpertAdapter = GeoMoE


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
        architecture_version: str = "paper_v1",
        checkpoint_path: str | Path | None = None,
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
            architecture_version=architecture_version,
            checkpoint_path=checkpoint_path,
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
    checkpoint_path: str | Path | None = None,
) -> CESDNetT:
    architecture_version = getattr(config, "architecture_version", "paper_v1")
    if checkpoint_path is not None:
        from CESDNet.utils.checkpoint import read_checkpoint
        payload = read_checkpoint(checkpoint_path)
        recorded_config = payload.get("config", {}) if isinstance(payload, Mapping) else {}
        recorded_version = recorded_config.get("architecture_version") if isinstance(recorded_config, Mapping) else None
        if recorded_version is None:
            if architecture_version != "legacy_v0":
                raise ValueError(
                    "Teacher checkpoint has no architecture metadata. Historical raw weights require "
                    "explicit --architecture-version legacy_v0; they cannot certify paper_v1 behavior."
                )
        elif recorded_version != architecture_version:
            raise ValueError(f"Teacher architecture mismatch: checkpoint={recorded_version}, requested={architecture_version}")
        del payload
    model = CESDNetT(
        repo_dir=config.teacher_repo,
        weights_path=config.teacher_backbone,
        num_classes=config.num_classes,
        use_gmr_prior=use_gmr_prior,
        texture_mode=texture_mode,
        head_profile=head_profile,
        architecture_version=getattr(config, "architecture_version", "paper_v1"),
        checkpoint_path=checkpoint_path,
    )
    if checkpoint_path is not None:
        from CESDNet.utils.checkpoint import load_state_dict_flexible
        model.checkpoint_load_result = load_state_dict_flexible(
            model, checkpoint_path,
            preferred_keys=("teacher_state_dict", "state_dict", "model", "net"), strict=True,
        )
        model._checkpoint_verified = True
        print(f"Strictly loaded complete CESDNet-T checkpoint: {checkpoint_path}")
    return model
