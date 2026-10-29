from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from CESDNet.config import CESDNetConfig


TensorDict = Mapping[str, torch.Tensor]
KDTree = Mapping[str, Any]


@dataclass
class KDTerms:
    L_SCD: torch.Tensor
    L_CPD: torch.Tensor
    L_RCRD: torch.Tensor
    L_CESD_KD: torch.Tensor
    L_total: Optional[torch.Tensor] = None

    def meters(self) -> Dict[str, float]:
        return {
            "L_SCD": float(self.L_SCD.detach().item()),
            "L_CPD": float(self.L_CPD.detach().item()),
            "L_RCRD": float(self.L_RCRD.detach().item()),
            "L_CESD_KD": float(self.L_CESD_KD.detach().item()),
            "L_total": float(self.L_total.detach().item()) if self.L_total is not None else 0.0,
        }


class StudentKDHead(nn.Module):
    def __init__(self, in_channels: int, kd_channels: int = 256) -> None:
        super().__init__()
        self.Q_sh = nn.Sequential(
            nn.Conv2d(in_channels, kd_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(kd_channels),
            nn.GELU(),
        )
        self.Q_tex = nn.Sequential(
            nn.Conv2d(in_channels, kd_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(kd_channels),
            nn.GELU(),
        )
        self.Q_geo = nn.Sequential(
            nn.Conv2d(in_channels, kd_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(kd_channels),
            nn.GELU(),
        )
        self.q = nn.Conv2d(kd_channels, 1, kernel_size=1)

    def forward(self, F_s_i: torch.Tensor) -> Dict[str, torch.Tensor]:
        Q_sh_i = self.Q_sh(F_s_i)
        Q_tex_i = self.Q_tex(F_s_i)
        Q_geo_i = self.Q_geo(F_s_i)
        q_i = self.q(Q_sh_i)
        return {
            "Q_sh": Q_sh_i,
            "Q_tex": Q_tex_i,
            "Q_geo": Q_geo_i,
            "q": q_i,
            "shared": Q_sh_i,
            "tex_private": Q_tex_i,
            "geo_private": Q_geo_i,
            "response": q_i,
        }


class MultiStageStudentKDHead(nn.Module):
    def __init__(
        self,
        in_channels: Mapping[str, int] | None = None,
        kd_channels: int = 256,
        stages: Sequence[str] = ("c3", "c4"),
    ) -> None:
        super().__init__()
        in_channels = dict(in_channels or {"c3": 128, "c4": 256})
        self.stages = tuple(stages)
        self.heads = nn.ModuleDict(
            {stage: StudentKDHead(in_channels[stage], kd_channels=kd_channels) for stage in self.stages}
        )

    def forward(self, F_s: Mapping[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        return {stage: self.heads[stage](F_s[stage]) for stage in self.stages}


def resize_like(x: torch.Tensor, ref: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    kwargs = {"align_corners": False} if mode == "bilinear" else {}
    return F.interpolate(x, size=ref.shape[-2:], mode=mode, **kwargs)


def get_stage_kd(kd_tree: KDTree, stage: str) -> TensorDict:
    direct = kd_tree.get(stage)
    if isinstance(direct, Mapping):
        return direct
    stages = kd_tree.get("stages")
    if isinstance(stages, Mapping) and isinstance(stages.get(stage), Mapping):
        return stages[stage]
    raise KeyError(f"Missing teacher KD stage '{stage}'. Available keys: {list(kd_tree.keys())}")


def get_required(stage_kd: TensorDict, key: str, stage: str) -> torch.Tensor:
    if key not in stage_kd:
        raise KeyError(f"Stage {stage} is missing '{key}'. Available: {list(stage_kd.keys())}")
    return stage_kd[key]


def normalized_entropy_confidence(
    O_t: torch.Tensor,
    num_classes: int,
    lambda_conf: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    P_t = F.softmax(O_t.float(), dim=1)
    H_t = -torch.sum(P_t * torch.log(P_t + eps), dim=1, keepdim=True) / math.log(num_classes)
    M_conf = torch.exp(-lambda_conf * H_t)
    return torch.clamp(M_conf, 0.0, 1.0).detach()


def _teacher_key(stage_kd: TensorDict, canonical: str, alias: str) -> torch.Tensor:
    return stage_kd[canonical] if canonical in stage_kd else stage_kd[alias]


def integrate_shared_knowledge(
    K_sh_i: torch.Tensor,
    r_sh_i: torch.Tensor,
    eta_r: float = 0.25,
) -> torch.Tensor:
    r_sh_i = resize_like(r_sh_i, K_sh_i)
    return torch.cat([K_sh_i, eta_r * r_sh_i], dim=1)


def compute_L_SCD(
    student_stage: TensorDict,
    teacher_stage: TensorDict,
    M_conf: torch.Tensor,
    stage: str,
    eta_r: float = 0.25,
) -> torch.Tensor:
    K_sh_i = _teacher_key(teacher_stage, "shared", "K_sh").detach()
    A_i = _teacher_key(teacher_stage, "agreement", "A").detach()
    r_sh_i = _teacher_key(teacher_stage, "response", "r_sh").detach()

    M_sh_i = torch.clamp(resize_like(M_conf, K_sh_i) * resize_like(A_i, K_sh_i), 0.0, 1.0).detach()
    Q_sh_i = resize_like(student_stage["Q_sh"], K_sh_i)
    q_i = resize_like(student_stage["q"], r_sh_i)

    # Equations (24)-(26): mean-reduced feature and response losses are
    # separate. Concatenating responses changes their loss and weighting.
    M_res_i = resize_like(M_sh_i, r_sh_i)
    feature_loss = F.smooth_l1_loss(M_sh_i * Q_sh_i, M_sh_i * K_sh_i)
    response_loss = F.mse_loss(M_res_i * torch.sigmoid(q_i), M_res_i * torch.sigmoid(r_sh_i))
    return feature_loss + eta_r * response_loss


def fourier_amplitude(F_i: torch.Tensor) -> torch.Tensor:
    return torch.abs(torch.fft.fft2(F_i.float(), norm="ortho"))


def topology_relation(F_i: torch.Tensor, pool_size: int = 16) -> torch.Tensor:
    b, c, _, _ = F_i.shape
    pooled = F.adaptive_avg_pool2d(F_i.float(), (pool_size, pool_size)).view(b, c, -1)
    pooled = F.normalize(pooled, p=2, dim=1, eps=1e-6)
    return torch.bmm(pooled.transpose(1, 2), pooled)


def compute_L_CPD(
    student_stage: TensorDict,
    teacher_stage: TensorDict,
    M_conf: torch.Tensor,
    stage: str,
    lambda_tex: float = 1.0,
    lambda_geo: float = 1.25,
) -> torch.Tensor:
    K_tex_i = _teacher_key(teacher_stage, "tex_private", "K_tex").detach()
    K_geo_i = _teacher_key(teacher_stage, "geo_private", "K_geo").detach()
    A_i = _teacher_key(teacher_stage, "agreement", "A").detach()
    C_i = teacher_stage.get("complementarity", teacher_stage.get("C", 1.0 - A_i)).detach()
    S_tex_i = get_required(teacher_stage, "tex_strength", stage).detach()
    S_geo_i = get_required(teacher_stage, "geo_strength", stage).detach()

    M_tex_i = torch.clamp(
        resize_like(M_conf, K_tex_i) * resize_like(C_i, K_tex_i) * resize_like(S_tex_i, K_tex_i),
        0.0,
        1.0,
    ).detach()
    M_geo_i = torch.clamp(
        resize_like(M_conf, K_geo_i) * resize_like(C_i, K_geo_i) * resize_like(S_geo_i, K_geo_i),
        0.0,
        1.0,
    ).detach()

    Q_tex_i = resize_like(student_stage["Q_tex"], K_tex_i)
    Q_geo_i = resize_like(student_stage["Q_geo"], K_geo_i)

    L_tex_i = F.l1_loss(fourier_amplitude(M_tex_i * Q_tex_i), fourier_amplitude(M_tex_i * K_tex_i))
    R_s_i = topology_relation(M_geo_i * Q_geo_i)
    R_t_i = topology_relation(M_geo_i * K_geo_i)
    L_geo_i = F.mse_loss(R_s_i, R_t_i)
    return lambda_tex * L_tex_i + lambda_geo * L_geo_i


def compute_L_RCRD(
    O_s: torch.Tensor,
    O_t: torch.Tensor,
    tau: float = 4.0,
    lambda_rel: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    if O_t.shape[-2:] != O_s.shape[-2:]:
        O_t = F.interpolate(O_t, size=O_s.shape[-2:], mode="bilinear", align_corners=False)

    b, k, h, w = O_s.shape
    n = h * w
    P_s_tau = F.softmax(O_s.float() / tau, dim=1).view(b, k, n)
    P_t_tau = F.softmax(O_t.float() / tau, dim=1).detach().view(b, k, n)

    P_t_plain = F.softmax(O_t.float(), dim=1).detach()
    H_t = -torch.sum(P_t_plain * torch.log(P_t_plain + eps), dim=1, keepdim=True) / math.log(k)
    M_cr = torch.exp(-lambda_rel * H_t).view(b, 1, n).detach()

    denom = M_cr.sum(dim=2, keepdim=True) + eps
    C_s = torch.bmm(P_s_tau * M_cr, P_s_tau.transpose(1, 2)) / denom
    C_t = torch.bmm(P_t_tau * M_cr, P_t_tau.transpose(1, 2)) / denom
    C_s = C_s / (C_s.sum(dim=2, keepdim=True) + eps)
    C_t = C_t / (C_t.sum(dim=2, keepdim=True) + eps)
    return F.mse_loss(C_s, C_t) * (tau * tau)


class StructuredKDLoss(nn.Module):
    def __init__(self, config: CESDNetConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        O_s: torch.Tensor,
        O_t: torch.Tensor,
        student_kd: Mapping[str, TensorDict],
        teacher_kd: KDTree,
        L_seg: Optional[torch.Tensor] = None,
        epoch: int = 0,
    ) -> KDTerms:
        M_conf = normalized_entropy_confidence(
            O_t,
            num_classes=self.config.num_classes,
            lambda_conf=self.config.lambda_conf,
        )
        L_SCD = O_s.new_zeros(())
        L_CPD = O_s.new_zeros(())

        for stage, alpha_i in zip(self.config.kd_stages, self.config.kd_stage_weights):
            t_stage = get_stage_kd(teacher_kd, stage)
            s_stage = student_kd[stage]
            L_SCD = L_SCD + float(alpha_i) * compute_L_SCD(
                s_stage,
                t_stage,
                M_conf,
                stage,
                eta_r=self.config.eta_r,
            )
            L_CPD = L_CPD + float(alpha_i) * compute_L_CPD(
                s_stage,
                t_stage,
                M_conf,
                stage,
                lambda_tex=self.config.lambda_tex,
                lambda_geo=self.config.lambda_geo,
            )

        L_RCRD = compute_L_RCRD(
            O_s,
            O_t,
            tau=self.config.tau,
            lambda_rel=self.config.lambda_rel,
        )
        L_CESD_KD = self.config.lambda_kd * (
            self.config.lambda_sh * L_SCD
            + self.config.lambda_pr * L_CPD
            + self.config.lambda_cr * L_RCRD
        )

        L_total = None
        if L_seg is not None:
            eta_kd = min(1.0, float(epoch + 1) / float(max(1, self.config.kd_ramp_epochs)))
            L_total = L_seg + eta_kd * L_CESD_KD
        return KDTerms(L_SCD=L_SCD, L_CPD=L_CPD, L_RCRD=L_RCRD, L_CESD_KD=L_CESD_KD, L_total=L_total)
