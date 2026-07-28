from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]

PAPER_TEACHER_CHECKPOINTS = {
    "Potsdam": "",
    "Vaihingen": "",
}

PAPER_KD_CHECKPOINTS = {
    "Potsdam": "",
    "Vaihingen": "",
}

BAIDU_NETDISK_URL = ""
BAIDU_NETDISK_EXTRACT_CODE = ""


@dataclass
class CESDNetConfig:
    dataset: str = "Potsdam"
    num_classes: int = 6
    ignore_index: int = 255
    epochs: int = 100
    teacher_batch_size: int = 8
    student_batch_size: int = 16
    kd_batch_size: int = 16
    num_workers: int = 4
    lr: float = 1e-4
    weight_decay: float = 5e-4
    seed: int = 3407
    kd_ramp_epochs: int = 20
    lambda_kd: float = 0.60
    lambda_sh: float = 0.45
    lambda_pr: float = 0.40
    lambda_cr: float = 0.15
    lambda_tex: float = 1.00
    lambda_geo: float = 1.25
    lambda_conf: float = 2.00
    lambda_rel: float = 2.00
    eta_r: float = 0.25
    tau: float = 4.00
    kd_stages: Tuple[str, str] = ("c3", "c4")
    kd_stage_weights: Tuple[float, float] = (0.4, 0.6)
    teacher_ortho_weight: float = 0.5
    dformer_weight: str | Path | None = ""
    teacher_repo: Path = REPO_ROOT / "toolbox/models/dinov3"
    teacher_backbone: str | Path | None = ""

    @property
    def config_path(self) -> Path:
        return REPO_ROOT / "configs" / f"{self.dataset}.json"

    @property
    def dataset_root(self) -> Path:
        return REPO_ROOT / "database" / self.dataset

    @property
    def eval_class_indices(self) -> Tuple[int, ...]:
        return tuple(range(5)) if self.dataset == "Vaihingen" else tuple(range(self.num_classes))

    @property
    def eval_ignore_labels(self) -> Tuple[int, ...]:
        return (5,) if self.dataset == "Vaihingen" else ()

    @property
    def paper_teacher_checkpoint(self) -> str:
        return PAPER_TEACHER_CHECKPOINTS[self.dataset]

    @property
    def paper_kd_checkpoint(self) -> str:
        return PAPER_KD_CHECKPOINTS[self.dataset]
