from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from CESDNet.config import CESDNetConfig
from toolbox.datasets.potsdam import Potsdam
from toolbox.datasets.vaihingen import Vaihingen


def seed_everything(seed: int, deterministic: bool = False) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(deterministic)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def load_dataset_cfg(config: CESDNetConfig) -> Dict[str, object]:
    with config.config_path.open("r", encoding="utf-8") as fp:
        cfg = json.load(fp)
    cfg["root"] = str(config.dataset_root)
    cfg["n_classes"] = config.num_classes
    return cfg


def build_dataset(config: CESDNetConfig, mode: str, require_ndsm: bool = True):
    cfg = load_dataset_cfg(config)
    cfg["require_ndsm"] = require_ndsm
    if config.dataset == "Potsdam":
        return Potsdam(cfg, mode=mode)
    if config.dataset == "Vaihingen":
        return Vaihingen(cfg, mode=mode)
    raise ValueError(f"Unsupported dataset: {config.dataset}")


def build_dataloaders(
    config: CESDNetConfig,
    batch_size: int,
    drop_last_train: bool = True,
    require_ndsm: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    train_dataset = build_dataset(config, "train", require_ndsm=require_ndsm)
    val_dataset = build_dataset(config, "val", require_ndsm=require_ndsm)
    train_labels = {str((train_dataset.root / row[2].strip()).resolve()) for row in train_dataset.rows}
    val_labels = {str((val_dataset.root / row[2].strip()).resolve()) for row in val_dataset.rows}
    if train_labels & val_labels:
        raise ValueError("Train and validation manifests share label files")
    if drop_last_train and len(train_dataset) < batch_size:
        raise ValueError(f"Training set has {len(train_dataset)} samples, smaller than batch_size={batch_size}; drop_last would produce no optimizer updates")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=drop_last_train,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        persistent_workers=False,
    )
    return train_loader, val_loader


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
