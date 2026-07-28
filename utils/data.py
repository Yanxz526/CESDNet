from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from CESDNet.config import CESDNetConfig
from toolbox.datasets.potsdam import Potsdam
from toolbox.datasets.vaihingen import Vaihingen


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)


def load_dataset_cfg(config: CESDNetConfig) -> Dict[str, object]:
    with config.config_path.open("r", encoding="utf-8") as fp:
        cfg = json.load(fp)
    cfg["root"] = str(config.dataset_root)
    cfg["n_classes"] = config.num_classes
    return cfg


def build_dataset(config: CESDNetConfig, mode: str):
    cfg = load_dataset_cfg(config)
    if config.dataset == "Potsdam":
        return Potsdam(cfg, mode=mode)
    if config.dataset == "Vaihingen":
        return Vaihingen(cfg, mode=mode)
    raise ValueError(f"Unsupported dataset: {config.dataset}")


def build_dataloaders(
    config: CESDNetConfig,
    batch_size: int,
    drop_last_train: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    train_loader = DataLoader(
        build_dataset(config, "train"),
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=drop_last_train,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=config.num_workers > 0,
    )
    val_loader = DataLoader(
        build_dataset(config, "test"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        persistent_workers=config.num_workers > 0,
    )
    return train_loader, val_loader


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
