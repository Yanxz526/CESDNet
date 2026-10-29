"""Aligned optical/nDSM/label patches; preserve IRRG channels and [0, 1] scaling."""
from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


PALETTE = ((255, 255, 255), (0, 0, 255), (0, 255, 255),
           (0, 255, 0), (255, 255, 0), (255, 0, 0))


def decode_label(array: np.ndarray) -> np.ndarray:
    """Black/unknown boundary colours become ignore=255, never class zero."""
    if array.ndim == 3:
        if array.shape[2] < 3:
            raise ValueError(f"RGB label needs three channels: {array.shape}")
        label = np.full(array.shape[:2], 255, dtype=np.uint8)
        for index, colour in enumerate(PALETTE):
            label[np.all(array[..., :3] == colour, axis=2)] = index
        return label
    if array.ndim != 2:
        raise ValueError(f"Label must be HxW or HxWx3, got {array.shape}")
    if not np.isin(array, [0, 1, 2, 3, 4, 5, 255]).all():
        raise ValueError(f"Unexpected label IDs: {np.unique(array).tolist()}; expected 0..5 or 255")
    return array.astype(np.uint8)


class ISPRSPatches(Dataset):
    """CSV manifests contain image,nDSM,label paths relative to ``root``.

    Validation always uses its explicit manifest. A historical ``test.txt``
    containing validation patches is supported only by configuring
    ``split_manifests.val`` explicitly; it is never selected automatically.
    """

    def __init__(self, cfg, mode="train"):
        if mode not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {mode}")
        self.mode = mode
        self.root = Path(cfg["root"]).expanduser().resolve()
        self.n_classes = int(cfg.get("n_classes", 6))
        self.require_ndsm = bool(cfg.get("require_ndsm", True))
        if self.n_classes != 6:
            raise ValueError("ISPRS requires six classes (including clutter)")
        crop_size = cfg.get("crop_size", [256, 256])
        self.crop_size = tuple(map(int, crop_size.split() if isinstance(crop_size, str) else crop_size))
        self.flip_p = float(cfg.get("p", 0.5))
        if not 0 <= self.flip_p <= 1 or len(self.crop_size) != 2 or min(self.crop_size) <= 0:
            raise ValueError("Invalid crop_size or flip probability")
        manifests = cfg.get("split_manifests", {})
        manifest = self.root / manifests.get(mode, f"{mode}.txt")
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing explicit {mode} manifest: {manifest}. Run scripts/prepare_isprs.py; do not substitute test data for validation.")
        with manifest.open(newline="", encoding="utf-8") as stream:
            self.rows = [tuple(row) for row in csv.reader(stream) if row and any(x.strip() for x in row)]
        if not self.rows:
            raise ValueError(f"Empty {mode} manifest: {manifest}")
        for line, row in enumerate(self.rows, 1):
            if len(row) == 2 and not self.require_ndsm:
                row = (row[0], "", row[1])
                self.rows[line - 1] = row
            if len(row) != 3:
                raise ValueError(f"{manifest}:{line}: expected image,nDSM,label")
            names = row if self.require_ndsm else (row[0], row[2])
            for name in names:
                if not (self.root / name.strip()).is_file():
                    raise FileNotFoundError(f"{manifest}:{line}: missing {self.root / name.strip()}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image_path, ndsm_path, label_path = [self.root / p.strip() for p in self.rows[index]]
        with Image.open(image_path) as source:
            optical = np.asarray(source).copy()
        ndsm = None
        if self.require_ndsm:
            with Image.open(ndsm_path) as source:
                ndsm = np.asarray(source).copy()
        with Image.open(label_path) as source:
            label = decode_label(np.asarray(source))
        if optical.ndim != 3 or optical.shape[2] != 3 or optical.dtype != np.uint8:
            raise ValueError(f"Expected uint8 IRRG HxWx3 image: {image_path}, {optical.shape}/{optical.dtype}")
        if ndsm is not None and (ndsm.ndim != 2 or ndsm.dtype != np.uint8):
            raise ValueError(f"Expected explicitly normalized uint8 HxW nDSM: {ndsm_path}; raw heights must be normalized during preparation")
        if optical.shape[:2] != label.shape or (ndsm is not None and ndsm.shape != label.shape):
            raise ValueError(f"Modalities/label are not spatially aligned: {image_path}")
        if self.mode == "train":
            h, w = label.shape
            ch, cw = self.crop_size
            if h < ch or w < cw:
                raise ValueError(f"Patch {label.shape} smaller than crop {self.crop_size}")
            top, left = random.randrange(h - ch + 1), random.randrange(w - cw + 1)
            optical, label = [a[top:top+ch, left:left+cw] for a in (optical, label)]
            if ndsm is not None:
                ndsm = ndsm[top:top+ch, left:left+cw]
            # A single set of draws is shared by both modalities and the label.
            turns = random.randrange(4)
            optical, label = [np.rot90(a, turns) for a in (optical, label)]
            if ndsm is not None:
                ndsm = np.rot90(ndsm, turns)
            for axis in (0, 1):
                if random.random() < self.flip_p:
                    optical, label = [np.flip(a, axis=axis) for a in (optical, label)]
                    if ndsm is not None:
                        ndsm = np.flip(ndsm, axis=axis)
        sample = {
            "image": torch.from_numpy(np.ascontiguousarray(optical.transpose(2, 0, 1))).float().div_(255),
            "label": torch.from_numpy(np.ascontiguousarray(label)).long(),
            "label_path": label_path.name,
        }
        if ndsm is not None:
            sample["dsm"] = torch.from_numpy(np.ascontiguousarray(ndsm[None])).float().div_(255)
        return sample
