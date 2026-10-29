"""Prepare disjoint ISPRS IRRG/nDSM patches with explicit split provenance.

Examples (source-root is the dataset folder, not its parent):
  python scripts/prepare_isprs.py --dataset Vaihingen --source-root /data/ISPRS/Vaihingen --output-root /data/patches/Vaihingen
  python scripts/prepare_isprs.py --dataset Potsdam --source-root /data/ISPRS/Potsdam --output-root /data/patches/Potsdam

The source nDSM must already use uint8 [0,255] normalization, as in the
original experiments. Raw height rasters require an explicit maximum height.
The manuscript does not specify the sampling stride; this release records
the selected stride (default 256) rather than inferring an unavailable value.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from toolbox.datasets.isprs import decode_label


DEFAULT_PATTERNS = {
    "Vaihingen": (
        "top/top_mosaic_09cm_area{tile_id}.tif",
        "ndsm/dsm_09cm_matching_area{tile_id}_normalized.jpg",
        "gts_for_participants/top_mosaic_09cm_area{tile_id}.tif",
    ),
    "Potsdam": (
        "3_Ortho_IRRG/top_potsdam_{tile_id}_IRRG.tif",
        "1_DSM_normalisation/dsm_potsdam_0{tile_id}_normalized_lastools.jpg",
        "5_Labels_all/top_potsdam_{tile_id}_label.tif",
    ),
}


def positions(length: int, patch_size: int, stride: int) -> list[int]:
    """Cover the complete tile; use an anchored final patch at the boundary."""
    if length < patch_size:
        raise ValueError(f"Tile dimension {length} smaller than patch_size={patch_size}")
    result = list(range(0, length - patch_size + 1, stride))
    if result[-1] != length - patch_size:
        result.append(length - patch_size)
    return result


def normalize_ndsm(array: np.ndarray, max_height: float | None) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"nDSM must be one aligned channel, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("nDSM contains NaN or infinite values")
    if max_height is None:
        if array.dtype != np.uint8:
            raise ValueError("Raw/non-uint8 nDSM requires --ndsm-max-height; implicit clipping is forbidden")
        return array
    if max_height <= 0:
        raise ValueError("--ndsm-max-height must be positive")
    return np.rint(np.clip(array.astype(np.float32), 0, max_height) / max_height * 255).astype(np.uint8)


def prepare(args) -> dict:
    with Path(args.config).open(encoding="utf-8") as stream:
        cfg = json.load(stream)
    if cfg["dataset"] != args.dataset:
        raise ValueError("Dataset/config mismatch")
    tile_ids = cfg["tile_ids"]
    all_ids = [tile for split in ("train", "val", "test") for tile in tile_ids[split]]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Source tile IDs overlap between train/val/test")
    if args.patch_size <= 0 or not 0 < args.stride <= args.patch_size:
        raise ValueError("Require patch_size > 0 and 0 < stride <= patch_size")
    source = Path(args.source_root).expanduser().resolve()
    output = Path(args.output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is nonempty; refusing to overwrite: {output}")
    patterns = (args.image_pattern, args.ndsm_pattern, args.label_pattern)
    plan = []
    # Complete missing-file/header preflight before writing output patches.
    for split in args.splits:
        for tile in tile_ids[split]:
            paths = tuple(source / pattern.format(tile_id=tile) for pattern in patterns)
            sizes = []
            for path in paths:
                if not path.is_file():
                    raise FileNotFoundError(f"Missing {split} tile {tile}: {path}")
                with Image.open(path) as image:
                    sizes.append(image.size)
            if len(set(sizes)) != 1:
                raise ValueError(f"Unaligned optical/nDSM/label for {tile}: {sizes}")
            positions(sizes[0][0], args.patch_size, args.stride)
            positions(sizes[0][1], args.patch_size, args.stride)
            plan.append((split, tile, paths))
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"dataset": args.dataset, "source_root": str(source),
                "optical_channel_order": ["NIR", "R", "G"], "optical_divisor": 255,
                "ndsm_max_height": args.ndsm_max_height, "ndsm_divisor": 255,
                "patch_size": args.patch_size, "stride": args.stride,
                "border_rule": "anchor final window; no padding", "ignore_index": 255,
                "tile_ids": tile_ids, "prepared_splits": args.splits,
                "patterns": dict(zip(("optical", "ndsm", "label"), patterns)), "counts": {}}
    for split in args.splits:
        rows = []
        for kind in ("image", "ndsm", "label"):
            (output / split / kind).mkdir(parents=True, exist_ok=True)
        for planned_split, tile, paths in plan:
            if planned_split != split:
                continue
            arrays = []
            for path in paths:
                with Image.open(path) as image:
                    arrays.append(np.asarray(image).copy())
            optical, ndsm, label = arrays
            if optical.ndim != 3 or optical.shape[2] != 3 or optical.dtype != np.uint8:
                raise ValueError(f"Optical must be uint8 IRRG HxWx3: {paths[0]}")
            ndsm = normalize_ndsm(ndsm, args.ndsm_max_height)
            label = decode_label(label)
            for top in positions(label.shape[0], args.patch_size, args.stride):
                for left in positions(label.shape[1], args.patch_size, args.stride):
                    filename = f"{tile}_y{top}_x{left}.png"
                    row = []
                    for kind, array in zip(("image", "ndsm", "label"), (optical, ndsm, label)):
                        relative = Path(split) / kind / filename
                        Image.fromarray(array[top:top+args.patch_size, left:left+args.patch_size]).save(output / relative)
                        row.append(relative.as_posix())
                    rows.append(row)
        with (output / f"{split}.txt").open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerows(rows)
        metadata["counts"][split] = len(rows)
        print(f"{split}: {len(rows)} paired patches", flush=True)
    (output / "preprocessing.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=tuple(DEFAULT_PATTERNS))
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val", "test"])
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--ndsm-max-height", type=float, default=None)
    parser.add_argument("--image-pattern", default=None)
    parser.add_argument("--ndsm-pattern", default=None)
    parser.add_argument("--label-pattern", default=None)
    args = parser.parse_args()
    if len(args.splits) != len(set(args.splits)):
        parser.error("--splits must not repeat a split")
    args.config = args.config or ROOT / "configs" / f"{args.dataset}.json"
    for key, default in zip(("image_pattern", "ndsm_pattern", "label_pattern"), DEFAULT_PATTERNS[args.dataset]):
        if getattr(args, key) is None:
            setattr(args, key, default)
    return args


if __name__ == "__main__":
    prepare(parse_args())
