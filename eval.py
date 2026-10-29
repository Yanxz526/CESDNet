from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from PIL import Image

from CESDNet.config import CESDNetConfig, REPO_ROOT
from CESDNet.utils.runtime import require_checkpoint, resolve_device
from CESDNet.utils.runtime import add_runtime_args, apply_runtime_args
from CESDNet.utils.data import ensure_dir
from CESDNet.utils.metrics import (
    compute_metrics,
    frequency_weighted_iou,
    mean_f1,
    metric_line,
    update_confusion,
)
from CESDNet.utils.palette import color_to_label


CLASS_NAMES = ("Imp.surf", "Buildings", "Low veg.", "Tree", "Car", "Clutter")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CESDNet weights or saved prediction images.")
    parser.add_argument("--dataset", default="Potsdam", choices=["Potsdam", "Vaihingen"])
    parser.add_argument("--pred-dir", default="")
    parser.add_argument("--checkpoint", "--weights", dest="checkpoint", default="")
    parser.add_argument("--model", default="auto", choices=["auto", "teacher", "student", "kd"])
    parser.add_argument("--label-dir", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--tile-ids", nargs="*", default=None)
    parser.add_argument("--teacher-repo", default="")
    parser.add_argument("--teacher-backbone", default="")
    parser.add_argument("--metric-mode", default="global", choices=["paper", "global", "both", "legacy-per-tile"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--optical-product', choices=['IRRG', 'RGB'], default='IRRG', help='RGB is only for explicit legacy diagnostics.')
    add_runtime_args(parser)
    return parser.parse_args()


def canonical_name(path: Path) -> str:
    stem = path.stem.lower()
    for suffix in ("_label_noboundary", "_noboundary", "_label", "_irrg", "_rgb"):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
    if stem.startswith("top_mosaic_09cm_area"):
        tile_id = stem.replace("top_mosaic_09cm_area", "", 1)
        if "_" in tile_id:
            stem = f"top_potsdam_{tile_id}"
    return stem


def read_label_image(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = color_to_label(arr[:, :, :3])
    return arr.astype(np.int64)


def collect_images(path: Path) -> dict[str, Path]:
    if not path.is_dir():
        raise FileNotFoundError(f"Image directory not found: {path}")
    exts = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}
    images = {}
    for path in sorted(path.iterdir()):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        key = canonical_name(path)
        if key in images:
            raise ValueError(f"Duplicate tile identity: {images[key]} and {path}")
        images[key] = path
    return images


def default_label_dir(dataset: str, isprs_root: str | Path | None = None) -> Path:
    root = Path(isprs_root) if isprs_root else REPO_ROOT / "ISPRS"
    if dataset == "Potsdam":
        return root / dataset / "5_Labels_all"
    if dataset == "Vaihingen":
        return root / dataset / "gts_for_participants"
    raise ValueError(f"Unsupported ISPRS dataset: {dataset}")


def selected_tile_ids(config: CESDNetConfig, args: argparse.Namespace) -> tuple[list[str], bool]:
    with config.config_path.open(encoding="utf-8") as stream:
        specification = json.load(stream)
    if specification["dataset"] != config.dataset:
        raise ValueError("Dataset and split configuration disagree")
    split_ids = specification["tile_ids"][config.split]
    requested = args.tile_ids if args.tile_ids is not None else split_ids
    if not requested:
        raise ValueError("Evaluation tile selection cannot be empty")
    if len(requested) != len(set(requested)):
        raise ValueError("Evaluation tile selection contains duplicate IDs")
    outside = set(requested) - set(split_ids)
    if outside:
        raise ValueError(f"Tile IDs outside the declared {config.split} split: {sorted(outside)}")
    return list(requested), set(requested) != set(split_ids)


def validate_prediction_selection(
    predictions: dict[str, Path], config: CESDNetConfig, tile_ids: list[str],
) -> None:
    prefix = "top_potsdam_" if config.dataset == "Potsdam" else "top_mosaic_09cm_area"
    expected = {prefix + tile_id for tile_id in tile_ids}
    missing = expected - set(predictions)
    unexpected = set(predictions) - expected
    if missing or unexpected:
        raise ValueError(
            f"Prediction coverage does not match the declared {config.split} selection: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}. "
            "For a diagnostic subset, pass --tile-ids explicitly and use a matching prediction directory."
        )


def infer_predictions_from_weights(
    args: argparse.Namespace,
    config: CESDNetConfig,
    pred_dir: Path | None,
) -> Path:
    from CESDNet.test import (
        infer_model_name,
        load_student_model,
        load_teacher_model,
        model_tag,
        test_tiles,
    )
    from CESDNet.utils.data import seed_everything

    seed_everything(config.seed, deterministic=getattr(config, "deterministic", False))
    device = resolve_device(args)
    model_name = args.model
    checkpoint = args.checkpoint
    if model_name == "auto":
        model_name = infer_model_name(checkpoint, config)
        if model_name == "teacher" and not checkpoint:
            checkpoint = config.paper_teacher_checkpoint
        print(f"Auto selected {model_tag(model_name)} from weights: {checkpoint}")

    if pred_dir is None:
        pred_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "results" / config.dataset / f"{model_tag(model_name)}-eval"

    if model_name == "teacher":
        model = load_teacher_model(config, checkpoint, device)
    else:
        model = load_student_model(config, checkpoint, device, model_name)

    tile_args = argparse.Namespace(
        model=model_name,
        checkpoint=checkpoint,
        output_dir=str(pred_dir),
        tile_ids=args.tile_ids,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        amp=args.amp,
        metric_mode=args.metric_mode,
        optical_product=args.optical_product,
    )
    test_tiles(model, config, tile_args, device)
    return pred_dir


def main() -> None:
    args = parse_args()
    config = CESDNetConfig(
        dataset=args.dataset,
        num_classes=args.num_classes,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    config = apply_runtime_args(config, args)
    if args.teacher_repo:
        config.teacher_repo = Path(args.teacher_repo)
    if args.teacher_backbone:
        config.teacher_backbone = Path(args.teacher_backbone)
    tile_ids, partial = selected_tile_ids(config, args)

    pred_dir = Path(args.pred_dir) if args.pred_dir else None
    if args.checkpoint or pred_dir is None:
        if not args.checkpoint:
            if args.model == "kd":
                args.checkpoint = config.paper_kd_checkpoint
            elif args.model in ("auto", "teacher"):
                args.checkpoint = config.paper_teacher_checkpoint
        require_checkpoint(args.checkpoint,
                           "Provide --weights with a complete compatible model checkpoint, or --pred-dir to evaluate saved predictions without weights.")
        pred_dir = infer_predictions_from_weights(args, config, pred_dir)
    if pred_dir is None:
        raise ValueError("Pass --pred-dir to evaluate saved predictions or --weights/--checkpoint to run inference first.")

    label_dir = Path(args.label_dir) if args.label_dir else default_label_dir(config.dataset, config.isprs_root)
    output_dir = ensure_dir(Path(args.output_dir)) if args.output_dir else pred_dir

    pred_images = collect_images(pred_dir)
    label_images = collect_images(label_dir)
    if not pred_images:
        raise FileNotFoundError(f"No prediction images found under: {pred_dir}")
    if not label_images:
        raise FileNotFoundError(f"No label images found under: {label_dir}")
    validate_prediction_selection(pred_images, config, tile_ids)
    missing_labels = set(pred_images) - set(label_images)
    if missing_labels:
        raise FileNotFoundError(f"Missing {config.split} labels for tiles: {sorted(missing_labels)}")

    confusion = np.zeros((config.num_classes, config.num_classes), dtype=np.int64)
    per_image = []
    matched = []
    for key, pred_path in pred_images.items():
        label_path = label_images.get(key)
        if label_path is None:
            raise FileNotFoundError(f"No label matches prediction: {pred_path}")
        pred = torch.from_numpy(read_label_image(pred_path)).unsqueeze(0)
        label = torch.from_numpy(read_label_image(label_path)).unsqueeze(0)
        if pred.shape != label.shape:
            raise ValueError(f"Shape mismatch for {pred_path.name}: pred={tuple(pred.shape)} label={tuple(label.shape)}")
        image_confusion = np.zeros((config.num_classes, config.num_classes), dtype=np.int64)
        image_confusion = update_confusion(
            image_confusion,
            pred,
            label,
            config.num_classes,
            config.ignore_index,
            ignore_labels=config.eval_ignore_labels,
        )
        confusion += image_confusion
        image_metrics = compute_metrics(image_confusion, config.eval_class_indices)
        per_image.append({
            "acc": float(image_metrics["oa"]),
            "macc": float(image_metrics["macc"]),
            "miou": float(image_metrics["miou"]),
            "fwiou": frequency_weighted_iou(image_confusion),
            "f1": mean_f1(image_confusion, config.eval_class_indices),
            "class_acc": np.asarray(image_metrics["acc"]).tolist(),
            "class_iou": np.asarray(image_metrics["iou"]).tolist(),
        })
        matched.append((pred_path.name, label_path.name))

    if not matched:
        raise ValueError(f"No matching prediction/label names between {pred_dir} and {label_dir}")

    metrics = compute_metrics(confusion, config.eval_class_indices)
    paper_metrics = {
        "acc": float(np.mean([item["acc"] for item in per_image])),
        "macc": float(np.mean([item["macc"] for item in per_image])),
        "miou": float(np.mean([item["miou"] for item in per_image])),
        "fwiou": float(np.mean([item["fwiou"] for item in per_image])),
        "f1": float(np.mean([item["f1"] for item in per_image])),
    }
    print(f"Matched images: {len(matched)}")
    if partial:
        print(f"Partial {config.split} evaluation: {tile_ids}; this is not a complete split score.")
    if args.metric_mode in ("legacy-per-tile", "both"):
        print(
            "Legacy per-tile mean (not manuscript protocol) | "
            f"Acc:{paper_metrics['acc'] * 100:.3f} "
            f"F1:{paper_metrics['f1'] * 100:.3f} "
            f"mAcc:{paper_metrics['macc'] * 100:.3f} "
            f"mIoU:{paper_metrics['miou'] * 100:.3f} "
            f"FWIoU:{paper_metrics['fwiou'] * 100:.3f}"
        )
    if args.metric_mode in ("paper", "global", "both"):
        print(f"Global eval | {metric_line(metrics)}")

    acc = np.asarray(metrics["acc"])
    iou = np.asarray(metrics["iou"])
    for idx, name in enumerate(CLASS_NAMES[: config.num_classes]):
        print(f"{idx}:{name} Acc={acc[idx] * 100:.3f} IoU={iou[idx] * 100:.3f}")

    report = {
        "dataset": config.dataset,
        "pred_dir": str(pred_dir),
        "label_dir": str(label_dir),
        "matched": matched,
        "legacy_per_tile": {key: value * 100 for key, value in paper_metrics.items()},
        "split": config.split,
        "tile_ids": tile_ids,
        "partial": partial,
        "metric_protocol": "global confusion",
        "global": {
            "macc": float(metrics["macc"]) * 100,
            "miou": float(metrics["miou"]) * 100,
            "oa": float(metrics["oa"]) * 100,
            "kappa": float(metrics["kappa"]) * 100,
        },
        "acc": acc.tolist(),
        "iou": iou.tolist(),
        "confusion": confusion.tolist(),
    }
    with (output_dir / "eval_metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    np.save(output_dir / "confusion.npy", confusion)


if __name__ == "__main__":
    main()
