from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator, Literal, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from PIL import Image
from torch.amp import autocast
from torch.utils.data import DataLoader

from CESDNet.config import CESDNetConfig, REPO_ROOT
from CESDNet.utils.runtime import require_checkpoint, resolve_device
from CESDNet.utils.runtime import add_runtime_args, apply_runtime_args, limited_batches
from CESDNet.models.student import CESDNetS, infer_student_architecture
from CESDNet.utils.checkpoint import load_state_dict_flexible
from CESDNet.utils.data import build_dataset, ensure_dir, seed_everything
from CESDNet.utils.metrics import (
    compute_metrics,
    frequency_weighted_iou,
    mean_f1,
    metric_line,
    update_confusion,
)
from CESDNet.utils.palette import color_to_label, label_to_color


POTSDAM_TEST_IDS = ("2_13", "2_14", "3_13", "3_14", "4_13", "4_14", "4_15", "5_13", "5_14", "5_15", "6_13", "6_14", "6_15", "7_13")
VAIHINGEN_TEST_IDS = ("2", "4", "6", "8", "10", "12", "14", "16", "20", "22", "24", "27", "29", "31", "33", "35", "38")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test CESDNet-T, CESDNet-S or CESDNet-KD.")
    parser.add_argument("--dataset", default="Potsdam", choices=["Potsdam", "Vaihingen"])
    parser.add_argument("--model", default="auto", choices=["auto", "teacher", "student", "kd"])
    parser.add_argument("--checkpoint", "--weights", dest="checkpoint", default="")
    parser.add_argument("--mode", default="patch", choices=["patch", "tile"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--tile-ids", nargs="*", default=None)
    parser.add_argument("--teacher-repo", default="")
    parser.add_argument("--teacher-backbone", default="")
    parser.add_argument("--metric-mode", default="global", choices=["paper", "global", "both", "legacy-per-tile"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument('--max-batches', type=int, default=0, help='Positive values are partial diagnostics, not final scores.')
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--optical-product', choices=['IRRG', 'RGB'], default='IRRG', help='RGB is only for explicit legacy diagnostics.')
    add_runtime_args(parser)
    return parser.parse_args()


ModelName = Literal["teacher", "student", "kd"]


def model_tag(model_name: str) -> str:
    return {"teacher": "CESDNet-T", "student": "CESDNet-S", "kd": "CESDNet-KD"}[model_name]


def infer_model_name(checkpoint: str, config: CESDNetConfig) -> ModelName:
    if not checkpoint:
        return "teacher"

    checkpoint_path = Path(checkpoint)
    checkpoint_text = str(checkpoint_path).lower()
    configured_teacher = config.paper_teacher_checkpoint
    if (
        "teacher" in checkpoint_text
        or "geomoe" in checkpoint_text
        or (
            configured_teacher
            and checkpoint_path.resolve() == Path(configured_teacher).expanduser().resolve()
        )
    ):
        return "teacher"
    if "student" in checkpoint_text:
        return "student"
    if "kd" in checkpoint_text or "distill" in checkpoint_text or "cesdnet-kd" in checkpoint_text:
        return "kd"

    checkpoint_obj = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if isinstance(checkpoint_obj, dict):
        if "teacher_state_dict" in checkpoint_obj:
            return "teacher"
        if "student_state_dict" in checkpoint_obj:
            return "kd" if "kd_heads_state_dict" in checkpoint_obj else "student"
        keys = list(checkpoint_obj.keys())[:80]
    else:
        keys = []
    key_text = " ".join(str(key).lower() for key in keys)
    if "geo_encoder" in key_text or "adapters" in key_text or "backbone.base_model" in key_text:
        return "teacher"
    if "dformer" in key_text or "decode_head" in key_text or "aux_head" in key_text:
        return "kd"
    raise ValueError(f"Cannot infer model type from checkpoint. Pass --model teacher|student|kd: {checkpoint}")


def load_student_model(
    config: CESDNetConfig,
    checkpoint: str,
    device: torch.device,
    model_name: ModelName,
) -> CESDNetS:
    if not checkpoint and model_name == "kd":
        checkpoint = config.paper_kd_checkpoint
    if not checkpoint:
        raise ValueError("--checkpoint/--weights is required when --model is student or kd.")
    student_arch = infer_student_architecture(checkpoint)
    model = CESDNetS(
        num_classes=config.num_classes,
        dformer_weight=None,
        use_pretrained=False,
        student_arch=student_arch,
    ).to(device)
    msg = load_state_dict_flexible(
        model,
        checkpoint,
        preferred_keys=("student_state_dict", "state_dict", "model", "net"),
        strict=True,
    )
    print(f"Loaded {model_tag(model_name)}: architecture={student_arch} missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    model.eval()
    return model


def load_teacher_model(config: CESDNetConfig, checkpoint: str, device: torch.device):
    from CESDNet.models.teacher import (
        build_cesdnet_t,
        infer_teacher_head_profile,
        infer_teacher_texture_mode,
        infer_teacher_use_gmr_prior,
    )

    checkpoint = checkpoint or config.paper_teacher_checkpoint
    if not checkpoint:
        raise ValueError("--checkpoint/--weights is required for CESDNet-T.")
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint}")
    use_gmr_prior = infer_teacher_use_gmr_prior(checkpoint)
    texture_mode = infer_teacher_texture_mode(checkpoint)
    head_profile = infer_teacher_head_profile(checkpoint)
    model = build_cesdnet_t(
        config,
        use_gmr_prior=use_gmr_prior,
        texture_mode=texture_mode,
        head_profile=head_profile,
        checkpoint_path=checkpoint,
    ).to(device)
    model.eval()
    return model


def forward_model(model, model_name: str, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
    if model_name == "teacher":
        if z is None:
            raise ValueError("CESDNet-T requires nDSM input z.")
        return model(x, z)
    return model(x)


def save_prediction(pred: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(label_to_color(pred.astype(np.uint8))).save(path)


def test_patches(
    model: CESDNetS,
    config: CESDNetConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    dataset = build_dataset(config, config.split, require_ndsm=args.model == "teacher")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=config.num_workers > 0,
    )
    output_dir = ensure_dir(Path(args.output_dir)) if args.output_dir else None
    if args.save_predictions:
        output_dir = ensure_dir(
            Path(args.output_dir)
            if args.output_dir
            else REPO_ROOT / "results" / config.dataset / f"{model_tag(args.model)}-patches"
        )
    elif args.output_dir:
        output_dir = ensure_dir(Path(args.output_dir))
    confusion = np.zeros((config.num_classes, config.num_classes), dtype=np.int64)

    with torch.no_grad():
        for sample in limited_batches(loader, args.max_batches):
            x = sample["image"].to(device, non_blocking=True).float()
            y = sample["label"].to(device, non_blocking=True).long()
            z = sample["dsm"].to(device, non_blocking=True).float() if args.model == "teacher" else None
            with autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                logits = forward_model(model, args.model, x, z)
            pred = logits.argmax(dim=1)
            confusion = update_confusion(
                confusion,
                pred,
                y,
                config.num_classes,
                config.ignore_index,
                ignore_labels=config.eval_ignore_labels,
            )

            if args.save_predictions and output_dir is not None:
                names = sample.get("label_path")
                if isinstance(names, str):
                    names = [names]
                for pred_i, name in zip(pred.detach().cpu().numpy(), names):
                    save_prediction(pred_i, output_dir / Path(str(name)).with_suffix(".png").name)

    metrics = compute_metrics(confusion, config.eval_class_indices)
    print(metric_line(metrics))
    if output_dir is not None:
        np.save(output_dir / "confusion.npy", confusion)
        report = {
            'dataset': config.dataset, 'split': config.split, 'model': args.model,
            'checkpoint': args.checkpoint, 'architecture_version': config.architecture_version,
            'metric_protocol': 'global confusion', 'partial': bool(args.max_batches),
            'evaluated_pixels': int(confusion.sum()),
            'macc': float(metrics['macc']) * 100, 'miou': float(metrics['miou']) * 100,
            'oa': float(metrics['oa']) * 100, 'kappa': float(metrics['kappa']) * 100,
            'class_iou': np.asarray(metrics['iou']).tolist(),
            'class_acc': np.asarray(metrics['acc']).tolist(), 'confusion': confusion.tolist(),
        }
        (output_dir / 'eval_metrics.json').write_text(json.dumps(report, indent=2))


def sliding_window(shape: tuple[int, int], window_size: int, stride: int) -> Iterator[tuple[int, int, int, int]]:
    h, w = shape
    if stride <= 0 or stride > window_size or window_size <= 0:
        raise ValueError("Require 0 < stride <= window_size for complete coverage")
    if h < window_size or w < window_size:
        raise ValueError(f"Image {shape} is smaller than window_size={window_size}")
    tops = list(range(0, h - window_size + 1, stride))
    lefts = list(range(0, w - window_size + 1, stride))
    if tops[-1] != h - window_size:
        tops.append(h - window_size)
    if lefts[-1] != w - window_size:
        lefts.append(w - window_size)
    for top in tops:
        for left in lefts:
            yield top, left, window_size, window_size


def batched(iterable: Iterable[tuple[int, int, int, int]], batch_size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def isprs_paths(dataset: str, tile_id: str, isprs_root=None, optical_product="IRRG") -> tuple[Path, Path, Path, str]:
    root = Path(isprs_root) if isprs_root else REPO_ROOT / "ISPRS"
    if dataset == "Potsdam":
        image_path = root / "Potsdam" / ("3_Ortho_IRRG" if optical_product == "IRRG" else "2_Ortho_RGB") / f"top_potsdam_{tile_id}_{optical_product}.tif"
        ndsm_path = root / "Potsdam" / "1_DSM_normalisation" / f"dsm_potsdam_0{tile_id}_normalized_lastools.jpg"
        label_name = f"top_potsdam_{tile_id}_label.tif"
    else:
        image_path = root / "Vaihingen" / "top" / f"top_mosaic_09cm_area{tile_id}.tif"
        ndsm_path = root / "Vaihingen" / "ndsm" / f"dsm_09cm_matching_area{tile_id}_normalized.jpg"
        label_name = f"top_mosaic_09cm_area{tile_id}.tif"
    if dataset == 'Potsdam':
        label_path = root / dataset / '5_Labels_all' / f'top_potsdam_{tile_id}_label.tif'
    else:
        label_path = root / dataset / 'gts_for_participants' / f'top_mosaic_09cm_area{tile_id}.tif'
    return image_path, ndsm_path, label_path, label_name


def read_rgb(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path))
    if image.dtype != np.uint8 or image.shape[2] != 3:
        raise ValueError(f'Expected uint8 three-channel optical image: {path}')
    return image.astype(np.float32) / 255.0


def read_label(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = color_to_label(arr[:, :, :3])
    return arr.astype(np.int64)


def read_ndsm(path: Path) -> np.ndarray:
    ndsm = np.asarray(Image.open(path))
    if ndsm.dtype != np.uint8 or ndsm.ndim != 2:
        raise ValueError(f'Expected normalized uint8 one-channel nDSM: {path}')
    return ndsm.astype(np.float32) / 255.0


def default_tile_ids(dataset: str) -> Sequence[str]:
    return POTSDAM_TEST_IDS if dataset == "Potsdam" else VAIHINGEN_TEST_IDS


def test_tiles(
    model: CESDNetS,
    config: CESDNetConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    output_dir = ensure_dir(
        Path(args.output_dir)
        if args.output_dir
        else REPO_ROOT / "results" / config.dataset / model_tag(args.model)
    )
    with config.config_path.open() as stream:
        split_ids = json.load(stream)['tile_ids'][config.split]
    tile_ids = args.tile_ids or split_ids
    if set(tile_ids) - set(split_ids):
        raise ValueError(f'Tile IDs are outside the declared {config.split} split: {tile_ids}')
    confusion = np.zeros((config.num_classes, config.num_classes), dtype=np.int64)
    per_image = []
    for tile_id in tile_ids:
        paths = isprs_paths(config.dataset, tile_id, config.isprs_root, args.optical_product)
        required = (paths[0], paths[2]) + ((paths[1],) if args.model == 'teacher' else ())
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(f'Missing {config.split} tile input/label: {path}')

    with torch.no_grad():
        for tile_id in tile_ids:
            image_path, ndsm_path, label_path, label_name = isprs_paths(config.dataset, tile_id, config.isprs_root, args.optical_product)
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing ISPRS image: {image_path}")
            if args.model == "teacher" and not ndsm_path.is_file():
                raise FileNotFoundError(f"Missing ISPRS nDSM: {ndsm_path}")
            image = read_rgb(image_path)
            ndsm = read_ndsm(ndsm_path) if args.model == "teacher" else None
            pred_sum = np.zeros((*image.shape[:2], config.num_classes), dtype=np.float32)

            coords = list(sliding_window(image.shape[:2], args.window_size, args.stride))
            for coord_batch in batched(coords, args.batch_size):
                patches = [
                    np.copy(image[top : top + h, left : left + w]).transpose(2, 0, 1)
                    for top, left, h, w in coord_batch
                ]
                x = torch.from_numpy(np.asarray(patches)).to(device).float()
                z = None
                if ndsm is not None:
                    ndsm_patches = [
                        np.copy(ndsm[top : top + h, left : left + w])[None, :, :]
                        for top, left, h, w in coord_batch
                    ]
                    z = torch.from_numpy(np.asarray(ndsm_patches)).to(device).float()
                with autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                    logits = forward_model(model, args.model, x, z)
                logits_np = logits.detach().cpu().numpy()
                for out, (top, left, h, w) in zip(logits_np, coord_batch):
                    pred_sum[top : top + h, left : left + w] += out.transpose(1, 2, 0)

            pred = pred_sum.argmax(axis=-1).astype(np.uint8)
            save_path = output_dir / Path(label_name).with_suffix(".png").name
            save_prediction(pred, save_path)
            print(f"Saved {save_path}")

            if label_path.is_file():
                label = torch.from_numpy(read_label(label_path)).unsqueeze(0)
                pred_t = torch.from_numpy(pred.astype(np.int64)).unsqueeze(0)
                image_confusion = np.zeros((config.num_classes, config.num_classes), dtype=np.int64)
                image_confusion = update_confusion(
                    image_confusion,
                    pred_t,
                    label,
                    config.num_classes,
                    config.ignore_index,
                    ignore_labels=config.eval_ignore_labels,
                )
                confusion = update_confusion(
                    confusion,
                    pred_t,
                    label,
                    config.num_classes,
                    config.ignore_index,
                    ignore_labels=config.eval_ignore_labels,
                )
                image_metrics = compute_metrics(image_confusion, config.eval_class_indices)
                per_image.append({
                    "acc": float(image_metrics["oa"]),
                    "macc": float(image_metrics["macc"]),
                    "miou": float(image_metrics["miou"]),
                    "fwiou": frequency_weighted_iou(image_confusion),
                    "f1": mean_f1(image_confusion, config.eval_class_indices),
                })

    if confusion.sum() > 0:
        metrics = compute_metrics(confusion, config.eval_class_indices)
        paper_metrics = None
        if per_image:
            paper_metrics = {
                "acc": float(np.mean([item["acc"] for item in per_image])),
                "macc": float(np.mean([item["macc"] for item in per_image])),
                "miou": float(np.mean([item["miou"] for item in per_image])),
                "fwiou": float(np.mean([item["fwiou"] for item in per_image])),
                "f1": float(np.mean([item["f1"] for item in per_image])),
            }
        if args.metric_mode in ("legacy-per-tile", "both") and paper_metrics is not None:
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
        np.save(output_dir / "confusion.npy", confusion)
        report = {
            'dataset': config.dataset, 'split': config.split, 'model': args.model,
            'checkpoint': args.checkpoint, 'architecture_version': config.architecture_version,
            'optical_product': args.optical_product, 'window_size': args.window_size,
            'stride': args.stride, 'batch_size': args.batch_size, 'tile_ids': list(tile_ids),
            'partial': set(tile_ids) != set(split_ids), 'metric_protocol': 'global confusion',
            'label_variant': 'full source labels',
            'macc': float(metrics['macc']) * 100, 'miou': float(metrics['miou']) * 100,
            'oa': float(metrics['oa']) * 100, 'kappa': float(metrics['kappa']) * 100,
            'class_iou': np.asarray(metrics['iou']).tolist(),
            'class_acc': np.asarray(metrics['acc']).tolist(), 'confusion': confusion.tolist(),
        }
        (output_dir / 'inference_metrics.json').write_text(json.dumps(report, indent=2))


def main() -> None:
    args = parse_args()
    config = CESDNetConfig(dataset=args.dataset, num_workers=args.num_workers, seed=args.seed)
    config = apply_runtime_args(config, args)
    if args.teacher_repo:
        config.teacher_repo = Path(args.teacher_repo)
    if args.teacher_backbone:
        config.teacher_backbone = Path(args.teacher_backbone)
    if not args.checkpoint:
        if args.model == "kd":
            args.checkpoint = config.paper_kd_checkpoint
        elif args.model in ("auto", "teacher"):
            args.checkpoint = config.paper_teacher_checkpoint
    require_checkpoint(args.checkpoint, "Provide --weights (or --checkpoint) with a complete compatible model checkpoint.")
    seed_everything(config.seed, deterministic=getattr(config, "deterministic", False))
    device = resolve_device(args)
    if args.model == "auto":
        args.model = infer_model_name(args.checkpoint, config)
        if args.model == "teacher" and not args.checkpoint:
            args.checkpoint = config.paper_teacher_checkpoint
        print(f"Auto selected {model_tag(args.model)} from weights: {args.checkpoint}")
    if args.model == "teacher":
        model = load_teacher_model(config, args.checkpoint, device)
    else:
        model = load_student_model(config, args.checkpoint, device, args.model)

    if args.mode == "patch":
        test_patches(model, config, args, device)
    else:
        test_tiles(model, config, args, device)


if __name__ == "__main__":
    main()
