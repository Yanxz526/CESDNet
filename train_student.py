from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from torch import optim
from torch.amp import GradScaler, autocast

from CESDNet.config import CESDNetConfig, REPO_ROOT
from CESDNet.models.student import build_cesdnet_s
from CESDNet.utils.checkpoint import load_state_dict_flexible, save_checkpoint
from CESDNet.utils.data import build_dataloaders, ensure_dir, seed_everything
from CESDNet.utils.metrics import compute_metrics, metric_line, update_confusion
from log import get_logger
from CESDNet.utils.runtime import require_checkpoint, resolve_device
from CESDNet.utils.runtime import (add_runtime_args, apply_runtime_args, limited_batches,
    checked_loss, snapshot_runtime, restore_training, data_signature)
from CESDNet.utils.runtime import SegmentationLoss as MscCrossEntropyLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CESDNet-S without KD.")
    parser.add_argument("--dataset", default="Potsdam", choices=["Potsdam", "Vaihingen"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--resume", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--amp", action="store_true")
    add_runtime_args(parser, training=True)
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> CESDNetConfig:
    return CESDNetConfig(
        dataset=args.dataset,
        epochs=args.epochs,
        student_batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
    )


def evaluate_student(model, val_loader, criterion, config: CESDNetConfig, device: torch.device):
    model.eval()
    val_loss = 0.0
    confusion = np.zeros((config.num_classes, config.num_classes), dtype=np.int64)
    with torch.no_grad():
        for sample in limited_batches(val_loader, config.max_val_batches):
            x = sample["image"].to(device, non_blocking=True).float()
            y = sample["label"].to(device, non_blocking=True).long()
            O_s = model(x)
            val_loss += float(criterion(O_s, y).item())
            confusion = update_confusion(
                confusion,
                O_s,
                y,
                config.num_classes,
                config.ignore_index,
                ignore_labels=config.eval_ignore_labels,
            )
    return val_loss / max(min(len(val_loader), config.max_val_batches or len(val_loader)), 1), compute_metrics(confusion, config.eval_class_indices)


def main() -> None:
    args = parse_args()
    config = apply_runtime_args(make_config(args), args)
    if args.resume:
        require_checkpoint(args.resume, "Provide --resume with a complete student training checkpoint.")
    elif not args.no_pretrained:
        require_checkpoint(config.dformer_weight,
                           "Provide --dformer-weight with DFormer-Tiny weights, or use --no-pretrained to train from scratch.")
    config.amp = args.amp
    seed_everything(config.seed, deterministic=getattr(config, "deterministic", False))
    device = resolve_device(args)

    train_loader, val_loader = build_dataloaders(config, batch_size=config.student_batch_size, require_ndsm=False)
    model = build_cesdnet_s(config, use_pretrained=not args.no_pretrained and not bool(args.resume)).to(device)

    if args.resume:
        load_state_dict_flexible(model, args.resume, preferred_keys=("student_state_dict",))

    criterion = MscCrossEntropyLoss(ignore_index=config.ignore_index).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "weight" / config.dataset / f"CESDNet-S({time.strftime('%Y-%m-%d-%H-%M')})"
    output_dir = ensure_dir(output_dir)
    logger = get_logger(str(output_dir))
    logger.info(
        f"CESDNet-S | dataset:{config.dataset} epochs:{config.epochs} batch:{config.student_batch_size} "
        f"lr:{config.lr} wd:{config.weight_decay}"
    )

    metrics_csv = output_dir / "metrics.csv"
    if not args.resume or not metrics_csv.is_file():
        with metrics_csv.open("w", newline="", encoding="utf-8") as fp:
            csv.writer(fp).writerow(["epoch", "train_loss", "val_loss", "macc", "miou", "oa", "kappa"])

    start_epoch, best_miou = 0, 0.0
    if args.resume:
        start_epoch, best_miou = restore_training(args.resume, optimizer, scheduler, scaler, train_loader, config)
    for epoch in range(start_epoch, config.epochs):
        model.train()
        train_loss = 0.0
        for step, sample in enumerate(limited_batches(train_loader, config.max_train_batches), start=1):
            x = sample["image"].to(device, non_blocking=True).float()
            y = sample["label"].to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                O_s = model(x)
                L_seg = criterion(O_s, y)
            scaler.scale(checked_loss(L_seg)).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(L_seg.item())

            if step % 50 == 0 or step == len(train_loader):
                print(
                    f"Epoch[{epoch + 1:03d}/{config.epochs:03d}] "
                    f"Iter[{step:04d}/{len(train_loader):04d}] L_seg:{L_seg.item():.4f}"
                )

        val_loss, metrics = evaluate_student(model, val_loader, criterion, config, device)
        train_loss /= max(min(len(train_loader), config.max_train_batches or len(train_loader)), 1)
        scheduler.step()

        logger.info(
            f"Epoch:{epoch + 1:03d} train:{train_loss:.6f} val:{val_loss:.6f} {metric_line(metrics)}"
        )
        print(f"Epoch {epoch + 1}: train={train_loss:.5f} val={val_loss:.5f} {metric_line(metrics)}")

        with metrics_csv.open("a", newline="", encoding="utf-8") as fp:
            csv.writer(fp).writerow([
                epoch + 1,
                train_loss,
                val_loss,
                float(metrics["macc"]) * 100,
                float(metrics["miou"]) * 100,
                float(metrics["oa"]) * 100,
                float(metrics["kappa"]) * 100,
            ])

        payload = {
            "epoch": epoch + 1,
            "student_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_miou": best_miou,
            "config": vars(config),
            **snapshot_runtime(scaler, train_loader),
            "data_signature": data_signature(config),
        }
        miou_pct = float(metrics["miou"]) * 100
        payload["best_miou"] = max(best_miou, miou_pct)
        save_checkpoint(output_dir / "CESDNet-S-last.pth", payload)
        miou_pct = float(metrics["miou"]) * 100
        if miou_pct >= best_miou:
            best_miou = miou_pct
            payload["best_miou"] = best_miou
            save_checkpoint(output_dir / f"CESDNet-S-{config.dataset}-best.pth", payload)
            print(f"Saved best CESDNet-S mIoU={best_miou:.3f}")

        if config.stop_after_epoch and epoch + 1 >= config.stop_after_epoch:
            print(f'Diagnostic stop after epoch {epoch + 1}; checkpoint saved.')
            break


if __name__ == "__main__":
    main()
