from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import optim
from torch.amp import GradScaler, autocast

from CESDNet.config import CESDNetConfig, REPO_ROOT
from CESDNet.models.teacher import (
    build_cesdnet_t,
    infer_teacher_head_profile,
    infer_teacher_texture_mode,
    infer_teacher_use_gmr_prior,
)
from CESDNet.utils.checkpoint import load_state_dict_flexible, save_checkpoint
from CESDNet.utils.data import build_dataloaders, ensure_dir, seed_everything
from CESDNet.utils.metrics import compute_metrics, metric_line, update_confusion
from log import get_logger
from toolbox.loss.loss import MscCrossEntropyLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CESDNet-T teacher.")
    parser.add_argument("--dataset", default="Potsdam", choices=["Potsdam", "Vaihingen"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--teacher-repo", default="")
    parser.add_argument("--teacher-backbone", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> CESDNetConfig:
    config = CESDNetConfig(
        dataset=args.dataset,
        epochs=args.epochs,
        teacher_batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    if args.teacher_repo:
        config.teacher_repo = Path(args.teacher_repo)
    if args.teacher_backbone:
        config.teacher_backbone = Path(args.teacher_backbone)
    return config


def evaluate_teacher(model, val_loader, criterion, config: CESDNetConfig, device: torch.device):
    model.eval()
    val_loss = 0.0
    confusion = np.zeros((config.num_classes, config.num_classes), dtype=np.int64)
    with torch.no_grad():
        for sample in val_loader:
            x = sample["image"].to(device, non_blocking=True).float()
            z = sample["dsm"].to(device, non_blocking=True).float()
            y = sample["label"].to(device, non_blocking=True).long()
            O_t = model(x, z, y=None, epoch=1)
            L_seg = criterion(O_t, y)
            L_ortho = getattr(model, "ortho_loss", O_t.new_zeros(())) * config.teacher_ortho_weight
            val_loss += float((L_seg + L_ortho).item())
            confusion = update_confusion(
                confusion,
                O_t,
                y,
                config.num_classes,
                config.ignore_index,
                ignore_labels=config.eval_ignore_labels,
            )
    metrics = compute_metrics(confusion, config.eval_class_indices)
    return val_loss / max(len(val_loader), 1), metrics


def main() -> None:
    args = parse_args()
    config = make_config(args)
    seed_everything(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_dataloaders(config, batch_size=config.teacher_batch_size)
    use_gmr_prior = infer_teacher_use_gmr_prior(args.resume) if args.resume else True
    texture_mode = infer_teacher_texture_mode(args.resume) if args.resume else "profile_b"
    head_profile = infer_teacher_head_profile(args.resume) if args.resume else "profile_b"
    model = build_cesdnet_t(
        config,
        use_gmr_prior=use_gmr_prior,
        texture_mode=texture_mode,
        head_profile=head_profile,
    ).to(device)

    if args.resume:
        msg = load_state_dict_flexible(model, args.resume, strict=False)
        print(
            f"Loaded resume checkpoint: missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}"
        )

    criterion = MscCrossEntropyLoss(ignore_index=config.ignore_index).to(device)
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "weight" / config.dataset / f"CESDNet-T({time.strftime('%Y-%m-%d-%H-%M')})"
    output_dir = ensure_dir(output_dir)
    logger = get_logger(str(output_dir))
    logger.info(
        f"CESDNet-T | dataset:{config.dataset} epochs:{config.epochs} batch:{config.teacher_batch_size} "
        f"lr:{config.lr} wd:{config.weight_decay} ortho:{config.teacher_ortho_weight}"
    )

    metrics_csv = output_dir / "metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as fp:
        csv.writer(fp).writerow(["epoch", "train_loss", "val_loss", "macc", "miou", "oa", "kappa"])

    best_miou = 0.0
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        for step, sample in enumerate(train_loader, start=1):
            x = sample["image"].to(device, non_blocking=True).float()
            z = sample["dsm"].to(device, non_blocking=True).float()
            y = sample["label"].to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                O_t = model(x, z, y=None, epoch=epoch + 1)
                L_seg = criterion(O_t, y)
                L_ortho = getattr(model, "ortho_loss", O_t.new_zeros(())) * config.teacher_ortho_weight
                L_teacher = L_seg + L_ortho

            scaler.scale(L_teacher).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(L_teacher.item())

            if step % 50 == 0 or step == len(train_loader):
                print(
                    f"Epoch[{epoch + 1:03d}/{config.epochs:03d}] "
                    f"Iter[{step:04d}/{len(train_loader):04d}] "
                    f"L_teacher:{L_teacher.item():.4f} L_seg:{L_seg.item():.4f} L_ortho:{L_ortho.item():.4f}"
                )

        val_loss, metrics = evaluate_teacher(model, val_loader, criterion, config, device)
        train_loss /= max(len(train_loader), 1)
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
            "teacher_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_miou": best_miou,
            "config": vars(config),
        }
        save_checkpoint(output_dir / "CESDNet-T-last.pth", payload)
        miou_pct = float(metrics["miou"]) * 100
        if miou_pct >= best_miou:
            best_miou = miou_pct
            payload["best_miou"] = best_miou
            save_checkpoint(output_dir / f"CESDNet-T-{config.dataset}-best.pth", payload)
            print(f"Saved best CESDNet-T mIoU={best_miou:.3f}")


if __name__ == "__main__":
    main()
