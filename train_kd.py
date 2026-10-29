from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Iterable

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from torch import optim
from torch.amp import GradScaler, autocast

from CESDNet.config import CESDNetConfig, REPO_ROOT
from CESDNet.losses.structured_kd import MultiStageStudentKDHead, StructuredKDLoss
from CESDNet.models.student import build_cesdnet_s
from CESDNet.models.teacher import (
    build_cesdnet_t,
    infer_teacher_head_profile,
    infer_teacher_texture_mode,
    infer_teacher_use_gmr_prior,
)
from CESDNet.utils.checkpoint import load_state_dict_flexible, save_checkpoint, checkpoint_sha256
from CESDNet.utils.data import build_dataloaders, ensure_dir, seed_everything
from CESDNet.utils.metrics import compute_metrics, metric_line, update_confusion
from log import get_logger
from CESDNet.utils.runtime import require_checkpoint, resolve_device
from CESDNet.utils.runtime import (add_runtime_args, apply_runtime_args, limited_batches,
    checked_loss, snapshot_runtime, restore_training, data_signature)
from CESDNet.utils.runtime import SegmentationLoss as MscCrossEntropyLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CESDNet-KD with SCD, CPD and RCRD.")
    parser.add_argument("--dataset", default="Potsdam", choices=["Potsdam", "Vaihingen"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--teacher-checkpoint", default=os.environ.get("TEACHER_CKPT", ""))
    parser.add_argument("--student-checkpoint", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--teacher-repo", default="")
    parser.add_argument("--teacher-backbone", default="")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--amp", action="store_true")
    add_runtime_args(parser, training=True)
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> CESDNetConfig:
    config = CESDNetConfig(
        dataset=args.dataset,
        epochs=args.epochs,
        kd_batch_size=args.batch_size,
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


def trainable_parameters(*modules: torch.nn.Module) -> Iterable[torch.nn.Parameter]:
    for module in modules:
        for param in module.parameters():
            if param.requires_grad:
                yield param


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
    teacher_checkpoint = args.teacher_checkpoint or config.paper_teacher_checkpoint
    require_checkpoint(teacher_checkpoint, "Provide --teacher-checkpoint with a complete CESDNet-T checkpoint.")
    if args.resume:
        require_checkpoint(args.resume, "Provide --resume with a complete KD training checkpoint.")
    if args.student_checkpoint:
        require_checkpoint(args.student_checkpoint, "Provide --student-checkpoint with compatible student weights.")
    elif not args.resume and not args.no_pretrained:
        require_checkpoint(config.dformer_weight,
                           "Provide --student-checkpoint or --dformer-weight, or use --no-pretrained to initialize the student from scratch.")
    config.amp = args.amp
    seed_everything(config.seed, deterministic=getattr(config, "deterministic", False))
    device = resolve_device(args)

    train_loader, val_loader = build_dataloaders(config, batch_size=config.kd_batch_size)

    teacher_sha256 = checkpoint_sha256(teacher_checkpoint)
    use_gmr_prior = infer_teacher_use_gmr_prior(teacher_checkpoint)
    texture_mode = infer_teacher_texture_mode(teacher_checkpoint)
    head_profile = infer_teacher_head_profile(teacher_checkpoint)
    teacher = build_cesdnet_t(
        config,
        use_gmr_prior=use_gmr_prior,
        texture_mode=texture_mode,
        head_profile=head_profile,
        checkpoint_path=teacher_checkpoint,
    ).to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    student = build_cesdnet_s(config, use_pretrained=not args.no_pretrained and not bool(args.student_checkpoint or args.resume)).to(device)
    if args.student_checkpoint:
        msg = load_state_dict_flexible(
            student,
            args.student_checkpoint,
            preferred_keys=("student_state_dict", "state_dict", "model", "net"),
            strict=True,
        )
        print(f"Loaded CESDNet-S init: missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")

    kd_heads = MultiStageStudentKDHead(
        in_channels={"c3": 128, "c4": 256},
        stages=config.kd_stages,
    ).to(device)

    criterion_seg = MscCrossEntropyLoss(ignore_index=config.ignore_index).to(device)
    criterion_kd = StructuredKDLoss(config).to(device)
    optimizer = optim.AdamW(
        list(trainable_parameters(student, kd_heads)),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    start_epoch = 0
    best_miou = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("teacher_sha256") != teacher_sha256:
            raise ValueError("KD resume teacher checkpoint SHA256 mismatch")
        student.load_state_dict(checkpoint["student_state_dict"], strict=True)
        kd_heads.load_state_dict(checkpoint["kd_heads_state_dict"], strict=True)
        start_epoch, best_miou = restore_training(args.resume, optimizer, scheduler, scaler, train_loader, config)
        print(f"Resume CESDNet-KD from epoch {start_epoch}, best_mIoU={best_miou:.3f}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else REPO_ROOT / "weight" / config.dataset / f"CESDNet-KD({time.strftime('%Y-%m-%d-%H-%M')})"
    )
    output_dir = ensure_dir(output_dir)
    logger = get_logger(str(output_dir))
    logger.info(
        f"CESDNet-KD | dataset:{config.dataset} epochs:{config.epochs} batch:{config.kd_batch_size} "
        f"lr:{config.lr} wd:{config.weight_decay} stages:{config.kd_stages} "
        f"alpha:{config.kd_stage_weights} lambda_kd:{config.lambda_kd} "
        f"lambda_sh:{config.lambda_sh} lambda_pr:{config.lambda_pr} lambda_cr:{config.lambda_cr} "
        f"teacher:{teacher_checkpoint}"
    )

    metrics_csv = output_dir / "metrics.csv"
    if start_epoch == 0 or not metrics_csv.is_file():
        with metrics_csv.open("w", newline="", encoding="utf-8") as fp:
            csv.writer(fp).writerow([
                "epoch",
                "train_loss",
                "seg",
                "L_SCD",
                "L_CPD",
                "L_RCRD",
                "L_CESD_KD",
                "val_loss",
                "macc",
                "miou",
                "oa",
                "kappa",
            ])

    for epoch in range(start_epoch, config.epochs):
        student.train()
        kd_heads.train()
        teacher.eval()
        sums = {"total": 0.0, "seg": 0.0, "L_SCD": 0.0, "L_CPD": 0.0, "L_RCRD": 0.0, "L_CESD_KD": 0.0}

        for step, sample in enumerate(limited_batches(train_loader, config.max_train_batches), start=1):
            x = sample["image"].to(device, non_blocking=True).float()
            z = sample["dsm"].to(device, non_blocking=True).float()
            y = sample["label"].to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                with autocast(device_type=device.type, enabled=False):
                    O_t, kd_t = teacher(x, z, y=None, return_kd=True, epoch=epoch + 1)

            with autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                s_out = student(x, return_features=True)
                O_s = s_out["O_s"]
                student_kd = kd_heads(s_out["F_s_enh"])
                L_seg = criterion_seg(O_s, y)
                terms = criterion_kd(O_s, O_t, student_kd, kd_t, L_seg=L_seg, epoch=epoch)
                L_total = terms.L_total if terms.L_total is not None else L_seg + terms.L_CESD_KD

            scaler.scale(checked_loss(L_total)).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(trainable_parameters(student, kd_heads)), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

            meters = terms.meters()
            sums["total"] += float(L_total.item())
            sums["seg"] += float(L_seg.item())
            for key in ("L_SCD", "L_CPD", "L_RCRD", "L_CESD_KD"):
                sums[key] += meters[key]

            if step % 50 == 0 or step == len(train_loader):
                eta_kd = min(1.0, float(epoch + 1) / float(max(1, config.kd_ramp_epochs)))
                print(
                    f"Epoch[{epoch + 1:03d}/{config.epochs:03d}] "
                    f"Iter[{step:04d}/{len(train_loader):04d}] "
                    f"L_total:{L_total.item():.4f} L_seg:{L_seg.item():.4f} "
                    f"SCD:{terms.L_SCD.item():.4f} CPD:{terms.L_CPD.item():.4f} "
                    f"RCRD:{terms.L_RCRD.item():.4f} KD:{terms.L_CESD_KD.item():.4f} "
                    f"eta:{eta_kd:.2f}"
                )

        scheduler.step()
        val_loss, metrics = evaluate_student(student, val_loader, criterion_seg, config, device)
        n = max(min(len(train_loader), config.max_train_batches or len(train_loader)), 1)
        avg_total = sums["total"] / n
        avg_seg = sums["seg"] / n
        avg_scd = sums["L_SCD"] / n
        avg_cpd = sums["L_CPD"] / n
        avg_rcrd = sums["L_RCRD"] / n
        avg_kd = sums["L_CESD_KD"] / n

        logger.info(
            f"Epoch:{epoch + 1:03d} train:{avg_total:.6f} seg:{avg_seg:.6f} "
            f"SCD:{avg_scd:.6f} CPD:{avg_cpd:.6f} RCRD:{avg_rcrd:.6f} KD:{avg_kd:.6f} "
            f"val:{val_loss:.6f} {metric_line(metrics)}"
        )
        print(
            f"Epoch {epoch + 1}: train={avg_total:.5f} val={val_loss:.5f} "
            f"SCD={avg_scd:.4f} CPD={avg_cpd:.4f} RCRD={avg_rcrd:.4f} {metric_line(metrics)}"
        )

        with metrics_csv.open("a", newline="", encoding="utf-8") as fp:
            csv.writer(fp).writerow([
                epoch + 1,
                avg_total,
                avg_seg,
                avg_scd,
                avg_cpd,
                avg_rcrd,
                avg_kd,
                val_loss,
                float(metrics["macc"]) * 100,
                float(metrics["miou"]) * 100,
                float(metrics["oa"]) * 100,
                float(metrics["kappa"]) * 100,
            ])

        payload = {
            "epoch": epoch + 1,
            "student_state_dict": student.state_dict(),
            "kd_heads_state_dict": kd_heads.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_miou": best_miou,
            "metrics": {
                "macc": float(metrics["macc"]) * 100,
                "miou": float(metrics["miou"]) * 100,
                "oa": float(metrics["oa"]) * 100,
                "kappa": float(metrics["kappa"]) * 100,
                "iou": np.asarray(metrics["iou"]).tolist(),
                "acc": np.asarray(metrics["acc"]).tolist(),
            },
            "config": vars(config),
            **snapshot_runtime(scaler, train_loader),
            "data_signature": data_signature(config),
            "teacher_checkpoint": teacher_checkpoint,
            "teacher_sha256": teacher_sha256,
        }
        miou_pct = float(metrics["miou"]) * 100
        payload["best_miou"] = max(best_miou, miou_pct)
        save_checkpoint(output_dir / "CESDNet-KD-last.pth", payload)
        miou_pct = float(metrics["miou"]) * 100
        if miou_pct >= best_miou:
            best_miou = miou_pct
            payload["best_miou"] = best_miou
            save_checkpoint(output_dir / f"CESDNet-KD-{config.dataset}-best.pth", payload)
            print(f"Saved best CESDNet-KD mIoU={best_miou:.3f}")

        if config.stop_after_epoch and epoch + 1 >= config.stop_after_epoch:
            print(f'Diagnostic stop after epoch {epoch + 1}; checkpoint saved.')
            break


if __name__ == "__main__":
    main()
