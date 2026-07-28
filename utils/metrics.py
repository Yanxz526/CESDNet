from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import torch


def update_confusion(
    confusion: np.ndarray,
    logits_or_pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
    ignore_labels: Optional[Iterable[int]] = None,
) -> np.ndarray:
    if logits_or_pred.ndim == 4:
        pred = logits_or_pred.argmax(dim=1)
    else:
        pred = logits_or_pred

    pred_np = pred.detach().cpu().numpy().astype(np.int64)
    target_np = target.detach().cpu().numpy().astype(np.int64)
    valid = (
        (target_np != ignore_index)
        & (target_np >= 0)
        & (target_np < num_classes)
        & (pred_np >= 0)
        & (pred_np < num_classes)
    )
    if ignore_labels:
        for label_id in ignore_labels:
            valid &= target_np != int(label_id)
    if valid.sum() == 0:
        return confusion

    label = num_classes * target_np[valid] + pred_np[valid]
    confusion += np.bincount(label, minlength=num_classes**2).reshape(num_classes, num_classes)
    return confusion


def compute_metrics(
    confusion: np.ndarray,
    eval_class_indices: Optional[Iterable[int]] = None,
) -> Dict[str, object]:
    confusion = confusion.astype(np.float64)
    tp = np.diag(confusion)
    row_sum = confusion.sum(axis=1)
    col_sum = confusion.sum(axis=0)
    total = confusion.sum()

    denom = row_sum + col_sum - tp
    iou = np.divide(tp, denom, out=np.full_like(tp, np.nan), where=denom > 0)
    acc = np.divide(tp, row_sum, out=np.full_like(tp, np.nan), where=row_sum > 0)

    if eval_class_indices is None:
        eval_indices = [idx for idx in range(len(tp)) if denom[idx] > 0]
    else:
        eval_indices = [idx for idx in eval_class_indices if idx < len(tp) and denom[idx] > 0]

    miou = float(np.nanmean(iou[eval_indices])) if eval_indices else 0.0
    macc = float(np.nanmean(acc[eval_indices])) if eval_indices else 0.0
    oa = float(tp.sum() / total) if total > 0 else 0.0
    pe = float((row_sum * col_sum).sum() / (total * total)) if total > 0 else 0.0
    kappa = float((oa - pe) / (1.0 - pe)) if abs(1.0 - pe) > 1e-12 else 0.0

    return {
        "acc": acc,
        "iou": iou,
        "macc": macc,
        "miou": miou,
        "oa": oa,
        "kappa": kappa,
        "confusion": confusion,
    }


def metric_line(metrics: Dict[str, object]) -> str:
    return (
        f"mAcc:{float(metrics['macc']) * 100:.3f} "
        f"mIoU:{float(metrics['miou']) * 100:.3f} "
        f"OA:{float(metrics['oa']) * 100:.3f} "
        f"Kappa:{float(metrics['kappa']) * 100:.3f}"
    )


def frequency_weighted_iou(confusion: np.ndarray) -> float:
    confusion = confusion.astype(np.float64)
    total = confusion.sum()
    if total <= 0:
        return 0.0
    freq = confusion.sum(axis=1) / total
    tp = np.diag(confusion)
    denom = confusion.sum(axis=1) + confusion.sum(axis=0) - tp
    iou = np.divide(tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float((freq[freq > 0] * iou[freq > 0]).sum())


def mean_f1(confusion: np.ndarray, eval_class_indices: Optional[Iterable[int]] = None) -> float:
    confusion = confusion.astype(np.float64)
    tp = np.diag(confusion)
    row_sum = confusion.sum(axis=1)
    col_sum = confusion.sum(axis=0)
    precision = np.divide(tp, col_sum, out=np.full_like(tp, np.nan), where=col_sum > 0)
    recall = np.divide(tp, row_sum, out=np.full_like(tp, np.nan), where=row_sum > 0)
    f1 = 2.0 * precision * recall / (precision + recall)
    if eval_class_indices is None:
        indices = [idx for idx in range(len(tp)) if not np.isnan(f1[idx])]
    else:
        indices = [idx for idx in eval_class_indices if idx < len(tp) and not np.isnan(f1[idx])]
    return float(np.nanmean(f1[indices])) if indices else 0.0
