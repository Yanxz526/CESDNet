"""Shared training contracts: explicit configuration, finite losses and complete training-state resume."""
from __future__ import annotations

import itertools
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from CESDNet.utils.checkpoint import read_checkpoint


class SegmentationLoss(nn.Module):
    """Pixel CE, excluding label 255 as specified in the manuscript."""
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        if not torch.any(labels != self.ignore_index):
            return logits.sum() * 0.0
        return F.cross_entropy(logits, labels, ignore_index=self.ignore_index)


def require_checkpoint(path, hint):
    checkpoint = Path(path).expanduser() if path else None
    if checkpoint is None or not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise SystemExit(f"Missing or empty weights: {path or '(not specified)'}. {hint}")


def add_runtime_args(parser, training=False):
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--deterministic', action='store_true', help='Require deterministic PyTorch kernels; unsupported operations stop with an error.')
    parser.add_argument('--data-root', default='', help='Directory containing train.txt, val.txt, test.txt.')
    parser.add_argument('--dataset-config', default='')
    parser.add_argument('--isprs-root', default='')
    parser.add_argument('--architecture-version', choices=['paper_v1', 'legacy_v0'], default='paper_v1')
    parser.add_argument('--student-arch', choices=['paper', 'cafe', 'hfip'], default='cafe')
    if training:
        parser.add_argument('--dformer-weight', default='')
        parser.add_argument('--stop-after-epoch', type=int, default=0, help='Diagnostic interruption after saving an epoch; 0 = run all.')
        parser.add_argument('--max-train-batches', type=int, default=0, help='0 = complete epoch; positive = diagnostic run.')
        parser.add_argument('--max-val-batches', type=int, default=0, help='0 = complete validation; positive = diagnostic run.')
    return parser


def apply_runtime_args(config, args):
    for key in ('data_root', 'dataset_config', 'isprs_root', 'architecture_version', 'student_arch', 'split', 'dformer_weight', 'deterministic', 'device'):
        value = getattr(args, key, None)
        if value not in (None, ''):
            setattr(config, key, value)
    for key in ('max_train_batches', 'max_val_batches', 'stop_after_epoch'):
        value = getattr(args, key, 0)
        if value < 0:
            raise ValueError(f'{key} must be nonnegative')
        setattr(config, key, value)
    if config.epochs < 1:
        raise ValueError('epochs must be positive')
    return config


def limited_batches(loader, maximum=0):
    return itertools.islice(loader, maximum) if maximum else loader


def checked_loss(loss):
    if not loss.requires_grad or not torch.isfinite(loss).item():
        raise RuntimeError(f'Training loss must be differentiable and finite, got {loss}')
    return loss


def data_signature(config):
    digest = hashlib.sha256()
    contents = config.config_path.read_bytes()
    digest.update(contents)
    manifests = json.loads(contents).get('split_manifests', {})
    for mode in ('train', 'val'):
        digest.update((config.dataset_root / manifests.get(mode, f'{mode}.txt')).read_bytes())
    digest.update(str(config.dataset_root.resolve()).encode())
    return digest.hexdigest()


def snapshot_runtime(scaler, loader):
    return {
        'scaler_state_dict': scaler.state_dict(),
        'rng_state': {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
                      'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
        'loader_generator_state': loader.generator.get_state(),
    }


def restore_training(path, optimizer, scheduler, scaler, loader, config):
    checkpoint = read_checkpoint(path)
    required = ('optimizer_state_dict', 'scheduler_state_dict', 'scaler_state_dict', 'epoch', 'rng_state', 'loader_generator_state', 'config', 'data_signature')
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f'Training-state resume requires {missing}; use model initialization options for weights-only checkpoints.')
    if checkpoint['data_signature'] != data_signature(config):
        raise ValueError('Resume data manifests/config/root differ from the saved run')
    saved = checkpoint['config']
    for key in ('dataset', 'student_arch', 'architecture_version', 'lr', 'weight_decay', 'epochs', 'seed', 'teacher_batch_size', 'student_batch_size', 'kd_batch_size', 'num_workers', 'kd_ramp_epochs', 'lambda_kd', 'lambda_sh', 'lambda_pr', 'lambda_cr', 'lambda_tex', 'lambda_geo', 'tau', 'lambda_conf', 'lambda_rel', 'eta_r', 'kd_stages', 'kd_stage_weights', 'teacher_ortho_weight', 'max_train_batches', 'max_val_batches', 'amp', 'deterministic', 'device'):
        if saved.get(key) != getattr(config, key, None):
            raise ValueError(f'Resume config mismatch for {key}: saved={saved.get(key)}, requested={getattr(config, key)}')
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    rng = checkpoint['rng_state']
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch'])
    if torch.cuda.is_available() and rng['cuda']:
        torch.cuda.set_rng_state_all(rng['cuda'])
    loader.generator.set_state(checkpoint['loader_generator_state'])
    return int(checkpoint['epoch']), float(checkpoint.get('best_miou', 0.0))


def resolve_device(args):
    requested = getattr(args, 'device', 'auto')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('--device cuda was requested but CUDA is unavailable')
    return torch.device('cuda' if requested == 'auto' and torch.cuda.is_available() else 'cpu' if requested == 'auto' else requested)
