from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, MutableMapping

import torch
from torch import nn


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_checkpoint(path: str | Path, expected_sha256: str | None = None):
    path = Path(path).expanduser()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f'Checkpoint missing or empty: {path}')
    if expected_sha256 and checkpoint_sha256(path) != expected_sha256.lower():
        raise ValueError(f'Checkpoint SHA256 mismatch: {path}')
    try:
        # Training checkpoints include optimizer/RNG/config objects; load only trusted files.
        return torch.load(str(path), map_location='cpu', weights_only=False)
    except Exception as exc:
        raise RuntimeError(f'Cannot deserialize checkpoint {path}: {exc}') from exc


def load_checkpoint_state(path: str | Path, preferred_keys: tuple[str, ...] = ()) -> Mapping[str, torch.Tensor]:
    checkpoint = read_checkpoint(path)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f'Expected a state mapping in checkpoint: {path}')
    for key in (*preferred_keys, 'state_dict', 'model', 'net', 'module', 'teacher_state_dict', 'student_state_dict'):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            checkpoint = value
            break
    if not checkpoint or any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in checkpoint.items()):
        raise ValueError(f'Checkpoint does not contain a nonempty tensor state_dict: {path}')
    return checkpoint


def strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {(key[7:] if key.startswith('module.') else key): value for key, value in state_dict.items()}


def load_state_dict_flexible(model: nn.Module, checkpoint_path: str | Path, preferred_keys: tuple[str, ...] = (), strict: bool = True):
    """Accept known checkpoint containers, but require every model tensor to match."""
    if not strict:
        raise ValueError('Partial model loading is disabled for reproducibility; strict=True is required.')
    state = strip_module_prefix(load_checkpoint_state(checkpoint_path, preferred_keys))
    try:
        return model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f'Checkpoint is incompatible with {type(model).__name__}: {checkpoint_path}\n{exc}') from exc


def save_checkpoint(path: str | Path, payload: MutableMapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(dict(payload), temporary)
    temporary.replace(path)
