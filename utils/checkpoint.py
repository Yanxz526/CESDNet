from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional

import torch
from torch import nn


def load_checkpoint_state(path: str | Path, preferred_keys: tuple[str, ...] = ()) -> Mapping[str, torch.Tensor]:
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        return checkpoint

    for key in preferred_keys:
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value

    for key in ("state_dict", "model", "net", "module", "teacher_state_dict", "student_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value
    return checkpoint


def strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def load_state_dict_flexible(
    model: nn.Module,
    checkpoint_path: str | Path,
    preferred_keys: tuple[str, ...] = (),
    strict: bool = False,
) -> torch.nn.modules.module._IncompatibleKeys:
    state_dict = load_checkpoint_state(checkpoint_path, preferred_keys=preferred_keys)
    if isinstance(state_dict, Mapping):
        state_dict = strip_module_prefix(state_dict)
    return model.load_state_dict(state_dict, strict=strict)


def save_checkpoint(path: str | Path, payload: MutableMapping[str, object]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), str(path))
