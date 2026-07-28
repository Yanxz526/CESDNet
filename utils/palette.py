from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


PALETTE: Dict[int, Tuple[int, int, int]] = {
    0: (255, 255, 255),
    1: (0, 0, 255),
    2: (0, 255, 255),
    3: (0, 255, 0),
    4: (255, 255, 0),
    5: (255, 0, 0),
    6: (0, 0, 0),
}
INVERT_PALETTE = {value: key for key, value in PALETTE.items()}


def label_to_color(label: np.ndarray) -> np.ndarray:
    color = np.zeros((*label.shape, 3), dtype=np.uint8)
    for cls_id, rgb in PALETTE.items():
        color[label == cls_id] = rgb
    return color


def color_to_label(color: np.ndarray) -> np.ndarray:
    label = np.zeros(color.shape[:2], dtype=np.uint8)
    for rgb, cls_id in INVERT_PALETTE.items():
        label[np.all(color == np.array(rgb, dtype=np.uint8).reshape(1, 1, 3), axis=2)] = cls_id
    return label
