"""Контейнеры данных: Segment (результат трассировки) и HitInfo (попадание)."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class Segment:
    """Один отрезок трассированного луча."""
    start: np.ndarray
    end: np.ndarray
    energy: float
    color: tuple

    def __iter__(self):
        return iter((self.start, self.end, self.energy, self.color))


@dataclass(slots=True)
class HitInfo:
    """Информация о пересечении луча с объектом сцены."""
    obj: Any
    t: float
    point: np.ndarray
    normal: np.ndarray
    allow_reflection: bool
    allow_refraction: bool
    absorbed: bool
    n_inside: float
    is_thin_lens: bool = False

    def __post_init__(self):
        self.point = np.asarray(self.point, dtype=np.float64)
        self.normal = np.asarray(self.normal, dtype=np.float64)
