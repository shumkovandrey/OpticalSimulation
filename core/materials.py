"""Материал поверхности/линзы: оптика + внешний вид."""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np


def parse_range(value):
    """Нормализует диапазон длин волн из любого формата"""
    if value is None:
        return None
    lo, hi = value[0], value[1]
    if hi == "Infinity" or (isinstance(hi, float) and np.isinf(hi)):
        hi = np.inf
    return (float(lo), float(hi))


@dataclass
class Material:
    # Оптика
    n: float = 1.5
    empirical_coeff: float = 0.005
    a_offset: float = 0.01

    # Диапазоны длин волн
    refraction_range: Optional[Tuple[float, float]] = (0.0, np.inf)
    reflection_range: Optional[Tuple[float, float]] = None
    absorption_range: Optional[Tuple[float, float]] = None

    # Визуал
    color: Tuple[float, float, float] = (0.0, 1.0, 1.0)
    opacity: float = 0.5
    reflectivity: float = 0.0

    def copy(self) -> "Material":
        return Material(
            n=self.n,
            empirical_coeff=self.empirical_coeff,
            a_offset=self.a_offset,
            refraction_range=self.refraction_range,
            reflection_range=self.reflection_range,
            absorption_range=self.absorption_range,
            color=tuple(self.color),
            opacity=self.opacity,
            reflectivity=self.reflectivity,
        )

    def to_dict(self) -> dict:
        """Сериализация в JSON (для save_scene)"""

        return {
            "n": self.n,
            "empirical_coeff": self.empirical_coeff,
            "a_offset": self.a_offset,
            "refraction_range": parse_range(self.refraction_range),
            "reflection_range": parse_range(self.reflection_range),
            "absorption_range": parse_range(self.absorption_range),
            "color": list(self.color),
            "opacity": self.opacity,
            "reflectivity": self.reflectivity,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Material":
        def _rng(r):
            if r is None:
                return None
            hi = np.inf if r[1] == "Infinity" or (isinstance(r[1], float) and np.isinf(r[1])) else r[1]
            return (float(r[0]), float(hi))

        return cls(
            n=float(d.get("n", 1.5)),
            empirical_coeff=float(d.get("empirical_coeff", 0.005)),
            a_offset=float(d.get("a_offset", 0.01)),
            refraction_range=_rng(d.get("refraction_range", (0.0, np.inf))),
            reflection_range=_rng(d.get("reflection_range")),
            absorption_range=_rng(d.get("absorption_range")),
            color=tuple(d.get("color", (0.0, 1.0, 1.0))),
            opacity=float(d.get("opacity", 0.5)),
            reflectivity=float(d.get("reflectivity", 0.0)),
        )
