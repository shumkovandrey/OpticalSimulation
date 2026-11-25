"""Материал поверхности/линзы: оптика + внешний вид."""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np


import numpy as np

MATERIAL_PRESETS = {
    # --- Газы ---
    "Воздух": {
        "n": 1.0003, "empirical_coeff": 0.0, "a_offset": 0.0,
        "color": (0.85, 0.92, 1.0), "opacity": 0.1, "reflectivity": 0.0,
        "refraction_range": (0.0, np.inf),
        "reflection_range": None, "absorption_range": None,
    },
    # --- Жидкости ---
    "Вода": {
        "n": 1.333, "empirical_coeff": 0.0032, "a_offset": 0.0032,
        "color": (0.4, 0.7, 1.0), "opacity": 0.2, "reflectivity": 0.0,
        "refraction_range": (0.0, np.inf),
        "reflection_range": None, "absorption_range": None,
    },
    "Глицерин": {
        "n": 1.473, "empirical_coeff": 0.0045, "a_offset": 0.0045,
        "color": (0.6, 0.8, 0.9), "opacity": 0.25, "reflectivity": 0.0,
        "refraction_range": (0.0, np.inf),
        "reflection_range": None, "absorption_range": None,
    },
    # --- Стёкла ---
    "BK7": {
        "n": 1.5168, "empirical_coeff": 0.0042, "a_offset": 0.0042,
        "color": (0.7, 0.9, 1.0), "opacity": 0.5, "reflectivity": 0.0,
        "refraction_range": (0.0, np.inf),
        "reflection_range": None, "absorption_range": None,
    },
    "Крон": {
        "n": 1.52, "empirical_coeff": 0.004, "a_offset": 0.004,
        "color": (0.7, 0.9, 1.0), "opacity": 0.5, "reflectivity": 0.0,
        "refraction_range": (0.0, np.inf),
        "reflection_range": None, "absorption_range": None,
    },
    "Флинт": {
        "n": 1.62, "empirical_coeff": 0.010, "a_offset": 0.010,
        "color": (0.75, 0.85, 0.95), "opacity": 0.55, "reflectivity": 0.0,
        "refraction_range": (0.0, np.inf),
        "reflection_range": None, "absorption_range": None,
    },
    "Плавленый кварц": {
        "n": 1.458, "empirical_coeff": 0.0035, "a_offset": 0.0035,
        "color": (0.75, 0.95, 1.0), "opacity": 0.45, "reflectivity": 0.0,
        "refraction_range": (0.0, np.inf),
        "reflection_range": None, "absorption_range": None,
    },
    # --- Кристаллы ---
    "Алмаз": {
        "n": 2.417, "empirical_coeff": 0.014, "a_offset": 0.014,
        "color": (0.95, 0.98, 1.0), "opacity": 0.6, "reflectivity": 0.3,
        "refraction_range": (0.0, np.inf),
        "reflection_range": None, "absorption_range": None,
    },
    "Сапфир": {
        "n": 1.77, "empirical_coeff": 0.008, "a_offset": 0.008,
        "color": (0.6, 0.8, 1.0), "opacity": 0.5, "reflectivity": 0.1,
        "refraction_range": (0.0, np.inf),
        "reflection_range": None, "absorption_range": None,
    },
    "Кварц": {
        "n": 1.544, "empirical_coeff": 0.0055, "a_offset": 0.0055,
        "color": (0.8, 0.95, 1.0), "opacity": 0.5, "reflectivity": 0.0,
        "refraction_range": (0.0, np.inf),
        "reflection_range": None, "absorption_range": None,
    },
    # --- Полимеры ---
    "Акрил": {
        "n": 1.49, "empirical_coeff": 0.006, "a_offset": 0.006,
        "color": (0.9, 0.95, 1.0), "opacity": 0.5, "reflectivity": 0.0,
        "refraction_range": (0.0, np.inf),
        "reflection_range": None, "absorption_range": None,
    },
    "Поликарбонат": {
        "n": 1.585, "empirical_coeff": 0.009, "a_offset": 0.009,
        "color": (0.85, 0.9, 0.95), "opacity": 0.55, "reflectivity": 0.0,
        "refraction_range": (0.0, np.inf),
        "reflection_range": None, "absorption_range": None,
    },
    # --- Металлы (для зеркал) ---
    "Серебро (зеркало)": {
        "n": 1.5, "empirical_coeff": 0.0, "a_offset": 0.0,
        "color": (0.9, 0.9, 0.95), "opacity": 1.0, "reflectivity": 1.0,
        "refraction_range": None,
        "reflection_range": (0.0, np.inf),
        "absorption_range": None,
    },
    "Золото (зеркало)": {
        "n": 1.5, "empirical_coeff": 0.0, "a_offset": 0.0,
        "color": (1.0, 0.8, 0.2), "opacity": 1.0, "reflectivity": 1.0,
        "refraction_range": None,
        "reflection_range": (0.0, np.inf),
        "absorption_range": None,
    },
}


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
