"""Поиск ближайшего пересечения луча с элементами сцены."""

from typing import List, Optional

import numpy as np

from core.rays import Ray
from core.segments import HitInfo
from surfaces.thin_lens import ThinLens


def find_best_hit(ray: "Ray", elements: List) -> Optional[HitInfo]:
    """Находит ближайшее пересечение луча с элементами сцены."""
    best_t = float("inf")
    best_hit = None
    wavelength = getattr(ray, "wavelength", None)

    for obj in elements:
        if hasattr(obj, "is_active") and not obj.is_active(wavelength):
            continue

        t = obj.intersect(ray)
        if t is None or t >= best_t:
            continue

        is_thin_lens = isinstance(obj, ThinLens)

        absorbed = False
        if hasattr(obj, "absorption_range") and obj.absorption_range is not None:
            if wavelength is None or (obj.absorption_range[0] <= wavelength <= obj.absorption_range[1]):
                absorbed = True

        allow_reflection = False
        if hasattr(obj, "reflection_range") and obj.reflection_range is not None:
            if wavelength is None or (obj.reflection_range[0] <= wavelength <= obj.reflection_range[1]):
                allow_reflection = True

        allow_refraction = False
        if hasattr(obj, "refraction_range") and obj.refraction_range is not None:
            if wavelength is None or (obj.refraction_range[0] <= wavelength <= obj.refraction_range[1]):
                allow_refraction = True

        if not is_thin_lens and not allow_reflection and not allow_refraction and not absorbed:
            continue

        n_inside = getattr(obj, "n", 1.0)
        best_t = t
        hit_point = ray.origin + ray.direction * t
        normal = obj.get_normal(hit_point) if hasattr(obj, "get_normal") else np.array([0.0, 0.0, 0.0])

        best_hit = HitInfo(
            obj=obj,
            t=t,
            point=hit_point,
            normal=normal,
            allow_reflection=allow_reflection,
            allow_refraction=allow_refraction,
            absorbed=absorbed,
            n_inside=n_inside,
            is_thin_lens=is_thin_lens,
        )

    return best_hit
