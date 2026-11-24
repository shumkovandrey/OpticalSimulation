"""Поиск ближайшего пересечения луча с элементами сцены."""

from typing import List, Optional

import numpy as np

from core.materials import Material
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

        material = getattr(obj, "material", Material())
        is_thin_lens = isinstance(obj, ThinLens)

        absorbed = False
        if material.absorption_range is not None:
            if wavelength is None or (material.absorption_range[0] <= wavelength <= material.absorption_range[1]):
                absorbed = True

        allow_reflection = False
        if material.reflection_range is not None:
            if wavelength is None or (material.reflection_range[0] <= wavelength <= material.reflection_range[1]):
                allow_reflection = True

        allow_refraction = False
        if material.refraction_range is not None:
            if wavelength is None or (material.refraction_range[0] <= wavelength <= material.refraction_range[1]):
                allow_refraction = True

        if not is_thin_lens and not allow_reflection and not allow_refraction and not absorbed:
            continue

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
            material=material,
            is_thin_lens=is_thin_lens,
        )

    return best_hit
