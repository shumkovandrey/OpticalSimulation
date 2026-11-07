"""Поиск ближайшего пересечения луча с элементами сцены."""

from typing import List, Optional

import numpy as np

from core.segments import HitInfo
from core.rays import Ray
from surfaces.thin_lens import ThinLens


def find_best_hit(ray: 'Ray', elements: List) -> Optional[HitInfo]:
    """Находит ближайшее пересечение луча с элементами сцены.

    Полностью эквивалентна find_best_hit из main.py (шаг 5).
    """
    best_t = float('inf')
    best_hit = None

    for obj in elements:
        if hasattr(obj, 'is_active') and not obj.is_active(ray.wavelength):
            continue

        t = obj.intersect(ray)
        if t is not None and t < best_t:
            best_t = t
            hit_point = ray.origin + ray.direction * t
            normal = obj.get_normal(hit_point) if hasattr(obj, 'get_normal') else np.array([0., 0., 0.])

            allow_reflection = False
            allow_refraction = False
            absorbed = False
            n_inside = 1.0
            is_thin_lens = isinstance(obj, ThinLens)

            if hasattr(obj, 'absorption_range') and obj.absorption_range is not None:
                if ray.wavelength is None or (obj.absorption_range[0] <= ray.wavelength <= obj.absorption_range[1]):
                    absorbed = True

            if hasattr(obj, 'reflection_range') and obj.reflection_range is not None:
                if ray.wavelength is None or (obj.reflection_range[0] <= ray.wavelength <= obj.reflection_range[1]):
                    allow_reflection = True

            if hasattr(obj, 'refraction_range') and obj.refraction_range is not None:
                if ray.wavelength is None or (obj.refraction_range[0] <= ray.wavelength <= obj.refraction_range[1]):
                    allow_refraction = True

            if hasattr(obj, 'n'):
                n_inside = obj.n

            if not is_thin_lens and not allow_reflection and not allow_refraction and not absorbed:
                continue

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
