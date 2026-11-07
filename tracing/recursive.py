"""Рекурсивная трассировка с ветвлением (TreeMode)."""

from typing import List, Optional

import numpy as np

from constants import RAY_INFINITY_DISTANCE, DEFAULT_OFFSET_DISTANCE, DEFAULT_MIN_ENERGY
from core.rays import Ray, RayPool, WhiteRay, DispersiveRay
from core.segments import Segment
from core.dispersion import get_dispersion_n
from tracing.hit_finder import find_best_hit
from tracing.interaction import split_ray


def _trace_recursive(ray: 'Ray',
                     elements: List,
                     depth: int,
                     min_energy: float = DEFAULT_MIN_ENERGY,
                     offset_distance: float = DEFAULT_OFFSET_DISTANCE,
                     total_limit: int = 5000,
                     pool: Optional[RayPool] = None) -> List[Segment]:
    """Рекурсивная трассировка дерева лучей.

    Полностью эквивалентна _trace_recursive из main.py (шаг 5).
    """
    segments = []

    def recurse(current_ray: 'Ray', d: int, from_pool: bool):
        nonlocal segments

        if len(segments) >= total_limit or d <= 0 or current_ray.energy < min_energy:
            return

        if isinstance(current_ray, WhiteRay):
            spectral_components = current_ray.generate_rays(pool=pool)
            for comp in spectral_components:
                recurse(comp, d, from_pool=(pool is not None))
                if pool:
                    pool.release(comp)
            return

        hit = find_best_hit(current_ray, elements)

        ray_color = current_ray.color
        if isinstance(current_ray, DispersiveRay) and current_ray.wavelength is not None:
            ray_color = current_ray.wavelength_to_rgb(current_ray.wavelength)

        if hit is None:
            end = current_ray.origin + current_ray.direction * RAY_INFINITY_DISTANCE
            segments.append(Segment(
                current_ray.origin.copy(), end.copy(),
                current_ray.energy, ray_color,
            ))
            return
        else:
            segments.append(Segment(
                current_ray.origin.copy(), hit.point.copy(),
                current_ray.energy, ray_color,
            ))

        if hit.absorbed:
            return

        if hit.is_thin_lens:
            new_dir = hit.obj.thin_lens_deflection(current_ray.direction, hit.point)
            start = hit.point + offset_distance * new_dir

            if pool:
                new_ray = pool.acquire(
                    start, new_dir,
                    energy=current_ray.energy,
                    current_n=current_ray.current_n,
                    color=current_ray.color,
                    wavelength=current_ray.wavelength,
                    energy_color_type=current_ray.energy_color_type,
                )
            else:
                new_ray = Ray(
                    start, new_dir,
                    energy=current_ray.energy,
                    current_n=current_ray.current_n,
                    color=current_ray.color,
                    wavelength=current_ray.wavelength,
                    energy_color_type=current_ray.energy_color_type,
                )

            recurse(new_ray, d - 1, from_pool=(pool is not None))
            if pool:
                pool.release(new_ray)
            return

        resolved_n_inside = get_dispersion_n(hit.n_inside, current_ray)
        n_next = resolved_n_inside if abs(current_ray.current_n - 1.0) < 1e-6 else 1.0

        if not hit.allow_reflection and not hit.allow_refraction:
            start = hit.point + offset_distance * current_ray.direction
            if pool:
                new_ray = pool.acquire(
                    start, current_ray.direction,
                    energy=current_ray.energy,
                    current_n=current_ray.current_n,
                    color=current_ray.color,
                    wavelength=current_ray.wavelength,
                    energy_color_type=current_ray.energy_color_type,
                )
            else:
                new_ray = Ray(
                    start, current_ray.direction,
                    energy=current_ray.energy,
                    current_n=current_ray.current_n,
                    color=current_ray.color,
                    wavelength=current_ray.wavelength,
                    energy_color_type=current_ray.energy_color_type,
                )
            recurse(new_ray, d - 1, from_pool=(pool is not None))
            if pool:
                pool.release(new_ray)
            return

        new_rays = split_ray(
            current_ray, hit.normal, n_next, hit.point,
            allow_reflection=hit.allow_reflection,
            allow_refraction=hit.allow_refraction,
            offset_distance=offset_distance,
            pool=pool,
        )

        for nr in new_rays:
            recurse(nr, d - 1, from_pool=(pool is not None))
            if pool:
                pool.release(nr)

    recurse(ray, depth, from_pool=False)
    return segments
