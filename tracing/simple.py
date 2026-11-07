"""Однолучевая последовательная трассировка (SimpleMode)."""

from typing import List, Optional

import numpy as np

from constants import RAY_INFINITY_DISTANCE, DEFAULT_OFFSET_DISTANCE
from core.rays import Ray, RayPool, DispersiveRay
from core.segments import Segment
from core.dispersion import get_dispersion_n
from fast_math import refract
from tracing.hit_finder import find_best_hit


def _trace_simple(ray: 'Ray',
                  elements: List,
                  max_bounces: int = 10,
                  offset_distance: float = DEFAULT_OFFSET_DISTANCE,
                  prioritize_refraction: bool = True,
                  pool: Optional['RayPool'] = None) -> List[Segment]:
    """Однолучевая последовательная трассировка.

    Полностью эквивалентна _trace_simple из main.py (шаг 5).
    НЕ ИСПРАВЛЯЕТ известный баг обработки ТИР — это будет на шаге 8.
    """
    segments: List[Segment] = []

    current_ray = ray
    current_n = ray.current_n
    current_from_pool = False

    for _ in range(max_bounces):
        hit = find_best_hit(current_ray, elements)

        if hit is None:
            end_point = current_ray.origin + current_ray.direction * RAY_INFINITY_DISTANCE
            segments.append(Segment(
                current_ray.origin.copy(), end_point.copy(),
                current_ray.energy, current_ray.color,
            ))
            break

        segments.append(Segment(
            current_ray.origin.copy(), hit.point.copy(),
            current_ray.energy, current_ray.color,
        ))

        if hit.absorbed:
            break

        if hit.is_thin_lens:
            new_dir = hit.obj.thin_lens_deflection(current_ray.direction, hit.point)
            start = hit.point + offset_distance * new_dir

            if pool:
                if current_from_pool:
                    pool.release(current_ray)
                next_ray = pool.acquire(
                    origin=start, direction=new_dir,
                    energy=current_ray.energy, current_n=current_n,
                    color=current_ray.color, wavelength=current_ray.wavelength,
                    energy_color_type=current_ray.energy_color_type,
                    ray_class=type(current_ray),
                )
                current_from_pool = True
            else:
                next_ray = type(current_ray)(
                    start, new_dir,
                    energy=current_ray.energy, current_n=current_n,
                    color=current_ray.color, wavelength=current_ray.wavelength,
                    energy_color_type=current_ray.energy_color_type,
                )
                current_from_pool = False
            current_ray = next_ray
            continue

        allow_reflection = hit.allow_reflection
        allow_refraction = hit.allow_refraction

        if prioritize_refraction and allow_refraction:
            allow_reflection = False

        if not allow_reflection and not allow_refraction:
            start = hit.point + offset_distance * current_ray.direction
            if pool:
                if current_from_pool:
                    pool.release(current_ray)
                next_ray = pool.acquire(
                    origin=start, direction=current_ray.direction,
                    energy=current_ray.energy, current_n=current_n,
                    color=current_ray.color, wavelength=current_ray.wavelength,
                    energy_color_type=current_ray.energy_color_type,
                    ray_class=type(current_ray),
                )
                current_from_pool = True
            else:
                next_ray = type(current_ray)(
                    start, current_ray.direction,
                    energy=current_ray.energy, current_n=current_n,
                    color=current_ray.color, wavelength=current_ray.wavelength,
                    energy_color_type=current_ray.energy_color_type,
                )
                current_from_pool = False
            current_ray = next_ray
            continue

        if allow_refraction:
            resolved_n_inside = get_dispersion_n(hit.n_inside, current_ray)
            n_next = resolved_n_inside if abs(current_n - 1.0) < 1e-6 else 1.0

            # Явная проверка полного внутреннего отражения.
            # На ТИР луч не может преломиться, значит current_n НЕ меняется,
            # и мы принудительно включаем отражение (даже если prioritize_refraction
            # его подавил).
            normal_tir = hit.normal
            if np.dot(normal_tir, current_ray.direction) > 0:
                normal_tir = -normal_tir
            cos_i_tir = -np.dot(normal_tir, current_ray.direction)
            eta = current_n / n_next
            sin2_t = eta * eta * (1.0 - cos_i_tir * cos_i_tir)

            if sin2_t >= 1.0:
                # ТИР: остаёмся в текущей среде, отражаемся.
                allow_reflection = True
                allow_refraction = False
                # current_n не трогаем!
            else:
                refracted_dir = refract(current_ray.direction, hit.normal, current_n, n_next)
                # По построению refract() возвращает не-None вне ТИР;
                # но оставляем защиту на случай будущих правок.
                if refracted_dir is not None:
                    current_n = n_next
                    new_dir = refracted_dir
                    allow_reflection = False
                else:
                    allow_reflection = True

        if allow_reflection:
            normal = hit.normal
            if np.dot(normal, current_ray.direction) > 0:
                normal = -normal
            new_dir = current_ray.direction - 2 * np.dot(current_ray.direction, normal) * normal
            new_dir /= np.linalg.norm(new_dir)

        start = hit.point + offset_distance * new_dir
        if pool:
            if current_from_pool:
                pool.release(current_ray)
            next_ray = pool.acquire(
                origin=start, direction=new_dir,
                energy=current_ray.energy, current_n=current_n,
                color=current_ray.color, wavelength=current_ray.wavelength,
                energy_color_type=current_ray.energy_color_type,
                ray_class=type(current_ray),
            )
            current_from_pool = True
        else:
            next_ray = type(current_ray)(
                start, new_dir,
                energy=current_ray.energy, current_n=current_n,
                color=current_ray.color, wavelength=current_ray.wavelength,
                energy_color_type=current_ray.energy_color_type,
            )
            current_from_pool = False
        current_ray = next_ray

    if current_from_pool and pool:
        pool.release(current_ray)

    return segments
