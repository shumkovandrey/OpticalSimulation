"""Однолучевая последовательная трассировка (SimpleMode)."""

from typing import List, Optional

import numpy as np

from constants import DEFAULT_OFFSET_DISTANCE, RAY_INFINITY_DISTANCE
from core.dispersion import get_dispersion_n
from core.rays import DispersiveRay, Ray, RayPool, spawn_ray
from core.segments import Segment
from fast_math import refract
from tracing.hit_finder import find_best_hit


def trace_simple(
    ray: "Ray",
    elements: List,
    max_bounces: int = 50,
    offset_distance: float = DEFAULT_OFFSET_DISTANCE,
    prioritize_refraction: bool = True,
    pool: Optional["RayPool"] = None,
    ambient_n: float = 1.0,
) -> List[Segment]:
    segments = []

    current_ray = ray
    current_n = ray.current_n
    current_from_pool = False

    for _ in range(max_bounces):
        hit = find_best_hit(current_ray, elements)

        if hit is None:
            end_point = current_ray.origin + current_ray.direction * RAY_INFINITY_DISTANCE
            segments.append(
                Segment(
                    current_ray.origin.copy(),
                    end_point.copy(),
                    current_ray.energy,
                    current_ray.color,
                )
            )
            break

        segments.append(
            Segment(
                current_ray.origin.copy(),
                hit.point.copy(),
                current_ray.energy,
                current_ray.color,
            )
        )

        if hit.absorbed:
            break

        if hit.is_thin_lens:
            new_dir = hit.obj.thin_lens_deflection(current_ray.direction, hit.point)
            start = hit.point + offset_distance * new_dir

            if pool and current_from_pool:
                pool.release(current_ray)
            next_ray = spawn_ray(
                current_ray,
                pool,
                origin=start,
                direction=new_dir,
                current_n=current_n,
            )
            current_from_pool = pool is not None

            current_ray = next_ray
            continue

        allow_reflection = hit.allow_reflection
        allow_refraction = hit.allow_refraction

        if prioritize_refraction and allow_refraction:
            allow_reflection = False

        if not allow_reflection and not allow_refraction:
            start = hit.point + offset_distance * current_ray.direction

            if pool and current_from_pool:
                pool.release(current_ray)
            next_ray = spawn_ray(
                current_ray,
                pool,
                origin=start,
                direction=current_ray.direction,
                current_n=current_n,
            )
            current_from_pool = pool is not None

            current_ray = next_ray
            continue

        if allow_refraction:
            resolved_n_inside = get_dispersion_n(hit.material, current_ray)
            if abs(current_n - resolved_n_inside) < 1e-6:
                n_next = ambient_n
            else:
                n_next = resolved_n_inside

            normal_tir = hit.normal
            if np.dot(normal_tir, current_ray.direction) > 0:
                normal_tir = -normal_tir
            cos_i_tir = -np.dot(normal_tir, current_ray.direction)
            eta = current_n / n_next
            sin2_t = eta * eta * (1.0 - cos_i_tir * cos_i_tir)

            if sin2_t >= 1.0:
                allow_reflection = True
                allow_refraction = False
            else:
                refracted_dir = refract(current_ray.direction, hit.normal, current_n, n_next)
                current_n = n_next
                new_dir = refracted_dir
                allow_reflection = False

        if allow_reflection:
            normal = hit.normal
            if np.dot(normal, current_ray.direction) > 0:
                normal = -normal
            new_dir = current_ray.direction - 2 * np.dot(current_ray.direction, normal) * normal
            new_dir /= np.linalg.norm(new_dir)

        start = hit.point + offset_distance * new_dir

        if pool and current_from_pool:
            pool.release(current_ray)
        next_ray = spawn_ray(
            current_ray,
            pool,
            origin=start,
            direction=new_dir,
            current_n=current_n,
        )
        current_from_pool = pool is not None

        current_ray = next_ray

    if current_from_pool and pool:
        pool.release(current_ray)

    return segments
