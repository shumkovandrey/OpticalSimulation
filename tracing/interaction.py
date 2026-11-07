"""Френелевское расщепление луча на отражённый и преломлённый."""

from typing import List, Optional

import numpy as np

from constants import DEFAULT_OFFSET_DISTANCE
from core.rays import Ray, RayPool
from fast_math import fresnel_amplitudes


def split_ray(ray: Ray, normal: np.ndarray, n_next: float, start_point: np.ndarray,
              allow_reflection: bool = True, allow_refraction: bool = True,
              offset_distance: float = DEFAULT_OFFSET_DISTANCE,
              pool: Optional[RayPool] = None) -> List[Ray]:
    """Расщепляет луч на отражённый и преломлённый по формулам Френеля.

    Полностью эквивалентна split_ray из main.py (шаг 5, без поляризации).
    """
    EPS = offset_distance

    cos_i = np.dot(normal, ray.direction)
    if cos_i > 0:
        normal = -normal
        cos_i = np.dot(normal, ray.direction)
    cos_i = -cos_i

    n1, n2 = ray.current_n, n_next

    r_s, r_p, t_s, t_p = fresnel_amplitudes(n1, n2, cos_i)

    R_s = abs(r_s) ** 2
    R_p = abs(r_p) ** 2
    R_coeff = 0.5 * (R_s + R_p)

    eta = n1 / n2
    sin2_t = (eta ** 2) * (1.0 - cos_i ** 2)
    is_tir = (sin2_t >= 1.0)

    if is_tir:
        R_coeff = 1.0
        T_coeff = 0.0
        cos_t = 0.0
        allow_refraction = False
    else:
        cos_t = np.sqrt(1.0 - sin2_t)
        T_coeff = 1.0 - R_coeff

    if not allow_reflection and not allow_refraction:
        return []
    if allow_reflection and not allow_refraction:
        R_coeff, T_coeff = 1.0, 0.0
    elif not allow_reflection and allow_refraction:
        R_coeff, T_coeff = 0.0, 1.0

    reflected_dir = ray.direction - 2 * np.dot(ray.direction, normal) * normal
    reflected_dir /= np.linalg.norm(reflected_dir)

    if not is_tir:
        refracted_dir = eta * ray.direction + (eta * cos_i - cos_t) * normal
        refracted_dir /= np.linalg.norm(refracted_dir)
    else:
        refracted_dir = reflected_dir.copy()

    new_rays = []

    # 1. ОТРАЖЁННЫЙ
    if allow_reflection and R_coeff > 1e-6:
        reflected_energy = ray.energy * R_coeff
        if reflected_energy > 1e-6:
            if pool:
                new_ray = pool.acquire(
                    origin=start_point + EPS * reflected_dir,
                    direction=reflected_dir,
                    energy=reflected_energy,
                    current_n=n1,
                    color=ray.color,
                    energy_color_type=ray.energy_color_type,
                    wavelength=ray.wavelength,
                )
            else:
                new_ray = type(ray)(
                    origin=start_point + EPS * reflected_dir,
                    direction=reflected_dir,
                    energy=reflected_energy,
                    current_n=n1,
                    color=ray.color,
                    wavelength=ray.wavelength,
                    energy_color_type=ray.energy_color_type,
                )
            new_rays.append(new_ray)

    # 2. ПРЕЛОМЛЁННЫЙ
    if allow_refraction and T_coeff > 1e-6:
        refracted_energy = ray.energy * T_coeff
        if refracted_energy > 1e-6:
            if pool:
                new_ray = pool.acquire(
                    origin=start_point + EPS * refracted_dir,
                    direction=refracted_dir,
                    energy=refracted_energy,
                    current_n=n2,
                    color=ray.color,
                    energy_color_type=ray.energy_color_type,
                    wavelength=ray.wavelength,
                    ray_class=type(ray),
                )
            else:
                new_ray = type(ray)(
                    origin=start_point + EPS * refracted_dir,
                    direction=refracted_dir,
                    energy=refracted_energy,
                    current_n=n2,
                    color=ray.color,
                    wavelength=ray.wavelength,
                    energy_color_type=ray.energy_color_type,
                )
            new_rays.append(new_ray)

    return new_rays
