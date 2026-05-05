# fast_math.py
import numpy as np
from numba import njit, f8, i8

# ---------- Базовые векторные операции ----------
@njit
def normalize(v):
    n = np.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    if n > 1e-12:
        return v / n
    return v

@njit
def dot(v1, v2):
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]

@njit
def cross(v1, v2):
    return np.array([v1[1]*v2[2] - v1[2]*v2[1],
                     v1[2]*v2[0] - v1[0]*v2[2],
                     v1[0]*v2[1] - v1[1]*v2[0]])

# ---------- Френель ----------
@njit
def fresnel_coeffs(n1, n2, cos_i):
    sin2_i = max(0.0, 1.0 - cos_i * cos_i)
    eta = n1 / n2
    sin2_t = eta * eta * sin2_i
    if sin2_t > 1.0:
        return 1.0, 0.0  # R, T (энергетические)
    cos_t = np.sqrt(1.0 - sin2_t)
    r_s = (n1*cos_i - n2*cos_t) / (n1*cos_i + n2*cos_t)
    r_p = (n2*cos_i - n1*cos_t) / (n2*cos_i + n1*cos_t)
    R = 0.5 * (r_s*r_s + r_p*r_p)
    T = 1.0 - R
    return R, T

@njit
def fresnel_amplitudes(n1, n2, cos_i):
    sin2_i = max(0.0, 1.0 - cos_i*cos_i)
    eta = n1 / n2
    sin2_t = eta*eta * sin2_i
    if sin2_t > 1.0:
        return 1.0+0j, 1.0+0j, 0.0+0j, 0.0+0j
    cos_t = np.sqrt(1.0 - sin2_t)
    r_s = (n1*cos_i - n2*cos_t) / (n1*cos_i + n2*cos_t)
    r_p = (n2*cos_i - n1*cos_t) / (n2*cos_i + n1*cos_t)
    t_s = 2.0 * n1 * cos_i / (n1*cos_i + n2*cos_t)
    t_p = 2.0 * n1 * cos_i / (n2*cos_i + n1*cos_t)
    return complex(r_s), complex(r_p), complex(t_s), complex(t_p)

# ---------- Преломление (Снеллиус) ----------
@njit
def refract(ray_dir, normal, n1, n2):
    eta = n1 / n2
    cos_i = dot(normal, ray_dir)
    actual_normal = normal.copy()
    if cos_i > 0.0:
        actual_normal = -actual_normal
        cos_i = dot(actual_normal, ray_dir)
    cos_i = -cos_i
    sin2_t = eta*eta * (1.0 - cos_i*cos_i)
    if sin2_t > 1.0:
        # Полное внутреннее отражение
        return ray_dir - 2.0 * dot(ray_dir, actual_normal) * actual_normal
    cos_t = np.sqrt(max(0.0, 1.0 - sin2_t))
    return eta * ray_dir + (eta * cos_i - cos_t) * actual_normal

# ---------- Пересечение с плоскостью ----------
@njit
def plane_intersect(ray_origin, ray_dir, point, normal, lens_origin, lens_axis,
                    edge_radius, half_sizes, tangents, use_rectangular):
    dot_dn = dot(ray_dir, normal)
    if abs(dot_dn) < 1e-6:
        return -1.0
    t = dot(point - ray_origin, normal) / dot_dn
    if t <= 1e-6:
        return -1.0
    p = ray_origin + t * ray_dir

    if use_rectangular:
        # tangents – массив 2x3
        u = dot(p - lens_origin, tangents[0])
        v = dot(p - lens_origin, tangents[1])
        if abs(u) <= half_sizes[0] + 1e-6 and abs(v) <= half_sizes[1] + 1e-6:
            return t
        return -1.0
    else:
        vec = p - lens_origin
        proj = dot(vec, lens_axis)
        dist = np.sqrt(dot(vec, vec) - proj*proj)
        if dist <= edge_radius + 1e-6:
            return t
        return -1.0

# ---------- Пересечение со сферой ----------
@njit
def sphere_intersect(ray_origin, ray_dir, center, radius, lens_origin, lens_axis,
                     edge_radius, thickness):
    oc = ray_origin - center
    a = dot(ray_dir, ray_dir)
    b = 2.0 * dot(oc, ray_dir)
    c = dot(oc, oc) - radius*radius
    disc = b*b - 4.0*a*c
    if disc < 0.0:
        return -1.0
    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2.0*a)
    t2 = (-b + sqrt_disc) / (2.0*a)
    best_t = -1.0
    for t in (t1, t2):
        if t <= 1e-6:
            continue
        p = ray_origin + t * ray_dir
        vec = p - lens_origin
        proj = dot(vec, lens_axis)
        dist = np.sqrt(dot(vec, vec) - proj*proj)
        if dist <= edge_radius + 1e-6 and abs(proj) <= thickness/2 + 5.0:
            if best_t < 0.0 or t < best_t:
                best_t = t
    return best_t
