from typing import Optional

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as R

from core.rays import Ray
from utils.geometry import get_tangents
from fast_math import aspheric_intersect_numba
from surfaces.base import BaseSurface


class AsphericSurface(BaseSurface):
    def __init__(self, center, radius, conic_constant=0.0, aspheric_coeffs=None,
                 rotation_degrees=(0,0,0), n_inside=1.0,
                 edge_radius=None, thickness=0.0,
                 reflection_range=None, refraction_range=None,
                 absorption_range=None,
                 lens_origin=None, lens_axis=None):
        self.center = np.array(center, dtype=float)
        self.radius = radius
        self.k = conic_constant
        self.aspheric_coeffs = aspheric_coeffs if aspheric_coeffs else []

        base_axis = np.array([1.0, 0.0, 0.0])
        rot = R.from_euler('xyz', rotation_degrees, degrees=True).as_matrix()
        default_axis = rot @ base_axis

        self.lens_origin = np.array(lens_origin, dtype=float) if lens_origin is not None else self.center.copy()
        self.lens_axis = np.array(lens_axis, dtype=float) if lens_axis is not None else default_axis.copy()
        self.lens_axis /= np.linalg.norm(self.lens_axis)

        self.n = n_inside
        self.edge_radius = edge_radius if edge_radius is not None else 0.0
        self.thickness = thickness
        self.reflection_range = reflection_range
        self.refraction_range = refraction_range
        self.absorption_range = absorption_range

        self._t1, self._t2 = get_tangents(self.lens_axis)
        self._rot_local_to_world = np.column_stack([self.lens_axis, self._t1, self._t2])
        self._rot_world_to_local = self._rot_local_to_world.T

    def _world_to_local(self, point):
        return self._rot_world_to_local @ (point - self.lens_origin)

    def _local_to_world(self, local_point):
        return self.lens_origin + self._rot_local_to_world @ local_point

    def sag(self, r):
        c = 1.0 / self.radius if self.radius != 0 else 0.0
        if abs(c) < 1e-12:
            sag0 = np.zeros_like(r)
        else:
            discr = 1.0 - (1.0 + self.k) * c**2 * r**2
            safe_discr = np.maximum(0.0, discr)
            sag0 = np.where(discr >= 0,
                            (c * r**2) / (1.0 + np.sqrt(safe_discr)),
                            np.inf)
        sag_asp = np.zeros_like(r)
        for i, A in enumerate(self.aspheric_coeffs):
            sag_asp += A * r**(2 * (i + 1))
        return sag0 + sag_asp

    def sag_derivative(self, r):
        c = 1.0 / self.radius if self.radius != 0 else 0.0
        if abs(c) < 1e-12:
            dsag0 = np.zeros_like(r)
        else:
            discr = 1.0 - (1.0 + self.k) * c**2 * r**2
            safe_discr = np.maximum(0.0, discr)
            dsag0 = np.where(discr >= 0,
                             (c * r) / np.sqrt(safe_discr),
                             0.0)
        dsag_asp = np.zeros_like(r)
        for i, A in enumerate(self.aspheric_coeffs):
            power = 2 * (i + 1)
            dsag_asp += A * power * r**(power - 1)
        return dsag0 + dsag_asp

    def intersect(self, ray: Ray) -> Optional[float]:
        # Переводим луч в локальные координаты
        origin_loc = self._world_to_local(ray.origin)
        dir_loc = self._rot_world_to_local @ ray.direction
        # Нормализуем направление на всякий случай
        dir_loc = dir_loc / np.linalg.norm(dir_loc)

        sag_max = self.sag(self.edge_radius) if self.edge_radius > 0 else 0.0

        # Проверка: если луч почти параллелен плоскости и не попадает в апертуру
        # (быстрый отсев для производительности)
        if abs(dir_loc[0]) < 1e-9:
            # Луч перпендикулярен оси, может пересечь только край, но это редко
            # Пропускаем, чтобы не застревать
            pass

        t = aspheric_intersect_numba(
            origin_loc.astype(np.float64),
            dir_loc.astype(np.float64),
            self.radius,
            self.k,
            np.array(self.aspheric_coeffs, dtype=np.float64),
            self.edge_radius,
            sag_max
        )
        if t < 0.0:
            return None
        return float(t)

    def get_normal(self, point: np.ndarray) -> np.ndarray:
        p_loc = self._world_to_local(point)
        r = np.sqrt(p_loc[1]**2 + p_loc[2]**2)
        if r < 1e-12:
            normal_loc = np.array([1.0, 0.0, 0.0])
        else:
            dsag = self.sag_derivative(r)
            normal_loc = np.array([1.0, -dsag * p_loc[1] / r, -dsag * p_loc[2] / r])
        normal_loc /= np.linalg.norm(normal_loc)
        return self._rot_local_to_world @ normal_loc

    def get_mesh(self, n_radial=40, n_azimuth=80):
        rs = np.linspace(0, self.edge_radius, n_radial)
        phis = np.linspace(0, 2*np.pi, n_azimuth)
        r_grid, phi_grid = np.meshgrid(rs, phis)
        y_loc = r_grid * np.cos(phi_grid)
        z_loc = r_grid * np.sin(phi_grid)
        r = np.sqrt(y_loc**2 + z_loc**2)
        x_loc = self.sag(r)
        x_loc = np.where(np.isfinite(x_loc), x_loc, 0.0)
        grid = pv.StructuredGrid(x_loc, y_loc, z_loc)
        poly = grid.extract_surface(algorithm='dataset_surface')
        matrix = np.eye(4)
        matrix[:3, :3] = self._rot_local_to_world
        matrix[:3, 3] = self.lens_origin
        poly.transform(matrix, inplace=True)
        return poly

    def rotate(self, angles_deg):
        rot = R.from_euler('xyz', angles_deg, degrees=True).as_matrix()
        v = self.center - self.lens_origin
        self.center = self.lens_origin + rot @ v
        self.lens_axis = rot @ self.lens_axis
        self._t1, self._t2 = get_tangents(self.lens_axis)
        self._rot_local_to_world = np.column_stack([self.lens_axis, self._t1, self._t2])
        self._rot_world_to_local = self._rot_local_to_world.T

    def translate(self, vec):
        self.center += np.asarray(vec)
        self.lens_origin += np.asarray(vec)
