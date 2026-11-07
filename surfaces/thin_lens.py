from typing import Optional

import numpy as np
import pyvista as pv

from core.rays import Ray
from utils.geometry import get_tangents
from surfaces.base import BaseSurface


class ThinLens(BaseSurface):
    """Тонкая линза (параксиальное приближение)."""
    def __init__(self, center, focal_length, edge_radius=3.0,
                 axis_dir=np.array([1.0, 0.0, 0.0]),
                 refraction_range=(0, np.inf)):
        self.center = np.asarray(center, dtype=float)
        self.f = focal_length          # положительное – собирающая, отрицательное – рассеивающая
        self.edge_radius = edge_radius
        self.axis_dir = np.asarray(axis_dir, dtype=float)
        self.axis_dir /= np.linalg.norm(self.axis_dir)
        self.refraction_range = refraction_range
        self.n = 1.0  # заглушка
        self._t1, self._t2 = get_tangents(self.axis_dir)

    def intersect(self, ray: Ray) -> Optional[float]:
        # Плоскость, проходящая через center с нормалью axis_dir
        dot_dir = np.dot(ray.direction, self.axis_dir)
        if abs(dot_dir) < 1e-6:
            return None
        t = np.dot(self.center - ray.origin, self.axis_dir) / dot_dir
        if t <= 1e-6:
            return None
        hit = ray.origin + ray.direction * t
        # Проверка круглой апертуры
        r_vec = (hit - self.center) - self.axis_dir * np.dot(hit - self.center, self.axis_dir)
        if np.linalg.norm(r_vec) > self.edge_radius + 1e-6:
            return None
        return t

    def thin_lens_deflection(self, ray_dir, hit_point):
        """Изменение направления луча в параксиальном приближении."""
        r_vec = (hit_point - self.center) - self.axis_dir * np.dot(hit_point - self.center, self.axis_dir)
        h = np.linalg.norm(r_vec)
        if h < 1e-12:
            return ray_dir       # на оси – без отклонения
        r_unit = r_vec / h
        # Отклонение луча: delta = (h/f) * r_unit (знак уже учтён)
        new_dir = ray_dir - (h / self.f) * r_unit
        new_dir /= np.linalg.norm(new_dir)
        return new_dir

    def get_mesh(self) -> pv.PolyData:
        disc = pv.Disc(center=self.center, normal=self.axis_dir,
                       inner=0, outer=self.edge_radius, c_res=64)
        return disc

    def is_active(self, wavelength):
        return self.refraction_range is not None
