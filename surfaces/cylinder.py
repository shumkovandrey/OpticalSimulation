from typing import Optional

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as R

from core.rays import Ray
from fast_math import cylinder_intersect
from surfaces.plane import PlaneSurface
from surfaces.base import BaseSurface


class CylinderSurface(BaseSurface):
    """Боковая цилиндрическая поверхность (ободок линзы)."""

    def __init__(self, center, axis_dir, radius, half_length,
                 n_inside=1.0,
                 reflection_range=None, refraction_range=None,
                 absorption_range=None,
                 capping=False):
        self.center = np.array(center, dtype=float)
        self.axis_dir = np.array(axis_dir, dtype=float)
        self.axis_dir /= np.linalg.norm(self.axis_dir)
        self.radius = radius
        self.half_length = half_length
        self.n = n_inside
        self.reflection_range = reflection_range
        self.refraction_range = refraction_range
        self.absorption_range = absorption_range

        self.capping = capping
        self._update_caps()

    def _update_caps(self):
        """Создает внутренние плоскости для торцов, если включен capping."""
        if self.capping:
            p1 = self.center + self.axis_dir * self.half_length
            p2 = self.center - self.axis_dir * self.half_length
            # Используем PlaneSurface в режиме круга ("circle")
            self.cap1 = PlaneSurface(point=p1, normal=self.axis_dir, n_inside=self.n, edge_radius=self.radius,
                                     reflection_range=self.reflection_range, refraction_range=self.refraction_range,
                                     absorption_range=self.absorption_range)
            self.cap2 = PlaneSurface(point=p2, normal=-self.axis_dir, n_inside=self.n, edge_radius=self.radius,
                                     reflection_range=self.reflection_range, refraction_range=self.refraction_range,
                                     absorption_range=self.absorption_range)
        else:
            self.cap1 = None
            self.cap2 = None

    def intersect(self, ray: Ray) -> Optional[float]:
        # Проверяем боковую поверхность
        t_side = cylinder_intersect(ray.origin, ray.direction,
                                    self.center, self.axis_dir,
                                    self.radius, self.half_length)
        best_t = float('inf') if t_side < 0.0 else float(t_side)

        # НАЧАЛО ИЗМЕНЕНИЙ: Математика пересечения с крышками
        if self.capping and self.cap1 and self.cap2:
            t1 = self.cap1.intersect(ray)
            t2 = self.cap2.intersect(ray)
            if t1 is not None and t1 < best_t: best_t = t1
            if t2 is not None and t2 < best_t: best_t = t2
        # КОНЕЦ ИЗМЕНЕНИЙ

        if best_t == float('inf'):
            return None
        return best_t

    def get_normal(self, point):
        # НАЧАЛО ИЗМЕНЕНИЙ: Определение нормали на крышках
        if self.capping:
            # Проверяем, близко ли точка к плоскостям торцов
            vec = point - self.center
            proj = np.dot(vec, self.axis_dir)
            if abs(proj - self.half_length) < 1e-4:
                return self.axis_dir
            elif abs(proj + self.half_length) < 1e-4:
                return -self.axis_dir
        # КОНЕЦ ИЗМЕНЕНИЙ

        vec = point - self.center
        proj = np.dot(vec, self.axis_dir)
        closest_on_axis = self.center + proj * self.axis_dir
        radial = point - closest_on_axis
        norm = np.linalg.norm(radial)
        if norm < 1e-12:
            return self.axis_dir
        return radial / norm

    def get_mesh(self) -> pv.PolyData:
        # Строим дефолтный цилиндр вдоль оси X (1, 0, 0) в нуле координат
        cylinder = pv.Cylinder(center=(0, 0, 0), direction=(1, 0, 0),
                               radius=self.radius, height=2 * self.half_length,
                               capping=self.capping, resolution=64)
        return cylinder

    def rotate(self, angles_deg):
        rot = R.from_euler('xyz', angles_deg, degrees=True).as_matrix()
        self.center = rot @ self.center
        self.axis_dir = rot @ self.axis_dir
        self.axis_dir /= np.linalg.norm(self.axis_dir)
        if self.capping: self._update_caps() # Пересчитываем крышки

    def translate(self, vec):
        self.center += np.asarray(vec)
        if self.capping: self._update_caps() # Пересчитываем крышки
