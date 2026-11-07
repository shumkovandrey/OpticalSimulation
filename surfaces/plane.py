from typing import Optional

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as R

from core.rays import Ray
from utils.geometry import get_tangents, calculate_rotation_matrix
from fast_math import plane_intersect_circular, plane_intersect_rectangular
from surfaces.base import BaseSurface


class PlaneSurface(BaseSurface):
    def __init__(self, point,
                 normal=None,
                 rotation_degrees=(0, 0, 0),
                 n_inside=1.0,
                 half_sizes=None, edge_radius=None,
                 reflection_range=None, refraction_range=None,
                 absorption_range=None,
                 lens_origin=None, lens_axis=None,
                 shape_type="circle", width=2.0, height=2.0):  # <-- Добавлены новые параметры
        self.point = np.array(point, dtype=float)

        self.base_normal = np.array([1.0, 0.0, 0.0])
        self.normal = self.base_normal.copy()

        self.base_t1, self.base_t2 = get_tangents(self.base_normal)
        self.face_tangents = (self.base_t1.copy(), self.base_t2.copy())

        self.n = n_inside
        self.edge_radius = edge_radius if edge_radius is not None else 1.5

        self.shape_type = shape_type
        self.width = float(width)
        self.height = float(height)
        self.half_sizes = np.array([self.width / 2.0, self.height / 2.0]) if half_sizes is None else np.array(
            half_sizes)

        self.reflection_range = reflection_range
        self.refraction_range = refraction_range
        self.absorption_range = absorption_range

        self.rotation_matrix = np.eye(3)

        if normal is not None:
            explicit_norm = np.array(normal, dtype=float)
            explicit_norm /= np.linalg.norm(explicit_norm)
            self.rotation_matrix = calculate_rotation_matrix(explicit_norm)
        else:
            self.rotation_matrix = R.from_euler('xyz', rotation_degrees, degrees=True).as_matrix()

        self._apply_current_rotation()

        self.lens_origin = np.array(lens_origin, dtype=float) if lens_origin is not None else self.point.copy()
        self.lens_axis = np.array(lens_axis, dtype=float) if lens_axis is not None else self.normal.copy()
        self.lens_axis /= np.linalg.norm(self.lens_axis)

    def _apply_current_rotation(self):
        """Обновляет мировые векторы нормали и тангенсов на основе накопленной матрицы поворота."""
        self.normal = self.rotation_matrix @ self.base_normal
        self.normal /= np.linalg.norm(self.normal)

        t1 = self.rotation_matrix @ self.base_t1
        t2 = self.rotation_matrix @ self.base_t2
        self.face_tangents = (t1 / np.linalg.norm(t1), t2 / np.linalg.norm(t2))

    def rotate(self, angles_deg):
        """ИСПРАВЛЕНИЕ: Перезаписываем абсолютную матрицу поворота из UI и синхронно вращаем все оси."""
        self.rotation_matrix = R.from_euler('xyz', angles_deg, degrees=True).as_matrix()
        self._apply_current_rotation()
        self.lens_axis = self.normal.copy()

    def translate(self, vec):
        self.point += np.asarray(vec)
        self.lens_origin += np.asarray(vec)

    def apply_transform(self, mat):
        """Применение аффинной матрицы 4x4."""
        R_part = mat[:3, :3]
        t_part = mat[:3, 3]
        self.point = R_part @ self.point + t_part
        self.lens_origin = R_part @ self.lens_origin + t_part

        # Обновляем матрицу поворота объекта
        self.rotation_matrix = R_part @ self.rotation_matrix
        self._apply_current_rotation()
        self.lens_axis = self.normal.copy()

    def intersect(self, ray: Ray) -> Optional[float]:
        if self.shape_type == "rectangle":
            tangents = np.array(self.face_tangents, dtype=np.float64)
            half = np.array([self.width / 2.0, self.height / 2.0], dtype=np.float64)
            t = plane_intersect_rectangular(
                ray.origin, ray.direction, self.point, self.normal,
                self.lens_origin, tangents, half,
            )
        else:
            t = plane_intersect_circular(
                ray.origin, ray.direction, self.point, self.normal,
                self.lens_origin, self.lens_axis, self.edge_radius,
            )
        if t < 0.0:
            return None
        return float(t)

    def get_normal(self, point: np.ndarray) -> np.ndarray:
        return self.normal

    def get_mesh(self) -> pv.PolyData:
        if self.shape_type == "rectangle":
            # Строим локальный прямоугольник в плоскости YZ (так как базовая нормаль)
            hu, hv = self.width / 2.0, self.height / 2.0
            vertices = np.array([
                [0.0, -hv, -hu],
                [0.0, hv, -hu],
                [0.0, hv, hu],
                [0.0, -hv, hu]
            ], dtype=np.float32)
            faces = np.array([4, 0, 1, 2, 3], dtype=np.int32)
            return pv.PolyData(vertices, faces)
        else:
            # Круглый диск в нуле с нормалью (1, 0, 0)
            radius = self.edge_radius if self.edge_radius else 1.0
            return pv.Disc(center=(0, 0, 0), normal=(1, 0, 0), inner=0, outer=radius, c_res=64)
