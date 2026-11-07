import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as R

from utils.geometry import get_tangents
from surfaces.base import LensAssembly
from surfaces.plane import PlaneSurface
from surfaces.aspheric import AsphericSurface
from surfaces.cylinder import CylinderSurface


class HyperbolicLens(LensAssembly):
    """
    Плоско-гиперболическая (асферическая) линза в произвольной ориентации.
    Параметры:
        n - показатель преломления
        f_target - фокусное расстояние
    Коническая константа k рассчитывается автоматически на основе геометрических параметров.
    """

    def __init__(self, origin, rotation_degrees=(0, 0, 0),
                 thickness=2.0, edge_radius=3.0, n=1.5, f_target=10.0,
                 reflection_range=None, refraction_range=(0, np.inf),
                 absorption_range=None):
        self.origin = np.array(origin, dtype=float)
        self.rotation = R.from_euler('xyz', rotation_degrees, degrees=True).as_matrix()
        self.axis_dir = self.rotation @ np.array([1.0, 0.0, 0.0])
        self.thickness = thickness
        self.edge_radius = edge_radius
        self.n = n
        self.f_target = float(f_target)

        # ПОЛНЫЙ АВТОРАСЧЕТ: Радиус вершины и коническая константа строго по законам оптики
        self.R_curvature = -float(self.f_target * (self.n - 1.0))
        self.k = -(self.n ** 2)

        self.aspheric_coeffs = []
        self.reflection_range = reflection_range
        self.refraction_range = refraction_range
        self.absorption_range = absorption_range
        self._last_hit_surface = None

        self._create_surfaces()

    def _create_surfaces(self):
        r = float(self.edge_radius)
        half = self.thickness / 2.0

        # Вычисляем стрелку прогиба (sagitta) для края гиперболы
        c = 1.0 / self.R_curvature if self.R_curvature != 0 else 0.0

        # Синхронизировано с логикой ядра Numba из main.py
        discr = 1.0 - (1.0 + self.k) * (c ** 2) * (r ** 2)
        if discr >= 0:
            sag1 = (c * r ** 2) / (1.0 + np.sqrt(discr))
        else:
            sag1 = (c * r ** 2) / 2.0

        # Длина бокового цилиндрического ободка с учётом знака профиля (v1_local - sag1)
        edge_thickness = self.thickness + sag1
        half_rim_length = edge_thickness / 2.0

        # Смещение центра ободка влево (в сторону минуса по X)
        rim_center_x = -sag1 / 2.0
        local_cylinder_center = np.array([rim_center_x, 0.0, 0.0])

        mat = np.eye(4)
        mat[:3, :3] = self.rotation
        mat[:3, 3] = self.origin

        # 1. Передняя поверхность — Асферическая (гиперболоид)
        self.front = AsphericSurface(
            center=[-half, 0, 0], radius=self.R_curvature, conic_constant=self.k,
            aspheric_coeffs=self.aspheric_coeffs, n_inside=self.n, edge_radius=self.edge_radius,
            thickness=self.thickness, reflection_range=self.reflection_range,
            refraction_range=self.refraction_range, absorption_range=self.absorption_range,
            lens_origin=[-half, 0, 0], lens_axis=[-1.0, 0.0, 0.0]
        )

        # 2. Задняя поверхность — Плоская
        self.back = PlaneSurface(
            point=[half, 0, 0], normal=[1.0, 0.0, 0.0],
            n_inside=self.n, edge_radius=self.edge_radius,
            reflection_range=self.reflection_range, refraction_range=self.refraction_range,
            absorption_range=self.absorption_range,
            lens_origin=[half, 0, 0], lens_axis=[1.0, 0.0, 0.0]
        )

        # 3. Боковая поверхность (цилиндрический ободок)
        self.cylinder = CylinderSurface(
            center=local_cylinder_center, axis_dir=[1.0, 0.0, 0.0], radius=self.edge_radius,
            half_length=half_rim_length, n_inside=self.n, reflection_range=self.reflection_range,
            refraction_range=self.refraction_range, absorption_range=self.absorption_range
        )

        # Синхронно позиционируем все поверхности по матрице
        self.front.apply_transform(mat)
        self.back.apply_transform(mat)

        self.cylinder.center = mat[:3, :3] @ self.cylinder.center + mat[:3, 3]
        self.cylinder.axis_dir = mat[:3, :3] @ self.cylinder.axis_dir

    def rotate(self, angles_deg):
        rot = R.from_euler('xyz', angles_deg, degrees=True).as_matrix()
        self.axis_dir = rot @ self.axis_dir
        self.axis_dir /= np.linalg.norm(self.axis_dir)
        self.rotation = rot @ self.rotation
        self._create_surfaces()

    def translate(self, vec):
        self.origin += np.asarray(vec)
        self._create_surfaces()

    def get_mesh(self) -> pv.PolyData:
        rs = np.linspace(0, self.edge_radius, 30)
        phis = np.linspace(0, 2 * np.pi, 60)
        r_grid, phi_grid = np.meshgrid(rs, phis)

        y = r_grid * np.cos(phi_grid)
        z = r_grid * np.sin(phi_grid)

        v1_local = -self.thickness / 2.0
        v2_local = self.thickness / 2.0

        r_flat = r_grid.flatten()
        sag_values = self.front.sag(r_flat).reshape(r_grid.shape)

        # Исправленный вариант профиля меша
        x_front_local = v1_local - sag_values
        x_back_local = np.full_like(r_grid, v2_local)

        front_mesh = pv.StructuredGrid(x_front_local, y, z).extract_surface(algorithm='dataset_surface')
        back_mesh = pv.StructuredGrid(x_back_local, y, z).extract_surface(algorithm='dataset_surface')

        rim_x = np.array([x_front_local[:, -1], x_back_local[:, -1]])
        rim_y = np.array([y[:, -1], y[:, -1]])
        rim_z = np.array([z[:, -1], z[:, -1]])
        rim_mesh = pv.StructuredGrid(rim_x, rim_y, rim_z).extract_surface(algorithm='dataset_surface')

        local_mesh = front_mesh.merge(back_mesh).merge(rim_mesh)

        return local_mesh
