import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as R

from core.materials import Material
from surfaces.aspheric import AsphericSurface
from surfaces.base import LensAssembly
from surfaces.cylinder import CylinderSurface
from surfaces.plane import PlaneSurface
from utils.geometry import get_tangents


class HyperbolicLens(LensAssembly):
    """
    Плоско-гиперболическая (асферическая) линза в произвольной ориентации.
    Параметры:
        n - показатель преломления
        f_target - фокусное расстояние
    Коническая константа k рассчитывается автоматически на основе геометрических параметров.
    """

    def __init__(
        self,
        origin,
        rotation_degrees=(0, 0, 0),
        thickness=2.0,
        edge_radius=3.0,
        material=None,
        f_target=10.0,
    ):
        self.origin = np.array(origin, dtype=float)
        self.rotation = R.from_euler("xyz", rotation_degrees, degrees=True).as_matrix()
        self.axis_dir = self.rotation @ np.array([1.0, 0.0, 0.0])
        self.thickness = thickness
        self.edge_radius = edge_radius
        self.material = material if material is not None else Material(n=1.0)
        self.f_target = float(f_target)

        # Радиус вершины и коническая константа вычисляются автоматически
        self.R_curvature = -float(self.f_target * (self.material.n - 1.0))
        self.k = -(self.material.n**2)

        self.aspheric_coeffs = []
        self._last_hit_surface = None

        self._create_surfaces()

    def _create_surfaces(self):
        r = float(self.edge_radius)
        half = self.thickness / 2.0

        c = 1.0 / self.R_curvature if self.R_curvature != 0 else 0.0

        discr = 1.0 - (1.0 + self.k) * (c**2) * (r**2)
        if discr >= 0:
            sag1 = (c * r**2) / (1.0 + np.sqrt(discr))
        else:
            sag1 = (c * r**2) / 2.0

        edge_thickness = self.thickness + sag1
        half_rim_length = edge_thickness / 2.0

        rim_center_x = -sag1 / 2.0
        local_cylinder_center = np.array([rim_center_x, 0.0, 0.0])

        mat = np.eye(4)
        mat[:3, :3] = self.rotation
        mat[:3, 3] = self.origin

        self.front = AsphericSurface(
            center=[-half, 0, 0],
            radius=self.R_curvature,
            conic_constant=self.k,
            aspheric_coeffs=self.aspheric_coeffs,
            material=self.material,
            edge_radius=self.edge_radius,
            thickness=self.thickness,
            lens_origin=[-half, 0, 0],
            lens_axis=[-1.0, 0.0, 0.0],
        )

        self.back = PlaneSurface(
            point=[half, 0, 0],
            normal=[1.0, 0.0, 0.0],
            material=self.material,
            edge_radius=self.edge_radius,
            lens_origin=[half, 0, 0],
            lens_axis=[1.0, 0.0, 0.0],
        )

        self.cylinder = CylinderSurface(
            center=local_cylinder_center,
            axis_dir=[1.0, 0.0, 0.0],
            radius=self.edge_radius,
            half_length=half_rim_length,
            material=self.material,
        )

        self.front.apply_transform(mat)
        self.back.apply_transform(mat)

        self.cylinder.center = mat[:3, :3] @ self.cylinder.center + mat[:3, 3]
        self.cylinder.axis_dir = mat[:3, :3] @ self.cylinder.axis_dir

    def rotate(self, angles_deg):
        rot = R.from_euler("xyz", angles_deg, degrees=True).as_matrix()
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

        x_front_local = v1_local - sag_values
        x_back_local = np.full_like(r_grid, v2_local)

        front_mesh = pv.StructuredGrid(x_front_local, y, z).extract_surface(algorithm="dataset_surface")
        back_mesh = pv.StructuredGrid(x_back_local, y, z).extract_surface(algorithm="dataset_surface")

        rim_x = np.array([x_front_local[:, -1], x_back_local[:, -1]])
        rim_y = np.array([y[:, -1], y[:, -1]])
        rim_z = np.array([z[:, -1], z[:, -1]])
        rim_mesh = pv.StructuredGrid(rim_x, rim_y, rim_z).extract_surface(algorithm="dataset_surface")

        local_mesh = front_mesh.merge(back_mesh).merge(rim_mesh)

        return local_mesh
