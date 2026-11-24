import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as R

from core.materials import Material
from surfaces.base import LensAssembly
from surfaces.cylinder import CylinderSurface
from surfaces.plane import PlaneSurface
from surfaces.sphere import SphereSurface


class UniversalLens(LensAssembly):
    """
    Двояковыпуклая/вогнутая/мениск линза в произвольной ориентации.
    Параметры:
        origin      – геометрический центр линзы,
        rotation_degrees – углы Эйлера поворота линзы,
        R1, R2      – радиусы кривизны передней и задней поверхностей (None = плоскость),
        thickness   – толщина вдоль оси по центру,
        edge_radius – радиус апертуры,
        n           – показатель преломления материала.
    """

    def __init__(
        self,
        origin,
        rotation_degrees=(0, 0, 0),
        R1=None,
        R2=None,
        thickness=2.0,
        edge_radius=3.0,
        material=None,
    ):
        self.origin = np.array(origin, dtype=float)
        self.rotation_degrees = rotation_degrees
        self.rotation = R.from_euler("xyz", rotation_degrees, degrees=True).as_matrix()
        self.axis_dir = self.rotation @ np.array([1.0, 0.0, 0.0])
        self.thickness = thickness
        self.edge_radius = edge_radius
        self.material = material if material is not None else Material(n=1.0)
        self.R1, self.R2 = R1, R2

        self._create_surfaces()

    def _create_surfaces(self):
        r = float(self.edge_radius)

        sag1 = 0.0
        if self.R1 is not None:
            effective_R1 = max(abs(self.R1), r + 1e-5)
            sag1 = effective_R1 - np.sqrt(effective_R1**2 - r**2)
            if self.R1 < 0:
                sag1 = -sag1

        sag2 = 0.0
        if self.R2 is not None:
            effective_R2 = max(abs(self.R2), r + 1e-5)
            sag2 = effective_R2 - np.sqrt(effective_R2**2 - r**2)
            if self.R2 < 0:
                sag2 = -sag2

        self.internal_thickness = self.thickness

        half = self.internal_thickness / 2
        edge_thickness = (
            self.internal_thickness - (sag1 if self.R1 is not None else 0.0) - (sag2 if self.R2 is not None else 0.0)
        )
        half_rim_length = edge_thickness / 2.0

        rim_center_x = (sag1 - sag2) / 2.0 if (self.R1 is not None and self.R2 is not None) else 0.0
        local_cylinder_center = np.array([rim_center_x, 0.0, 0.0])

        mat = np.eye(4)
        mat[:3, :3] = self.rotation
        mat[:3, 3] = self.origin

        if self.R1 is None:
            self.front = PlaneSurface(
                point=[-half, 0, 0],
                normal=[-1.0, 0.0, 0.0],
                material=self.material,
                edge_radius=self.edge_radius,
            )
        else:
            self.front = SphereSurface(
                radius=self.R1,
                material=self.material,
                edge_radius=self.edge_radius,
                thickness=self.internal_thickness,
                lens_origin=[-half, 0, 0],
                lens_axis=[-1.0, 0.0, 0.0],
            )

        if self.R2 is None:
            self.back = PlaneSurface(
                point=[half, 0, 0],
                normal=[1.0, 0.0, 0.0],
                material=self.material,
                edge_radius=self.edge_radius,
            )
        else:
            self.back = SphereSurface(
                radius=self.R2,
                material=self.material,
                edge_radius=self.edge_radius,
                thickness=self.internal_thickness,
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

    def get_mesh(self) -> pv.PolyData:
        """Обновленный метод генерации полигонального меша с учетом измененной внутренней толщины."""
        rs = np.linspace(0, self.edge_radius, 30)
        phis = np.linspace(0, 2 * np.pi, 60)
        r_grid, phi_grid = np.meshgrid(rs, phis)

        y = r_grid * np.cos(phi_grid)
        z = r_grid * np.sin(phi_grid)

        v1_local = -self.internal_thickness / 2
        v2_local = self.internal_thickness / 2

        def get_local_x(R, v_x, r_vals, is_front):
            if R is not None:
                c_x = v_x + R if is_front else v_x - R
                dx = np.sqrt(np.maximum(0, abs(R) ** 2 - r_vals**2))
                return c_x - dx if (is_front and R > 0) or (not is_front and R < 0) else c_x + dx
            else:
                return np.full_like(r_vals, v_x)

        x_front = get_local_x(self.R1, v1_local, r_grid, True)
        x_back = get_local_x(self.R2, v2_local, r_grid, False)

        front_mesh = pv.StructuredGrid(x_front, y, z).extract_surface(algorithm="dataset_surface")
        back_mesh = pv.StructuredGrid(x_back, y, z).extract_surface(algorithm="dataset_surface")

        rim_x = np.array([x_front[:, -1], x_back[:, -1]])
        rim_y = np.array([y[:, -1], y[:, -1]])
        rim_z = np.array([z[:, -1], z[:, -1]])
        rim_mesh = pv.StructuredGrid(rim_x, rim_y, rim_z).extract_surface(algorithm="dataset_surface")

        local_mesh = front_mesh.merge(back_mesh).merge(rim_mesh)

        return local_mesh

    def rotate(self, angles_deg):
        rot = R.from_euler("xyz", angles_deg, degrees=True).as_matrix()
        self.axis_dir = rot @ self.axis_dir
        self.axis_dir /= np.linalg.norm(self.axis_dir)
        self.rotation = rot @ self.rotation
        self._create_surfaces()

    def translate(self, vec):
        self.origin += np.asarray(vec)
        self._create_surfaces()
