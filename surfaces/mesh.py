from typing import Optional

import numpy as np
import pyvista as pv
import trimesh
from scipy.spatial.transform import Rotation as R

from core.materials import Material
from core.rays import Ray
from surfaces.base import BaseSurface


class MeshSurface(BaseSurface):
    """
    Произвольная треугольная поверхность.
    Математика пересечений работает в мировых координатах,
    а get_mesh() возвращает локальную модель для GPU-рендеринга через user_matrix.
    """

    def __init__(
        self,
        mesh,
        rotation_degrees=(0, 0, 0),
        translation=(0, 0, 0),
        material=None,
        scale_factors=(1.0, 1.0, 1.0),
    ):

        if isinstance(mesh, str):
            self.trimesh_obj = trimesh.load(mesh)
            if isinstance(self.trimesh_obj, trimesh.Scene):
                self.trimesh_obj = trimesh.util.concatenate(
                    [g for g in self.trimesh_obj.geometry.values() if isinstance(g, trimesh.Trimesh)]
                )
            if not isinstance(self.trimesh_obj, trimesh.Trimesh):
                raise TypeError("Файл не содержит треугольной сетки")
        elif isinstance(mesh, trimesh.Trimesh):
            self.trimesh_obj = mesh.copy()
        elif isinstance(mesh, pv.PolyData):
            verts, faces = mesh.points, mesh.faces.reshape(-1, 4)[:, 1:4]
            self.trimesh_obj = trimesh.Trimesh(vertices=verts, faces=faces)
        else:
            raise TypeError("mesh должен быть str, trimesh.Trimesh или pv.PolyData")

        center_mass = self.trimesh_obj.bounding_box.center_mass
        self.trimesh_obj.apply_translation(-center_mass)

        if scale_factors is not None and not np.allclose(scale_factors, 1.0):
            scale_mat = np.eye(4)
            scale_mat[0, 0] = scale_factors[0]
            scale_mat[1, 1] = scale_factors[1]
            scale_mat[2, 2] = scale_factors[2]
            self.trimesh_obj.apply_transform(scale_mat)

        self._local_trimesh = self.trimesh_obj.copy()

        rot_4x4 = np.eye(4)
        rot_4x4[:3, :3] = R.from_euler("xyz", rotation_degrees, degrees=True).as_matrix()
        self.trimesh_obj.apply_transform(rot_4x4)

        if translation is not None:
            self.trimesh_obj.apply_translation(translation)

        self.mesh = self.trimesh_obj
        self.intersector = trimesh.ray.ray_triangle.RayMeshIntersector(self.mesh)

        self.material = material if material is not None else Material(n=1.0)
        self._last_hit_triangle_idx = None

    def get_mesh(self):
        """Возвращает ЧИСТЫЙ локальный меш в нуле. WebGL сам подвинет его через user_matrix."""
        verts = self._local_trimesh.vertices
        faces = np.hstack([np.full((len(self._local_trimesh.faces), 1), 3), self._local_trimesh.faces])
        return pv.PolyData(verts, faces)

    def intersect(self, ray: Ray) -> Optional[float]:
        origins = np.array([ray.origin])
        directions = np.array([ray.direction])
        locations, _, tri_indices = self.intersector.intersects_location(origins, directions, multiple_hits=False)
        if len(locations) == 0:
            return None
        hit_point = locations[0]
        t = np.linalg.norm(hit_point - ray.origin)
        if t <= 1e-6:
            return None
        self._last_hit_triangle_idx = tri_indices[0]
        return t

    def get_normal(self, point):
        if self._last_hit_triangle_idx is not None:
            raw_normal = self.mesh.face_normals[self._last_hit_triangle_idx]
        else:
            _, _, tri_idx = trimesh.proximity.closest_point(self.mesh, [point])
            raw_normal = self.mesh.face_normals[tri_idx[0]]
        norm = np.linalg.norm(raw_normal)
        if norm < 1e-12:
            return raw_normal
        return raw_normal / norm
