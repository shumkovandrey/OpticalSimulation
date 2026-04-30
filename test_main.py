import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as R
from typing import List, Optional, Tuple

# -------------------------------
# Утилиты для оптических расчётов
# -------------------------------

def refract(ray_dir: np.ndarray, normal: np.ndarray, n1: float, n2: float) -> Optional[np.ndarray]:
    """
    Закон Снеллиуса с автоматической коррекцией нормали.
    Возвращает новый вектор направления или None, если луч поглощён.
    """
    eta = n1 / n2
    cos_i = np.dot(normal, ray_dir)

    # Убеждаемся, что нормаль направлена навстречу лучу
    actual_normal = normal
    if cos_i > 0:
        actual_normal = -normal
        cos_i = np.dot(actual_normal, ray_dir)

    cos_i = -cos_i  # теперь cos_i >= 0
    sin2_t = eta ** 2 * (1.0 - cos_i ** 2)

    if sin2_t > 1.0:  # Полное внутреннее отражение
        return ray_dir - 2 * np.dot(ray_dir, actual_normal) * actual_normal

    cos_t = np.sqrt(max(0.0, 1.0 - sin2_t))
    return eta * ray_dir + (eta * cos_i - cos_t) * actual_normal


# ---------------------
# Классы элементов сцены
# ---------------------

class Ray:
    """Луч с началом и единичным направлением."""
    def __init__(self, origin: np.ndarray, direction: np.ndarray):
        self.origin = np.array(origin, dtype=float)
        self.direction = np.array(direction, dtype=float)
        self.direction /= np.linalg.norm(self.direction)


class PlaneSurface:
    """
    Плоская поверхность (грань, экран, зеркало).
    Ограничена круглой апертурой (edge_radius) в плоскости,
    перпендикулярной lens_axis с центром в lens_origin.
    """
    def __init__(self, point: np.ndarray, normal: np.ndarray, n_inside: float,
                 lens_origin: np.ndarray, lens_axis: np.ndarray, edge_radius: float):
        self.point = np.array(point, dtype=float)
        self.normal = np.array(normal, dtype=float)
        self.normal /= np.linalg.norm(self.normal)
        self.n = n_inside
        self.lens_origin = np.array(lens_origin, dtype=float)
        self.lens_axis = np.array(lens_axis, dtype=float)
        self.lens_axis /= np.linalg.norm(self.lens_axis)
        self.edge_radius = edge_radius

    def intersect(self, ray: Ray) -> Optional[float]:
        """Возвращает расстояние t до пересечения или None."""
        dot_dn = np.dot(ray.direction, self.normal)
        if abs(dot_dn) < 1e-6:
            return None
        t = np.dot(self.point - ray.origin, self.normal) / dot_dn
        if t <= 1e-6:
            return None

        hit_p = ray.origin + ray.direction * t
        # Проверка: попадает ли точка в апертуру (круг)
        vec_to_hit = hit_p - self.lens_origin
        projection = np.dot(vec_to_hit, self.lens_axis)
        dist_to_axis = np.linalg.norm(vec_to_hit - projection * self.lens_axis)

        if dist_to_axis <= self.edge_radius + 1e-6:
            return t
        return None

    def get_normal(self, point: np.ndarray) -> np.ndarray:
        return self.normal

    def interact(self, ray_dir: np.ndarray, normal: np.ndarray, n1: float, n2: float) -> Optional[np.ndarray]:
        return refract(ray_dir, normal, n1, n2)


class SphereSurface:
    """
    Сферическая поверхность с ограничениями по радиусу апертуры и продольному
    положению (толщине) вдоль оптической оси линзы.
    """
    def __init__(self, center: np.ndarray, radius: float, n_inside: float,
                 lens_origin: np.ndarray, lens_axis: np.ndarray,
                 edge_radius: float, thickness: float):
        self.center = np.array(center, dtype=float)
        self.radius = radius
        self.n = n_inside
        self.lens_origin = np.array(lens_origin, dtype=float)
        self.lens_axis = np.array(lens_axis, dtype=float)
        self.lens_axis /= np.linalg.norm(self.lens_axis)
        self.edge_radius = edge_radius
        self.thickness = thickness

    def intersect(self, ray: Ray) -> Optional[float]:
        """Пересечение луча со сферой с учётом границ линзы."""
        oc = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius ** 2

        disc = b ** 2 - 4 * a * c
        if disc < 0:
            return None

        t1 = (-b - np.sqrt(disc)) / (2.0 * a)
        t2 = (-b + np.sqrt(disc)) / (2.0 * a)

        valid_ts = []
        for t in (t1, t2):
            if t <= 1e-6:
                continue
            hit_p = ray.origin + ray.direction * t

            # Проверка на апертуру (радиус от оптической оси)
            vec_to_hit = hit_p - self.lens_origin
            projection = np.dot(vec_to_hit, self.lens_axis)
            dist_to_axis = np.linalg.norm(vec_to_hit - projection * self.lens_axis)

            in_radius = dist_to_axis <= self.edge_radius + 1e-6
            # Грубая проверка по глубине (должна быть уточнена при интегрировании с рёбрами линзы)
            in_thickness = abs(projection) <= self.thickness / 2 + 5.0  # TODO: заменить точным ограничением

            if in_radius and in_thickness:
                valid_ts.append(t)

        return min(valid_ts) if valid_ts else None

    def get_normal(self, point: np.ndarray) -> np.ndarray:
        normal = (point - self.center) / self.radius
        return normal / np.linalg.norm(normal)

    def interact(self, ray_dir: np.ndarray, normal: np.ndarray, n1: float, n2: float) -> Optional[np.ndarray]:
        return refract(ray_dir, normal, n1, n2)


class Screen(PlaneSurface):
    """Экран, поглощающий лучи."""
    def __init__(self, point: np.ndarray, normal: np.ndarray, size: float):
        super().__init__(
            point=point,
            normal=normal,
            n_inside=1.0,
            lens_origin=point,
            lens_axis=normal,
            edge_radius=size / 2
        )
        self.size = size

    def interact(self, ray_dir: np.ndarray, normal: np.ndarray, n1: float, n2: float) -> None:
        return None  # сигнал остановки трассировки


class BoxPrism:
    """Прямоугольная призма из материала с показателем преломления n."""
    def __init__(self, origin: np.ndarray, size_x: float, size_y: float, size_z: float, n: float = 1.5):
        self.origin = np.array(origin, dtype=float)
        self.size = np.array([size_x, size_y, size_z], dtype=float)
        self.n = n
        self.surfaces: List[PlaneSurface] = []

        face_configs = [
            ([1, 0, 0], size_x / 2, [size_y, size_z]),
            ([-1, 0, 0], size_x / 2, [size_y, size_z]),
            ([0, 1, 0], size_y / 2, [size_x, size_z]),
            ([0, -1, 0], size_y / 2, [size_x, size_z]),
            ([0, 0, 1], size_z / 2, [size_x, size_y]),
            ([0, 0, -1], size_z / 2, [size_x, size_y])
        ]

        for norm, offset, edge_dims in face_configs:
            face_center = self.origin + np.array(norm) * offset
            r_limit = np.sqrt(edge_dims[0] ** 2 + edge_dims[1] ** 2) / 2
            surf = PlaneSurface(
                point=face_center,
                normal=norm,
                n_inside=n,
                lens_origin=self.origin,
                lens_axis=norm,
                edge_radius=r_limit
            )
            self.surfaces.append(surf)

    def get_surfaces(self) -> List[PlaneSurface]:
        return self.surfaces

    def get_mesh(self) -> pv.PolyData:
        return pv.Cube(
            center=self.origin,
            x_length=self.size[0],
            y_length=self.size[1],
            z_length=self.size[2]
        )

    def rotate(self, angles_deg: Tuple[float, float, float]):
        rot = R.from_euler('xyz', angles_deg, degrees=True).as_matrix()
        for surf in self.surfaces:
            local_pos = surf.point - self.origin
            surf.point = self.origin + rot @ local_pos
            surf.normal = rot @ surf.normal


class UniversalLens:
    """
    Двояковыпуклая/вогнутая/мениск линза в произвольной ориентации.
    Параметры:
        origin      – геометрический центр линзы,
        axis_dir    – вектор оптической оси (направление от передней к задней грани),
        R1, R2      – радиусы кривизны передней и задней поверхностей (None = плоскость),
        thickness   – толщина вдоль оси,
        edge_radius – радиус апертуры,
        n           – показатель преломления материала.
    """
    def __init__(self, origin: np.ndarray, axis_dir: np.ndarray,
                 R1: Optional[float], R2: Optional[float],
                 thickness: float, edge_radius: float, n: float = 1.5):
        self.R1 = R1
        self.R2 = R2
        self.thickness = thickness
        self.edge_radius = edge_radius
        self.n = n
        self.origin = np.array(origin, dtype=float)
        self.axis_dir = np.array(axis_dir, dtype=float)
        self.axis_dir /= np.linalg.norm(self.axis_dir)

        # Матрица поворота от локальной оси X к axis_dir
        v_old = np.array([1, 0, 0])
        v_new = self.axis_dir
        if np.allclose(v_old, v_new):
            self.rotation = np.eye(3)
        elif np.allclose(v_old, -v_new):
            self.rotation = np.diag([-1, -1, 1])
        else:
            v = np.cross(v_old, v_new)
            s = np.linalg.norm(v)
            c = np.dot(v_old, v_new)
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            self.rotation = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))

        half = thickness / 2
        v1_local = -half
        v2_local = half

        # Передняя поверхность
        if R1 is None:
            p1_world = self.origin + self.rotation @ np.array([v1_local, 0, 0])
            n1_world = self.rotation @ np.array([-1, 0, 0])
            self.front = PlaneSurface(
                point=p1_world, normal=n1_world, n_inside=n,
                lens_origin=self.origin, lens_axis=self.axis_dir, edge_radius=edge_radius
            )
        else:
            c1_world = self.origin + self.rotation @ np.array([v1_local + R1, 0, 0])
            self.front = SphereSurface(
                center=c1_world, radius=abs(R1), n_inside=n,
                lens_origin=self.origin, lens_axis=self.axis_dir,
                edge_radius=edge_radius, thickness=thickness
            )

        # Задняя поверхность
        if R2 is None:
            p2_world = self.origin + self.rotation @ np.array([v2_local, 0, 0])
            n2_world = self.rotation @ np.array([1, 0, 0])
            self.back = PlaneSurface(
                point=p2_world, normal=n2_world, n_inside=n,
                lens_origin=self.origin, lens_axis=self.axis_dir, edge_radius=edge_radius
            )
        else:
            c2_world = self.origin + self.rotation @ np.array([v2_local - R2, 0, 0])
            self.back = SphereSurface(
                center=c2_world, radius=abs(R2), n_inside=n,
                lens_origin=self.origin, lens_axis=self.axis_dir,
                edge_radius=edge_radius, thickness=thickness
            )

    def get_surfaces(self) -> List:
        return [self.front, self.back]

    def get_mesh(self) -> pv.PolyData:
        """Генерирует полигональную модель линзы."""
        rs = np.linspace(0, self.edge_radius, 30)
        phis = np.linspace(0, 2 * np.pi, 60)
        r_grid, phi_grid = np.meshgrid(rs, phis)

        y = r_grid * np.cos(phi_grid)
        z = r_grid * np.sin(phi_grid)

        v1_local = -self.thickness / 2
        v2_local = self.thickness / 2

        def get_local_x(R, v_x, r_vals, is_front):
            if R is not None:
                c_x = v_x + R if is_front else v_x - R
                dx = np.sqrt(np.maximum(0, abs(R) ** 2 - r_vals ** 2))
                return c_x - dx if (is_front and R > 0) or (not is_front and R < 0) else c_x + dx
            else:
                return np.full_like(r_vals, v_x)

        x_front = get_local_x(self.R1, v1_local, r_grid, True)
        x_back = get_local_x(self.R2, v2_local, r_grid, False)

        front_mesh = pv.StructuredGrid(x_front, y, z).extract_surface()
        back_mesh = pv.StructuredGrid(x_back, y, z).extract_surface()

        # Ободок (соединение краёв)
        rim_x = np.array([x_front[:, -1], x_back[:, -1]])
        rim_y = np.array([y[:, -1], y[:, -1]])
        rim_z = np.array([z[:, -1], z[:, -1]])
        rim_mesh = pv.StructuredGrid(rim_x, rim_y, rim_z).extract_surface()

        local_mesh = front_mesh.merge(back_mesh).merge(rim_mesh)

        # Перенос в мировые координаты
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.origin
        return local_mesh.transform(matrix)


# --------------------------------
# Трассировка лучей и визуализация
# --------------------------------

def run_simulation(start_ray: Ray, elements: List, max_bounces: int = 10) -> np.ndarray:
    """Трассирует один луч через последовательность оптических элементов."""
    path = [start_ray.origin]
    current_ray = start_ray
    current_n = 1.0  # начинаем из воздуха

    for _ in range(max_bounces):
        best_t = float('inf')
        hit_obj = None

        for obj in elements:
            t = obj.intersect(current_ray)
            if t and t < best_t:
                best_t = t
                hit_obj = obj

        if hit_obj is None:
            # Луч уходит в бесконечность
            path.append(current_ray.origin + current_ray.direction * 50)
            break

        hit_point = current_ray.origin + current_ray.direction * best_t
        path.append(hit_point)

        normal = hit_obj.get_normal(hit_point)
        next_n = hit_obj.n if abs(current_n - 1.0) < 1e-6 else 1.0
        new_dir = hit_obj.interact(current_ray.direction, normal, current_n, next_n)

        if new_dir is None:  # Поглощён (например, экраном)
            break

        current_ray = Ray(hit_point, new_dir)
        current_n = next_n

    return np.array(path)


def visualize_scene(plotter: pv.Plotter, trajectory_list: List[np.ndarray],
                    elements: List, lenses: Optional[List[UniversalLens]] = None):
    """Отрисовка траекторий, поверхностей и линз."""
    plotter.set_background("black")
    # Лучи
    for traj in trajectory_list:
        path = pv.PolyData(traj)
        path.lines = np.hstack(([len(traj)], range(len(traj))))
        plotter.add_mesh(path, color="yellow", line_width=2, render_lines_as_tubes=True)
        pts = pv.PolyData(traj)
        plotter.add_mesh(pts, color="purple", point_size=10, render_points_as_spheres=True)

    drawn_surfaces = set()
    if lenses:
        for lens in lenses:
            plotter.add_mesh(lens.get_mesh(), color="cyan", opacity=0.75, smooth_shading=True)
            drawn_surfaces.add(lens.front)
            drawn_surfaces.add(lens.back)

    # Остальные элементы (зеркала, призмы и т.д.)
    for obj in elements:
        if obj in drawn_surfaces:
            continue
        if isinstance(obj, PlaneSurface):
            plane = pv.Plane(center=obj.point, direction=obj.normal, i_size=5, j_size=5)
            plotter.add_mesh(plane, color="lightblue", opacity=0.5)
        elif isinstance(obj, SphereSurface):
            sphere = pv.Sphere(radius=obj.radius, center=obj.center)
            plotter.add_mesh(sphere, color="grey", opacity=0.2)

    plotter.add_axes()


# --------------------------------
# Пример использования
# --------------------------------

if __name__ == "__main__":
    plotter = pv.Plotter()

    # Собирающая линза на оси X
    l1 = UniversalLens(
        origin=[-20, 0, 0],
        axis_dir=[1, 0, 0],
        R1=5, R2=5, thickness=0.5, edge_radius=1.75, n=1.5
    )

    all_lenses = [l1]

    for i, lens in enumerate(all_lenses):
        plotter.add_mesh(lens.get_mesh(), color="cyan", opacity=0.3, smooth_shading=True)

        # Пучок параллельных лучей вдоль оси линзы
        for off in np.linspace(-1.0, 1.0, 20):
            # Локальный старт: смещение по Y и Z в плоскости апертуры
            local_start = np.array([-15, off, 0])
            world_start = lens.origin + lens.rotation @ local_start
            world_dir = lens.axis_dir

            ray = Ray(origin=world_start, direction=world_dir)
            trajectory = run_simulation(ray, lens.get_surfaces(), max_bounces=5)

            path = pv.PolyData(trajectory)
            path.lines = np.hstack(([len(trajectory)], range(len(trajectory))))
            plotter.add_mesh(path, color="orange", opacity=0.7, line_width=1)

            if len(trajectory) > 2:
                hits = pv.PolyData(trajectory[1:-1])
                plotter.add_mesh(hits, color="red", point_size=8, render_points_as_spheres=True)

    plotter.set_background("black")
    plotter.show_grid(color="white")
    plotter.view_xy()
    plotter.enable_parallel_projection()
    plotter.enable_terrain_style(mouse_wheel_zooms=True)
    plotter.show()
