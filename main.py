import random

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as R
from typing import List, Optional, Tuple

pv.global_theme.allow_empty_mesh = True

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


def calculate_rotation_matrix(v_to):
    """
    Создает матрицу поворота, которая переводит вектор [1, 0, 0]
    в вектор v_to.
    """
    v_to = np.array(v_to, dtype=float)
    # Нормализация входного вектора (приведение к длине 1)
    norm = np.linalg.norm(v_to)
    if norm < 1e-10:
        return np.eye(3)
    v_to /= norm

    v_from = np.array([1.0, 0.0, 0.0])  # Базовая ось симуляции (X)

    # 1. Если векторы уже совпадают
    if np.allclose(v_from, v_to):
        return np.eye(3)

    # 2. Если векторы противоположны (разворот на 180 градусов)
    if np.allclose(v_from, -v_to):
        # Поворот на 180 вокруг оси Y
        return np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

    # 3. Общий случай: находим ось поворота (векторное произведение)
    # и косинус угла (скалярное произведение)
    v = np.cross(v_from, v_to)  # Вектор оси поворота
    c = np.dot(v_from, v_to)  # Косинус угла между векторами

    # Кососимметричная матрица K
    K = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    # Формула Родрига для поворота вектора к вектору
    R = np.eye(3) + K + (K @ K) * (1 / (1 + c))
    return R


def calculate_radius(target_f, n, thickness=0):
    """
    Рассчитывает радиус R для двояковыпуклой линзы.
    Если thickness=0, используется формула тонкой линзы.
    """
    if thickness == 0:
        return 2 * target_f * (n - 1)

    # Коэффициенты квадратного уравнения Ax^2 + Bx + C = 0, где x = 1/R
    A = ((n - 1) * thickness) / n
    B = 2
    C = -1 / (target_f * (n - 1))

    # Решаем через дискриминант
    D = B ** 2 - 4 * A * C
    if D < 0:
        return None  # Решения нет для такой толщины

    x = (-B + np.sqrt(D)) / (2 * A)
    return 1 / x


def normalize(vec):
    return vec / np.linalg.norm(vec)


def get_tangents(normal):
    # Выбираем произвольный вектор, не параллельный normal
    if abs(normal[0]) < 0.9:
        arbitrary = np.array([1, 0, 0])
    else:
        arbitrary = np.array([0, 1, 0])
    t1 = np.cross(normal, arbitrary)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)
    t2 /= np.linalg.norm(t2)
    return t1, t2


# def get_mirror_mesh(surface):
#     """
#     surface: объект класса SphereSurface (у которого есть center, radius,
#              lens_origin, lens_axis, edge_radius)
#     """
#     # 1. Создаем полную сферу в локальном НУЛЕ
#     # Это гарантирует, что у нас не будет паразитных смещений
#     radius = surface.radius
#     mesh = pv.Sphere(radius=radius, center=(0, 0, 0),
#                      phi_resolution=80, theta_resolution=80)
#
#     # 2. Обрезаем сферу, чтобы оставить только "шапочку" нужного радиуса
#     # В локальных координатах (ось X) вершина сферы находится в точке [radius, 0, 0]
#     # Отрезаем всё, что дальше edge_radius по высоте (Y и Z)
#     # Самый простой способ для сферы — обрезать плоскостью по координате X
#     sagitta = radius - np.sqrt(max(0, radius ** 2 - surface.edge_radius ** 2))
#
#     # Отрезаем заднюю часть сферы, оставляя только "купол" высотой sagitta
#     mesh = mesh.clip(normal=[1, 0, 0], origin=[radius - sagitta, 0, 0], invert=False)
#
#     # 3. Глобальная трансформация (синхронизация с математикой)
#     # Создаем матрицу 4x4
#     matrix = np.eye(4)
#
#     # Используем ту же функцию вращения, что и для линз,
#     # чтобы направить локальную ось X вдоль lens_axis
#     rot_matrix = -calculate_rotation_matrix(surface.lens_axis)
#     matrix[:3, :3] = rot_matrix
#
#     # СМЕЩЕНИЕ:
#     # В локальных координатах вершина в [radius, 0, 0].
#     # Нам нужно, чтобы после поворота она попала в lens_origin.
#     # Поэтому центр сферы в мировых координатах = lens_origin - axis * radius
#     world_center = surface.lens_origin + (surface.lens_axis * radius)
#     matrix[:3, 3] = world_center
#
#     return mesh.transform(matrix)


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
    def __init__(self, point, normal, n_inside,
                 lens_origin, lens_axis, edge_radius, is_mirror=False,
                 half_sizes=None, face_tangents=None):
        self.point = np.array(point, dtype=float)
        self.normal = np.array(normal, dtype=float)
        self.normal /= np.linalg.norm(self.normal)
        self.n = n_inside
        self.lens_origin = np.array(lens_origin, dtype=float)
        self.lens_axis = np.array(lens_axis, dtype=float)
        self.lens_axis /= np.linalg.norm(self.lens_axis)
        self.edge_radius = edge_radius
        self.is_mirror = is_mirror
        # Прямоугольная апертура (опционально)
        self.half_sizes = half_sizes
        self.face_tangents = face_tangents

    def intersect(self, ray: Ray) -> Optional[float]:
        dot_dn = np.dot(ray.direction, self.normal)
        if abs(dot_dn) < 1e-6:
            return None
        t = np.dot(self.point - ray.origin, self.normal) / dot_dn
        if t <= 1e-6:
            return None

        hit_p = ray.origin + ray.direction * t

        # Прямоугольная проверка (если заданы параметры)
        if self.half_sizes is not None and self.face_tangents is not None:
            vec = hit_p - self.lens_origin
            u = np.dot(vec, self.face_tangents[0])
            v = np.dot(vec, self.face_tangents[1])

            if abs(u) <= self.half_sizes[0] + 1e-6 and abs(v) <= self.half_sizes[1] + 1e-6:
                return t
            return None

        # Круговая проверка (если прямоугольные параметры не заданы)
        vec_to_hit = hit_p - self.lens_origin
        projection = np.dot(vec_to_hit, self.lens_axis)
        dist_to_axis = np.linalg.norm(vec_to_hit - projection * self.lens_axis)

        if dist_to_axis <= self.edge_radius + 1e-6:
            return t
        return None

    def get_normal(self, point: np.ndarray) -> np.ndarray:
        return self.normal

    def interact(self, ray_dir, normal, n1, n2):
        if self.is_mirror:
            dot = np.dot(normal, ray_dir)
            actual_normal = normal if dot < 0 else -normal
            reflected_dir = ray_dir - 2 * np.dot(ray_dir, actual_normal) * actual_normal
            return reflected_dir / np.linalg.norm(reflected_dir)
        return refract(ray_dir, normal, n1, n2)


class SphereSurface:
    """
    Сферическая поверхность с ограничениями по радиусу апертуры и продольному
    положению (толщине) вдоль оптической оси линзы.
    """
    def __init__(self, center: np.ndarray, radius: float, n_inside: float,
                 lens_origin: np.ndarray, lens_axis: np.ndarray,
                 edge_radius: float, thickness: float, is_mirror=False):
        self.center = np.array(center, dtype=float)
        self.radius = radius
        self.n = n_inside
        self.lens_origin = np.array(lens_origin, dtype=float)
        self.lens_axis = np.array(lens_axis, dtype=float)
        self.lens_axis /= np.linalg.norm(self.lens_axis)
        self.edge_radius = edge_radius
        self.thickness = thickness
        self.is_mirror = is_mirror

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
        if self.is_mirror:
            # Находим косинус угла падения
            dot = np.dot(normal, ray_dir)

            # ГЛАВНЫЙ СЕКРЕТ: нормаль должна всегда смотреть НАВСТРЕЧУ лучу
            # Если dot > 0, значит нормаль и луч смотрят в одну сторону -> разворачиваем нормаль
            actual_normal = normal if dot < 0 else -normal

            # Формула отражения
            reflected_dir = ray_dir - 2 * np.dot(ray_dir, actual_normal) * actual_normal
            return reflected_dir / np.linalg.norm(reflected_dir)
        return refract(ray_dir, normal, n1, n2)

    def get_mesh(self) -> pv.PolyData:
        """Полигональная модель сферической поверхности (линзы или зеркала)."""
        abs_radius = abs(self.radius)
        # Полная сфера в начале координат
        mesh = pv.Sphere(radius=abs_radius, center=(0.0, 0.0, 0.0),
                         phi_resolution=80, theta_resolution=80)

        # Стрелка прогиба (sagitta) для апертуры edge_radius
        sagitta = abs_radius - np.sqrt(max(0.0, abs_radius ** 2 - self.edge_radius ** 2))

        # Вершина в локальных координатах
        R = self.radius  # со знаком
        if R > 0:
            # Вогнутая поверхность: вершина на +X, обрезаем всё, что левее (X < R - sagitta)
            clip_normal = [1, 0, 0]
            clip_origin = [R - sagitta, 0, 0]
            mesh = mesh.clip(normal=clip_normal, origin=clip_origin, invert=False)
        else:
            # Выпуклая поверхность: вершина на -X (R отрицателен), обрезаем всё, что правее (X > R + sagitta)
            clip_normal = [-1, 0, 0]
            clip_origin = [R + sagitta, 0, 0]  # R + sagitta находится ближе к нулю
            mesh = mesh.clip(normal=clip_normal, origin=clip_origin, invert=False)

        # Мировая матрица: поворот + перенос
        matrix = np.eye(4)
        # Локальная ось X направлена от центра к вершине (точка (R,0,0)).
        # Нам нужно, чтобы после поворота эта ось совпала с направлением от центра к lens_origin.
        # Центр в мировых координатах: world_center = lens_origin + lens_axis * self.radius.
        # Вектор от центра к вершине: lens_origin - world_center = -lens_axis * self.radius / |self.radius|?
        # Но проще использовать уже проверенный метод:
        rot_matrix = -calculate_rotation_matrix(self.lens_axis)
        matrix[:3, :3] = rot_matrix
        world_center = self.lens_origin + self.lens_axis * self.radius
        matrix[:3, 3] = world_center

        return mesh.transform(matrix, inplace=False)


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
        self.surfaces = []
        self.rotation_matrix = np.eye(3)

        # Полуразмеры и направления рёбер (в локальной системе до поворота)
        half = self.size / 2
        # Набор грань: нормаль, смещение центра, пара локальных осей (вдоль сторон), полуразмеры
        face_configs = [
            ([1, 0, 0], [half[0], 0, 0], ([0, 1, 0], [0, 0, 1]), (half[1], half[2])),
            ([-1, 0, 0], [-half[0], 0, 0], ([0, 1, 0], [0, 0, 1]), (half[1], half[2])),
            ([0, 1, 0], [0, half[1], 0], ([1, 0, 0], [0, 0, 1]), (half[0], half[2])),
            ([0, -1, 0], [0, -half[1], 0], ([1, 0, 0], [0, 0, 1]), (half[0], half[2])),
            ([0, 0, 1], [0, 0, half[2]], ([1, 0, 0], [0, 1, 0]), (half[0], half[1])),
            ([0, 0, -1], [0, 0, -half[2]], ([1, 0, 0], [0, 1, 0]), (half[0], half[1]))
        ]

        for norm_vec, center_offset, (ax1, ax2), (half_u, half_v) in face_configs:
            norm_vec = np.array(norm_vec, dtype=float)
            center = self.origin + np.array(center_offset)
            tangents = (np.array(ax1, dtype=float), np.array(ax2, dtype=float))
            surf = PlaneSurface(
                point=center,
                normal=norm_vec,
                n_inside=n,
                lens_origin=center,
                lens_axis=norm_vec,
                edge_radius=0.0,  # не используется при наличии half_sizes
                half_sizes=(half_u, half_v),
                face_tangents=tangents
            )
            self.surfaces.append(surf)

    def get_surfaces(self) -> List[PlaneSurface]:
        return self.surfaces

    def get_mesh(self) -> pv.PolyData:
        # Создаём куб с центром в начале координат
        mesh = pv.Cube(
            center=(0.0, 0.0, 0.0),
            x_length=self.size[0],
            y_length=self.size[1],
            z_length=self.size[2]
        )
        # Матрица полного преобразования: поворот + смещение
        transform = np.eye(4)
        transform[:3, :3] = self.rotation_matrix
        transform[:3, 3] = self.origin
        return mesh.transform(transform, inplace=False)

    def rotate(self, angles_deg: Tuple[float, float, float]):
        rot = R.from_euler('xyz', angles_deg, degrees=True).as_matrix()
        self.rotation_matrix = rot @ self.rotation_matrix
        for surf in self.surfaces:
            local_pos = surf.point - self.origin
            surf.point = self.origin + rot @ local_pos
            surf.normal = rot @ surf.normal
            surf.lens_axis = surf.normal
            surf.lens_origin = surf.point
            # Поворачиваем локальные оси грани, если они есть
            if surf.face_tangents is not None:
                t1, t2 = surf.face_tangents
                surf.face_tangents = (rot @ t1, rot @ t2)


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

        # 2. РАСЧЕТ ОПТИЧЕСКИХ ПАРАМЕТРОВ (Формула толстой линзы)
        # Для формулы: r1 и r2 (радиус второй берется с инверсией знака)
        r1_val = R1 if R1 else 1e10
        r2_val = -R2 if R2 else -1e10

        inv_f = (n - 1) * (1 / r1_val - 1 / r2_val + ((n - 1) * thickness) / (n * r1_val * r2_val))

        if abs(inv_f) < 1e-10:
            self.f_dist = float('inf')
        else:
            self.f_dist = 1 / inv_f

        # Расчет положения главной плоскости H2 для корректного draw_axis
        self.h2 = -(self.f_dist * (n - 1) * thickness) / (n * r1_val)

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

    def draw_axis(self, plot, length=100):
        # if not self.show_axis:
        #     return

        # 1. Отрисовка основной оси
        # Направляем её вдоль вектора axis_dir
        axis_start = self.origin - self.axis_dir * (length / 2)
        axis_stop = self.origin + self.axis_dir * (length / 2)
        axis_line = pv.Line(axis_start, axis_stop)
        plot.add_mesh(axis_line, color="white", line_width=1, opacity=0.5)

        # 2. Проверка на бесконечный фокус
        if np.isinf(self.f_dist) or abs(self.f_dist) > 1000:
            return

        # 3. Расчет положения главной плоскости H2 относительно центра линзы
        # (необходимо для точного отображения фокуса в "толстой" линзе)
        r1_val = self.R1 if self.R1 else 1e10
        h2 = -(self.f_dist * (self.n - 1) * self.thickness) / (self.n * r1_val)

        # Точка отсчета фокуса (Главная точка на оптической оси)
        # Смещаемся от центра вдоль оси на (толщина/2 + h2)
        p_point = self.origin + self.axis_dir * (self.thickness / 2 + h2)

        # 4. Метки фокусов F и -F
        f = self.f_dist

        # Вектор "вверх" для отрисовки засечек (перпендикулярен оси)
        up_vec = np.array([0, 1, 0]) if abs(self.axis_dir[0]) < 0.9 else np.array([0, 0, 1])
        mark_vec = np.cross(self.axis_dir, up_vec)
        mark_vec /= np.linalg.norm(mark_vec)

        for i in range(-5, 5):
            # Позиция метки в мировых координатах
            f_pos = p_point + self.axis_dir * i * f

            # Вертикальная засечка
            mark_line = pv.Line(f_pos - mark_vec * 0.5, f_pos + mark_vec * 0.5)
            plot.add_mesh(mark_line, color="red", line_width=3)

            # Текстовая подпись
            plot.add_point_labels([f_pos + mark_vec * 0.8], [f"{i}F"],
                                     font_size=12, text_color="yellow",
                                     shape=None, show_points=False)


class RectangularMirror(PlaneSurface):
    """
    Прямоугольное плоское зеркало.
    Параметры:
        center  – геометрический центр зеркала,
        normal  – нормаль к поверхности (единичный вектор),
        width   – полная ширина (размер вдоль первой касательной),
        height  – полная высота (размер вдоль второй касательной),
        n_inside – показатель преломления внутри (по умолчанию 1.0).
    """
    def __init__(self, center: np.ndarray, normal: np.ndarray,
                 width: float, height: float, n_inside: float = 1.0):
        self.width = width
        self.height = height
        half_u = width / 2.0
        half_v = height / 2.0

        center = np.asarray(center, dtype=float)
        normal = np.asarray(normal, dtype=float)
        normal /= np.linalg.norm(normal)

        tangent1, tangent2 = get_tangents(normal)

        super().__init__(
            point=center,
            normal=normal,
            n_inside=n_inside,
            lens_origin=center,
            lens_axis=normal,
            edge_radius=0.0,         # не используется
            is_mirror=True,
            half_sizes=(half_u, half_v),
            face_tangents=(tangent1, tangent2)
        )

    def get_mesh(self) -> pv.PolyData:
        """Прямоугольник, заданный центром, касательными и полуразмерами."""
        t1, t2 = self.face_tangents
        hu, hv = self.half_sizes
        center = self.lens_origin

        # Четыре угла прямоугольника
        p0 = center - hu * t1 - hv * t2
        p1 = center + hu * t1 - hv * t2
        p2 = center + hu * t1 + hv * t2
        p3 = center - hu * t1 + hv * t2

        # Два треугольника
        vertices = np.array([p0, p1, p2, p3])
        faces = np.array([[3, 0, 1, 2],
                          [3, 0, 2, 3]])  # полигоны: треугольники (0,1,2) и (0,2,3)
        mesh = pv.PolyData(vertices, faces)
        return mesh


class Aperture(PlaneSurface):
    """Диафрагма: непрозрачная плоскость с круглым отверстием."""
    def __init__(self, point: np.ndarray, normal: np.ndarray,
                 aperture_radius: float, outer_radius: float = 3.0,
                 n_inside: float = 1.0):
        super().__init__(
            point=point,
            normal=normal,
            n_inside=n_inside,
            lens_origin=point,
            lens_axis=normal,
            edge_radius=0.0,
            is_mirror=False
        )
        self.aperture_radius = aperture_radius
        self.outer_radius = outer_radius
        self._hit_opaque = False

    def intersect(self, ray: Ray) -> Optional[float]:
        dot_dn = np.dot(ray.direction, self.normal)
        if abs(dot_dn) < 1e-6:
            return None
        t = np.dot(self.point - ray.origin, self.normal) / dot_dn
        if t <= 1e-6:
            return None
        hit_point = ray.origin + ray.direction * t
        vec = hit_point - self.lens_origin
        dist_to_axis = np.linalg.norm(vec - np.dot(vec, self.lens_axis) * self.lens_axis)
        self._hit_opaque = (dist_to_axis > self.aperture_radius + 1e-6)
        return t

    def interact(self, ray_dir: np.ndarray, normal: np.ndarray,
                 n1: float, n2: float) -> Optional[np.ndarray]:
        if self._hit_opaque:
            self._hit_opaque = False
            return None
        return ray_dir

    def get_mesh(self) -> pv.PolyData:
        """Кольцо, правильно ориентированное по self.normal."""
        # Диск в плоскости XY (нормаль Z) с центром в (0,0,0)
        disc = pv.Disc(
            center=(0, 0, 0),
            normal=(0, 0, 1),
            inner=self.aperture_radius,
            outer=self.outer_radius,
            c_res=100
        )

        # Строим матрицу поворота от [0,0,1] к self.normal
        v_from = np.array([0.0, 0.0, 1.0])
        v_to = self.normal
        v_from = v_from / np.linalg.norm(v_from)
        v_to = v_to / np.linalg.norm(v_to)

        if np.allclose(v_from, v_to):
            rot_matrix = np.eye(3)
        elif np.allclose(v_from, -v_to):
            # Поворот на 180° вокруг оси, перпендикулярной v_from
            if abs(v_from[0]) < 0.9:
                perp = np.array([1.0, 0.0, 0.0])
            else:
                perp = np.array([0.0, 1.0, 0.0])
            rot_matrix = -np.eye(3) + 2 * np.outer(perp, perp) / np.dot(perp, perp)
        else:
            v = np.cross(v_from, v_to)
            c = np.dot(v_from, v_to)
            K = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
            rot_matrix = np.eye(3) + K + (K @ K) * (1.0 / (1.0 + c))

        # Мировая матрица: поворот + перенос
        transform = np.eye(4)
        transform[:3, :3] = rot_matrix
        transform[:3, 3] = self.lens_origin

        return disc.transform(transform, inplace=False)


# --------------------------------
# Трассировка лучей и визуализация
# --------------------------------

def run_simulation(start_ray: Ray, elements: List, max_bounces: int = 20) -> np.ndarray:
    EPS = 1e-4  # малое смещение

    path = [start_ray.origin]
    current_ray = start_ray
    current_n = 1.0

    for _ in range(max_bounces):
        best_t = float('inf')
        hit_obj = None

        for obj in elements:
            t = obj.intersect(current_ray)
            if t and t < best_t:
                best_t = t
                hit_obj = obj

        if hit_obj is None:
            path.append(current_ray.origin + current_ray.direction * 50)
            break

        hit_point = current_ray.origin + current_ray.direction * best_t
        path.append(hit_point)

        normal = hit_obj.get_normal(hit_point)
        next_n = hit_obj.n if abs(current_n - 1.0) < 1e-6 else 1.0
        new_dir = hit_obj.interact(current_ray.direction, normal, current_n, next_n)

        if new_dir is None:
            break

        # Смещаем точку вдоль нового направления, чтобы не задеть ту же поверхность
        hit_point_safe = hit_point + EPS * new_dir
        current_ray = Ray(hit_point_safe, new_dir)
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


plotter = pv.Plotter()
plotter.set_background("black")
plotter.view_isometric()
plotter.enable_parallel_projection()
plotter.enable_terrain_style(mouse_wheel_zooms=True)
plotter.view_vector((0, 0, 1), viewup=(0, 1, 0))
plotter.add_axes()

if __name__ == "__main__":
    # plotter = pv.Plotter()
    # plotter.set_background("black")
    # plotter.show_grid(color="white")
    # plotter.view_isometric()
    # plotter.enable_parallel_projection()
    # plotter.enable_terrain_style(mouse_wheel_zooms=True)
    # plotter.view_xy()

    # 1. Настройка сцены
    plotter = pv.Plotter()
    plotter.set_background("black")
    plotter.show_grid(color="white")
    plotter.view_xy()
    # plotter.add_legend()
    # plotter.enable_parallel_projection()
    plotter.enable_terrain_style(mouse_wheel_zooms=True)

    # Параметры призмы (BoxPrism)
    origin = np.array([0.0, 0.0, 0.0])
    size = [10.0, 20.0, 15.0]  # [X, Y, Z]

    # Настройка спектра лучей
    colors = ["red", "orange", "yellow", "green", "cyan", "blue", "violet"]
    n_values = np.linspace(1.50, 1.65, len(colors))
    source_pos = np.array([-25.0, 5.0, 0.0])
    target_point = np.array([-5.0, 0.0, 0.0])
    direction = (target_point - source_pos) / np.linalg.norm(target_point - source_pos)

    # Создаем "актеров" для лучей, чтобы обновлять их геометрию
    ray_actors = []
    for color in colors:
        actor = plotter.add_mesh(pv.PolyData(), color=color, opacity=0.9, line_width=3)
        ray_actors.append(actor)

    # 2. Основной цикл анимации
    plotter.show(interactive_update=True)

    angle_step_z = 0.5
    angle_step_x = -0.25
    angle_step_y = 1
    current_angle_z = 0.0
    current_angle_x = 0.0
    current_angle_y = 0.0

    state = {'paused': False}


    def toggle_pause():
        state['paused'] = not state['paused']
        if state['paused']:
            print("Пауза")
        else:
            print("Воспроизведение")


    # 2. Настраиваем обработчик событий перед показом
    # При нажатии "Space" (пробел) будет вызываться функция toggle_pause
    plotter.add_key_event("space", toggle_pause)

    plotter.show(interactive_update=True)
    while plotter.render:
        if state['paused']:
            plotter.update()
            continue

        current_angle_z += angle_step_z
        current_angle_x += angle_step_x
        current_angle_y += angle_step_y

        # Очищаем старую визуализацию призмы
        plotter.remove_actor("prism_mesh")

        # Создаем и поворачиваем математическую модель BoxPrism
        prism = BoxPrism(origin=origin, size_x=size[0], size_y=size[1], size_z=size[2], n=2)
        prism.rotate((current_angle_x, current_angle_y, current_angle_z))  # Метод поворота всех граней

        # Обновляем визуальную модель
        prism_mesh = prism.get_mesh()
        plotter.add_mesh(prism_mesh, color="cyan", opacity=0.2, name="prism_mesh", show_edges=True)


        sc = Screen(point=[20, 0, 0], normal=[-1, 0, 0], size=30)
        screen_mesh = pv.Plane(center=sc.point, direction=sc.normal, i_size=sc.size, j_size=sc.size)

        plotter.add_mesh(screen_mesh, color="white", opacity=1, name="prism_mesh", show_edges=True)

        # 3. Пересчет траекторий лучей
        surfaces = prism.get_surfaces() + [sc]
        for i, n_val in enumerate(n_values):
            # Применяем показатель преломления для конкретного цвета
            for surf in surfaces:
                surf.n = n_val

            ray = Ray(origin=source_pos, direction=direction)
            # Рассчитываем путь луча через повернутые грани
            trajectory = run_simulation(ray, surfaces, max_bounces=4)

            # Обновляем PolyData луча без пересоздания актера
            new_path = pv.PolyData(trajectory)
            new_path.lines = np.hstack(([len(trajectory)], range(len(trajectory))))
            ray_actors[i].mapper.dataset.copy_from(new_path)

        # Обновление окна
        plotter.update()

    # # ------------------------------------------------------------
    # # 1. Создаём набор разнотипных линз с разными положениями и ориентациями
    # # ------------------------------------------------------------
    # lenses = []
    #
    # # Двояковыпуклая линза (R1>0, R2>0) – стандартная
    # lenses.append(UniversalLens(
    #     origin=np.array([0, 20, 0]),
    #     axis_dir=np.array([1, 0, 0]),
    #     R1=10.0, R2=10.0,
    #     thickness=2.0, edge_radius=3.0, n=1.5
    # ))
    #
    # # Двояковогнутая линза (R1<0, R2<0)
    # lenses.append(UniversalLens(
    #     origin=np.array([0, 10, 0]),
    #     axis_dir=np.array([1, 0, 0]),
    #     R1=-10.0, R2=-10.0,
    #     thickness=2.0, edge_radius=3.0, n=1.5
    # ))
    #
    # # Плоско-выпуклая (передняя плоская, задняя выпуклая)
    # lenses.append(UniversalLens(
    #     origin=np.array([0, 0, 0]),
    #     axis_dir=np.array([1, 0, 0]),
    #     R1=None, R2=10.0,
    #     thickness=2.0, edge_radius=3.0, n=1.5
    # ))
    #
    # # Выпукло-плоская (передняя выпуклая, задняя плоская)
    # lenses.append(UniversalLens(
    #     origin=np.array([0, -10, 0]),
    #     axis_dir=np.array([1, 0, 0]),
    #     R1=10.0, R2=None,
    #     thickness=2.0, edge_radius=3.0, n=1.5
    # ))
    #
    # # Мениск (R1>0, R2>0, но разной кривизны)
    # lenses.append(UniversalLens(
    #     origin=np.array([0, -20, 0]),
    #     axis_dir=np.array([1, 0, 0]),
    #     R1=10.0, R2=5.0,
    #     thickness=2.0, edge_radius=3.0, n=1.5
    # ))
    #
    # # Линза, повёрнутая на 45° вокруг оси Z (наклон в плоскости XY)
    # lenses.append(UniversalLens(
    #     origin=np.array([40, 30, 0]),
    #     axis_dir=np.array([np.cos(np.radians(45)), np.sin(np.radians(45)), 0]),
    #     R1=10.0, R2=10.0,
    #     thickness=2.0, edge_radius=3.0, n=1.5
    # ))
    #
    # # Линза, повёрнутая на -30° вокруг оси Z
    # lenses.append(UniversalLens(
    #     origin=np.array([40, -30, 0]),
    #     axis_dir=np.array([np.cos(np.radians(-30)), np.sin(np.radians(-30)), 0]),
    #     R1=-10.0, R2=None,
    #     thickness=2.0, edge_radius=3.0, n=1.5
    # ))
    #
    # # ------------------------------------------------------------
    # # 2. Собираем все поверхности в общий список для трассировки
    # # ------------------------------------------------------------
    # all_surfaces = []
    # for lens in lenses:
    #     all_surfaces.extend(lens.get_surfaces())
    #
    # # ------------------------------------------------------------
    # # 3. Генерируем параллельные лучи для каждой линзы
    # # ------------------------------------------------------------
    # all_trajectories = []
    # # Набор смещений по Y для параллельных лучей (будут общими для всех линз)
    # y_offsets = np.linspace(-2.5, 2.5, 6)  # 6 лучей через каждую линзу
    #
    # for i, lens in enumerate(lenses):
    #     # Лучи идут вдоль оси X (слева направо) или по направлению axis_dir линзы?
    #     # Удобнее пускать лучи параллельно глобальной оси X, а линзы могут быть повёрнуты.
    #     # Чтобы лучи были параллельны друг другу для каждой линзы, будем использовать
    #     # направление (1,0,0), стартуя с x = -20, y = lens.origin[1] + offset, z = 0.
    #     trajectories = []
    #     for dy in y_offsets:
    #         start = np.array([-20.0, lens.origin[1] + dy, 0.0])
    #         direction = np.array([1.0, 0.0, 0.0])
    #         ray = Ray(origin=start, direction=direction)
    #         traj = run_simulation(ray, all_surfaces, max_bounces=4)
    #         trajectories.append(traj)
    #     all_trajectories.append({"trajectories": trajectories, "color": ["red", "orange", "yellow", "green", "cyan", "blue", "violet"][i % 7]})
    #
    # # ------------------------------------------------------------
    # # 4. Визуализация
    # # ------------------------------------------------------------
    # # Добавляем линзы
    # for lens in lenses:
    #     plotter.add_mesh(lens.get_mesh(), color="cyan", opacity=0.6, smooth_shading=True)
    #     # Опционально: отрисовка оптической оси и фокусов
    #     lens.draw_axis(plotter, length=15)
    #
    # # Добавляем все лучи
    # for trajectories in all_trajectories:
    #     for traj in trajectories["trajectories"]:
    #         path = pv.PolyData(traj)
    #         path.lines = np.hstack(([len(traj)], range(len(traj))))
    #         plotter.add_mesh(path, color=trajectories["color"], opacity = 0.5, line_width=2, render_lines_as_tubes=True)
    #         pts = pv.PolyData(traj)
    #         plotter.add_mesh(pts, color="purple", point_size=6, render_points_as_spheres=True)
    #
    # # Подпишем линзы (точки над ними)
    # label_points = np.array([
    #     [0, 8, 4],
    #     [0, 4, 4],
    #     [0, 0, 4],
    #     [0, -4, 4],
    #     [0, -8, 4],
    #     [20, 6, 4],
    #     [20, -6, 4]
    # ])
    # labels = [
    #     "Biconvex",
    #     "Biconcave",
    #     "Plano-convex",
    #     "Convex-plano",
    #     "Meniscus",
    #     "Tilted 45°",
    #     "Tilted -30°"
    # ]
    # plotter.add_point_labels(label_points, labels, font_size=10, text_color="white",
    #                          shape=None, show_points=False, point_size=1)
    #
    # plotter.reset_camera()
    # plotter.show()

