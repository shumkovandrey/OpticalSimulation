import random

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as R
from typing import List, Optional, Tuple

import trimesh

from matplotlib.colors import to_rgb

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


def fresnel_coeffs(n1: float, n2: float, cos_i: float) -> Tuple[float, float]:
    """
    Возвращает (коэффициент отражения R, коэффициент пропускания T).
    cos_i – косинус угла падения (положительный, от 0 до 1).
    """
    # sin_i**2
    sin2_i = max(0.0, 1.0 - cos_i * cos_i)
    # sin2_t по закону Снелла
    eta = n1 / n2
    sin2_t = (eta ** 2) * sin2_i

    if sin2_t > 1.0:
        # Полное внутреннее отражение
        return 1.0, 0.0

    cos_t = np.sqrt(1.0 - sin2_t)

    # Амплитудные коэффициенты Френеля
    # s-поляризация
    r_s = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    # p-поляризация
    r_p = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)

    # Интенсивность (коэффициенты отражения)
    R_s = r_s * r_s
    R_p = r_p * r_p
    R = 0.5 * (R_s + R_p)
    T = 1.0 - R
    return R, T


def split_ray(ray, normal, n_next, start_point,
              allow_reflection=True, allow_refraction=True,
              offset_distance=0.01):
    EPS = offset_distance
    cos_i = np.dot(normal, ray.direction)
    if cos_i > 0:
        normal = -normal
        cos_i = np.dot(normal, ray.direction)
    cos_i = -cos_i
    n1, n2 = ray.current_n, n_next

    R, T = fresnel_coeffs(n1, n2, cos_i)

    # Логика переопределения коэффициентов
    if not allow_refraction:
        T = 0.0
        if allow_reflection:
            R = 1.0   # идеальное зеркало
    if not allow_reflection:
        R = 0.0
        if allow_refraction:
            T = 1.0   # идеальное просветление (нет отражения)

    if R == 0.0 and T == 0.0:
        return []

    new_rays = []
    if R > 1e-6:
        reflected_dir = ray.direction - 2 * np.dot(ray.direction, normal) * normal
        reflected_energy = ray.energy * R
        origin = start_point + EPS * reflected_dir
        new_rays.append(Ray(origin, reflected_dir, reflected_energy, n1,
                            color=ray.color, wavelength=ray.wavelength,
                            energy_color_type=ray.energy_color_type))
    if T > 1e-6:
        eta = n1 / n2
        cos_t = np.sqrt(max(0.0, 1.0 - (eta**2) * (1.0 - cos_i**2)))
        refracted_dir = eta * ray.direction + (eta * cos_i - cos_t) * normal
        refracted_dir /= np.linalg.norm(refracted_dir)
        refracted_energy = ray.energy * T
        origin = start_point + EPS * refracted_dir
        new_rays.append(Ray(origin, refracted_dir, refracted_energy, n2,
                            color=ray.color, wavelength=ray.wavelength,
                            energy_color_type=ray.energy_color_type))
    return new_rays


def trace_ray_tree(ray: Ray, elements: List, max_depth: int,
                   min_energy: float = 0.01) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Возвращает список отрезков в виде (p1, p2, energy).
    Глубина ограничена max_depth, лучи с энергией < min_energy отбрасываются.
    """
    segments = []
    _trace_recursive(ray, elements, max_depth, min_energy, segments)
    return segments


def _trace_recursive(ray, elements, depth, min_energy, segments,
                     total_limit=5000, offset_distance=0.01):
    if len(segments) >= total_limit or depth <= 0 or ray.energy < min_energy:
        return

    best_t = float('inf')
    hit_obj = None
    for obj in elements:
        if hasattr(obj, 'is_active') and not obj.is_active(ray.wavelength):
            continue
        t = obj.intersect(ray)
        if t is not None and t < best_t:
            best_t = t
            hit_obj = obj

    if hit_obj is None:
        p2 = ray.origin + ray.direction * 500.0
        segments.append((ray.origin, p2, ray.energy))
        return

    hit_point = ray.origin + ray.direction * best_t
    segments.append((ray.origin, hit_point, ray.energy))

    # Поглощение
    if hasattr(hit_obj, 'absorption_range') and hit_obj.absorption_range is not None:
        if ray.wavelength is None or (hit_obj.absorption_range[0] <= ray.wavelength <= hit_obj.absorption_range[1]):
            return

    # Определяем разрешённые действия
    allow_reflection = False
    allow_refraction = False
    if hasattr(hit_obj, 'reflection_range') and hit_obj.reflection_range is not None:
        if ray.wavelength is None or (hit_obj.reflection_range[0] <= ray.wavelength <= hit_obj.reflection_range[1]):
            allow_reflection = True
    if hasattr(hit_obj, 'refraction_range') and hit_obj.refraction_range is not None:
        if ray.wavelength is None or (hit_obj.refraction_range[0] <= ray.wavelength <= hit_obj.refraction_range[1]):
            allow_refraction = True

    # Ничего не разрешено – проходим сквозь
    if not allow_reflection and not allow_refraction:
        new_ray = Ray(hit_point + offset_distance * ray.direction, ray.direction,
                      energy=ray.energy, current_n=ray.current_n,
                      color=ray.color, wavelength=ray.wavelength,
                      energy_color_type=ray.energy_color_type)
        segments.append((hit_point, new_ray.origin, new_ray.energy))   # соединительный отрезок
        _trace_recursive(new_ray, elements, depth-1, min_energy, segments,
                         total_limit, offset_distance)
        return

    n_next = hit_obj.n if abs(ray.current_n - 1.0) < 1e-6 else 1.0
    new_rays = split_ray(ray, hit_obj.get_normal(hit_point), n_next, hit_point,
                         allow_reflection=allow_reflection,
                         allow_refraction=allow_refraction,
                         offset_distance=offset_distance)
    for new_ray in new_rays:
        # Соединительный отрезок от точки удара до старта нового луча
        segments.append((hit_point, new_ray.origin, new_ray.energy))
        _trace_recursive(new_ray, elements, depth-1, min_energy, segments,
                         total_limit, offset_distance)


# ---------------------
# Классы элементов сцены
# ---------------------


class Ray:
    def __init__(self, origin, direction,
                 energy=1.0, current_n=1.0,
                 color="yellow",
                 energy_color_type=2,
                 wavelength=None):          # новая опция
        self.origin = np.array(origin, dtype=float)
        self.direction = np.array(direction, dtype=float)
        self.direction /= np.linalg.norm(self.direction)
        self.energy = energy
        self.current_n = current_n
        self.color = color
        self.energy_color_type = energy_color_type
        self.wavelength = wavelength        # например, 550 (нм)


class RayCloud:
    """
    Единый актор для множества отрезков.
    Гибкая настройка прозрачности по энергии.

    Параметры:
        plotter: pv.Plotter
        energy_color_type: int (0, 1, 2) – как энергия влияет на непрозрачность.
                           0 – не используется (opacity=1).
                           1 – opacity = energy.
                           2 – opacity = max(min_alpha, energy ** gamma).
        default_color: цвет по умолчанию (строка или RGB-кортеж).
        line_width: толщина линий.
        min_alpha: минимальная непрозрачность для режима 2 (по умолчанию 0.05).
        gamma: показатель степени для режима 2 (по умолчанию 0.3).
    """
    def __init__(self, plotter: pv.Plotter,
                 energy_color_type: int = 1,
                 default_color = "yellow",
                 line_width: float = 2.0,
                 min_alpha: float = 0.05,
                 gamma: float = 0.3):
        self.plotter = plotter
        self.energy_color_type = energy_color_type
        self.default_color = default_color
        self.line_width = line_width
        self.min_alpha = min_alpha
        self.gamma = gamma

        # Временный меш для инициализации
        temp_points = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        temp_mesh = pv.PolyData(temp_points)
        # Всегда используем RGBA-актор, чтобы можно было менять прозрачность
        temp_mesh.point_data["colors"] = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        self.actor = plotter.add_mesh(
            temp_mesh,
            scalars="colors",
            rgba=True,
            line_width=line_width,
            render_lines_as_tubes=False,
            name="RayCloud"
        )

    @staticmethod
    def _to_rgb(color) -> np.ndarray:
        """Приводит цвет (строка или RGB-кортеж) к массиву RGB из трёх float."""
        from matplotlib.colors import to_rgb
        return np.array(to_rgb(color), dtype=np.float32)

    def _energy_to_alpha(self, energy: float) -> float:
        """Вычисляет непрозрачность по энергии в зависимости от energy_color_type."""
        if self.energy_color_type == 0:
            return 1.0
        elif self.energy_color_type == 1:
            return np.clip(energy, 0.0, 1.0)
        elif self.energy_color_type == 2:
            return max(self.min_alpha, energy ** self.gamma)
        else:
            return 1.0

    def _build_rgba(self, color, alpha: float) -> np.ndarray:
        """Создаёт массив RGBA из цвета и альфа-канала."""
        rgb = self._to_rgb(color)
        return np.array([*rgb, alpha], dtype=np.float32)

    # ---------- Методы обновления ----------

    def update_from_trajectories(self, trajectories, colors=None):
        if not trajectories:
            self.actor.mapper.dataset.copy_from(pv.PolyData())
            return

        n_rays = len(trajectories)
        if colors is None:
            colors = [self.default_color] * n_rays
        elif len(colors) != n_rays:
            raise ValueError("colors length must match number of trajectories")

        points = []
        lines = []
        offset = 0
        rgba_list = []

        for traj, color in zip(trajectories, colors):
            n = len(traj)
            if n < 2:
                continue
            # Проверка формы точек
            if any(p.shape != (3,) for p in traj):
                print("⚠️ Пропущена траектория с некорректными точками")
                continue
            points.extend(traj)
            lines.append(np.hstack([n, np.arange(offset, offset + n)]))
            rgba = self._build_rgba(color, 1.0)
            rgba_list.extend([rgba] * n)
            offset += n

        if not points:
            self.actor.mapper.dataset.copy_from(pv.PolyData())
            return

        points = np.array(points, dtype=np.float32)
        lines = np.hstack(lines).astype(int)
        new_mesh = pv.PolyData(points, lines=lines)
        new_mesh.point_data["colors"] = np.array(rgba_list, dtype=np.float32)
        new_mesh.active_scalars_name = "colors"

        self.actor.mapper.dataset.copy_from(new_mesh)
        self.actor.mapper.SetColorModeToDirectScalars()

    def update_from_segments(self, segments: list,
                             base_colors=None,
                             energy_types=None):
        if not segments:
            self.actor.mapper.dataset.copy_from(pv.PolyData())
            return

        if energy_types is not None and len(energy_types) != len(segments):
            raise ValueError("energy_types length must match segments")

        points = []
        lines = []
        offset = 0
        rgba_list = []

        # Проверка и фильтрация битых сегментов
        valid_segments = []
        valid_colors = []
        valid_types = []
        for i, seg in enumerate(segments):
            p1, p2, energy = seg
            if p1.shape != (3,) or p2.shape != (3,):
                continue
            if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
                continue
            valid_segments.append(seg)
            if base_colors:
                valid_colors.append(base_colors[i])
            if energy_types:
                valid_types.append(energy_types[i])

        if not valid_segments:
            self.actor.mapper.dataset.copy_from(pv.PolyData())
            return

        for i, (p1, p2, energy) in enumerate(valid_segments):
            points.append(p1)
            points.append(p2)
            lines.append([2, offset, offset + 1])

            # Определяем цвет
            color = valid_colors[i] if valid_colors else self.default_color

            # Определяем тип затухания (индивидуальный или глобальный)
            etype = valid_types[i] if valid_types else self.energy_color_type

            # Вычисляем непрозрачность по нужному закону
            if etype == 0:
                alpha = 1.0
            elif etype == 1:
                alpha = np.clip(energy, 0.0, 1.0)
            elif etype == 2:
                alpha = max(self.min_alpha, energy ** self.gamma)
            else:
                alpha = 1.0

            rgba = self._build_rgba(color, alpha)
            rgba_list.extend([rgba, rgba])
            offset += 2

        points = np.array(points, dtype=np.float32)
        lines = np.hstack(lines).astype(int)
        new_mesh = pv.PolyData(points, lines=lines)
        new_mesh.point_data["colors"] = np.array(rgba_list, dtype=np.float32)
        new_mesh.active_scalars_name = "colors"

        self.actor.mapper.dataset.copy_from(new_mesh)
        self.actor.mapper.SetColorModeToDirectScalars()


class RayTracer:
    def __init__(self, mode='tree', max_depth=6, min_energy=0.01, offset_distance=0.5):
        if mode not in ('simple', 'tree'):
            raise ValueError("mode must be 'simple' or 'tree'")
        self.mode = mode
        self.max_depth = max_depth
        self.min_energy = min_energy
        self.offset_distance = offset_distance
        self.rays = []
        self.elements = []

    def add_ray(self, ray: Ray):
        self.rays.append(ray)

    def add_element(self, element):
        self.elements.append(element)

    def trace_all(self):
        if self.mode == 'simple':
            trajectories, colors = [], []
            for ray in self.rays:
                traj = trace_ray(ray, self.elements, mode='simple',
                                 max_depth=self.max_depth,
                                 offset_distance=self.offset_distance)
                trajectories.append(traj)
                colors.append(ray.color)
            return trajectories, colors, None
        else:  # tree
            all_segments, all_colors, all_types = [], [], []
            for ray in self.rays:
                segs = trace_ray(ray, self.elements, mode='tree',
                                 max_depth=self.max_depth,
                                 min_energy=self.min_energy,
                                 offset_distance=self.offset_distance)
                all_segments.extend(segs)
                all_colors.extend([ray.color] * len(segs))
                all_types.extend([ray.energy_color_type] * len(segs))
            return all_segments, all_colors, all_types

    def render(self, plotter, **cloud_kwargs):
        result, colors, types = self.trace_all()
        if self.mode == 'simple':
            cloud = RayCloud(plotter, energy_color_type=0, **cloud_kwargs)
            cloud.update_from_trajectories(result, colors=colors)
        else:
            cloud = RayCloud(plotter, energy_color_type=2, **cloud_kwargs)
            cloud.update_from_segments(result, base_colors=colors, energy_types=types)
        return cloud


class PlaneSurface:
    def __init__(self, point, normal, n_inside,
                 lens_origin, lens_axis, edge_radius,
                 half_sizes=None, face_tangents=None, reflection_range=None, refraction_range=None,
                 absorption_range=None):
        self.point = np.array(point, dtype=float)
        self.normal = np.array(normal, dtype=float)
        self.normal /= np.linalg.norm(self.normal)
        self.n = n_inside
        self.lens_origin = np.array(lens_origin, dtype=float)
        self.lens_axis = np.array(lens_axis, dtype=float)
        self.lens_axis /= np.linalg.norm(self.lens_axis)
        self.edge_radius = edge_radius
        # Прямоугольная апертура (опционально)
        self.half_sizes = half_sizes
        self.face_tangents = face_tangents

        # Диапазоны отражения, преломления, поглощения
        self.reflection_range = reflection_range
        self.refraction_range = refraction_range
        self.absorption_range = absorption_range

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

    # def interact(self, ray_dir, normal, n1, n2):
    #     if self.is_mirror:
    #         dot = np.dot(normal, ray_dir)
    #         actual_normal = normal if dot < 0 else -normal
    #         reflected_dir = ray_dir - 2 * np.dot(ray_dir, actual_normal) * actual_normal
    #         return reflected_dir / np.linalg.norm(reflected_dir)
    #     return refract(ray_dir, normal, n1, n2)

    def is_active(self, wavelength):
        """Возвращает True, если поверхность должна взаимодействовать с данной длиной волны."""
        if wavelength is None:
            return True
        # Если ни один диапазон не задан, объект невидим (прозрачен) – нет взаимодействия
        if self.reflection_range is None and self.refraction_range is None and self.absorption_range is None:
            return False
        # Проверяем попадание хотя бы в один диапазон
        in_ref = self.reflection_range is not None and (self.reflection_range[0] <= wavelength <= self.reflection_range[1])
        in_refr = self.refraction_range is not None and (self.refraction_range[0] <= wavelength <= self.refraction_range[1])
        in_abs = self.absorption_range is not None and (self.absorption_range[0] <= wavelength <= self.absorption_range[1])
        return in_ref or in_refr or in_abs


class SphereSurface:
    """
    Сферическая поверхность с ограничениями по радиусу апертуры и продольному
    положению (толщине) вдоль оптической оси линзы.
    """
    def __init__(self, center: np.ndarray, radius: float, n_inside: float,
                 lens_origin: np.ndarray, lens_axis: np.ndarray,
                 edge_radius: float, thickness: float, reflection_range=None, refraction_range=None,
                 absorption_range=None):
        self.center = np.array(center, dtype=float)
        self.radius = radius
        self.n = n_inside
        self.lens_origin = np.array(lens_origin, dtype=float)
        self.lens_axis = np.array(lens_axis, dtype=float)
        self.lens_axis /= np.linalg.norm(self.lens_axis)
        self.edge_radius = edge_radius
        self.thickness = thickness

        # Диапазоны отражения, преломления, поглощения
        self.reflection_range = reflection_range
        self.refraction_range = refraction_range
        self.absorption_range = absorption_range

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

    # def interact(self, ray_dir: np.ndarray, normal: np.ndarray, n1: float, n2: float) -> Optional[np.ndarray]:
    #     if self.is_mirror:
    #         # Находим косинус угла падения
    #         dot = np.dot(normal, ray_dir)
    #
    #         # ГЛАВНЫЙ СЕКРЕТ: нормаль должна всегда смотреть НАВСТРЕЧУ лучу
    #         # Если dot > 0, значит нормаль и луч смотрят в одну сторону -> разворачиваем нормаль
    #         actual_normal = normal if dot < 0 else -normal
    #
    #         # Формула отражения
    #         reflected_dir = ray_dir - 2 * np.dot(ray_dir, actual_normal) * actual_normal
    #         return reflected_dir / np.linalg.norm(reflected_dir)
    #     return refract(ray_dir, normal, n1, n2)

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

    def is_active(self, wavelength):
        """Возвращает True, если поверхность должна взаимодействовать с данной длиной волны."""
        if wavelength is None:
            return True
        # Если ни один диапазон не задан, объект невидим (прозрачен) – нет взаимодействия
        if self.reflection_range is None and self.refraction_range is None and self.absorption_range is None:
            return False
        # Проверяем попадание хотя бы в один диапазон
        in_ref = self.reflection_range is not None and (self.reflection_range[0] <= wavelength <= self.reflection_range[1])
        in_refr = self.refraction_range is not None and (self.refraction_range[0] <= wavelength <= self.refraction_range[1])
        in_abs = self.absorption_range is not None and (self.absorption_range[0] <= wavelength <= self.absorption_range[1])
        return in_ref or in_refr or in_abs


class MeshSurface:
    """
    Произвольная треугольная поверхность, загружаемая из файла или создаваемая из меша.
    Может быть зеркальной, преломляющей или поглощающей.
    """
    def __init__(self, mesh,  # может быть путём к файлу или готовым trimesh.Trimesh / pv.PolyData
                 n_inside: float = 1.0, reflection_range=None, refraction_range=None,
                 absorption_range=None):
        # Загрузка, если передан путь
        if isinstance(mesh, str):
            self.trimesh_obj = trimesh.load(mesh)
            if isinstance(self.trimesh_obj, trimesh.Scene):
                self.trimesh_obj = trimesh.util.concatenate(
                    [g for g in self.trimesh_obj.geometry.values()
                     if isinstance(g, trimesh.Trimesh)]
                )
            if not isinstance(self.trimesh_obj, trimesh.Trimesh):
                raise TypeError("Файл не содержит треугольной сетки")
        elif isinstance(mesh, trimesh.Trimesh):
            self.trimesh_obj = mesh
        elif isinstance(mesh, pv.PolyData):
            # Конвертируем PyVista → trimesh
            verts = mesh.points
            faces = mesh.faces.reshape(-1, 4)[:, 1:4]
            self.trimesh_obj = trimesh.Trimesh(vertices=verts, faces=faces)
        else:
            raise TypeError("mesh должен быть str, trimesh.Trimesh или pv.PolyData")

        self.mesh = self.trimesh_obj
        # Ускоритель для пересечений
        self.intersector = trimesh.ray.ray_triangle.RayMeshIntersector(self.mesh)

        self.n = n_inside

        # Кэш последнего пересечения
        self._last_ray_origin = None
        self._last_ray_direction = None
        self._last_hit_triangle_idx = None

        # Диапазоны отражения, преломления, поглощения
        self.reflection_range = reflection_range
        self.refraction_range = refraction_range
        self.absorption_range = absorption_range


    def intersect(self, ray: Ray) -> Optional[float]:
        """
        Возвращает параметр t для ближайшего пересечения > 0,
        либо None, если пересечений нет.
        """
        origins = np.array([ray.origin])
        directions = np.array([ray.direction])

        # Ищем все пересечения, trimesh возвращает их уже отсортированными по расстоянию
        locations, _, tri_indices = self.intersector.intersects_location(
            origins, directions, multiple_hits=False
        )

        if len(locations) == 0:
            return None

        hit_point = locations[0]
        t = np.linalg.norm(hit_point - ray.origin)

        if t <= 1e-6:
            return None

        # Сохраняем для get_normal
        self._last_ray_origin = ray.origin.copy()
        self._last_ray_direction = ray.direction.copy()
        self._last_hit_triangle_idx = tri_indices[0]
        return t

    def get_normal(self, point: np.ndarray) -> np.ndarray:
        """
        Возвращает нормаль к поверхности в точке пересечения.
        Использует нормаль треугольника из последнего пересечения.
        """
        if self._last_hit_triangle_idx is not None:
            # Берём нормаль треугольника, который был пересечён
            normal = self.mesh.face_normals[self._last_hit_triangle_idx]
        else:
            # Запасной вариант: ищем ближайший треугольник
            _, _, tri_idx = trimesh.proximity.closest_point(self.mesh, [point])
            normal = self.mesh.face_normals[tri_idx[0]]

        # Нормализуем
        return normal / np.linalg.norm(normal)

    def interact(self, ray_dir: np.ndarray, normal: np.ndarray, n1: float, n2: float) -> Optional[np.ndarray]:
        """
        Определяет поведение луча при столкновении:
          - зеркало: отражение
          - экран/поглотитель: остановка (возвращает None)
          - иначе: преломление (вызывается refract)
        """
        if self.is_screen:
            return None  # луч поглощён

        if self.is_mirror:
            # Отражение (нормаль всегда направлена против луча)
            dot = np.dot(normal, ray_dir)
            actual_normal = normal if dot < 0 else -normal
            reflected_dir = ray_dir - 2 * np.dot(ray_dir, actual_normal) * actual_normal
            return reflected_dir / np.linalg.norm(reflected_dir)

        # Преломление
        return refract(ray_dir, normal, n1, n2)

    def get_mesh(self) -> pv.PolyData:
        """
        Возвращает PyVista меш для визуализации.
        """
        verts = self.mesh.vertices
        faces = np.hstack([np.full((len(self.mesh.faces), 1), 3),
                           self.mesh.faces])
        return pv.PolyData(verts, faces)

    def is_active(self, wavelength):
        """Возвращает True, если поверхность должна взаимодействовать с данной длиной волны."""
        if wavelength is None:
            return True
        # Если ни один диапазон не задан, объект невидим (прозрачен) – нет взаимодействия
        if self.reflection_range is None and self.refraction_range is None and self.absorption_range is None:
            return False
        # Проверяем попадание хотя бы в один диапазон
        in_ref = self.reflection_range is not None and (self.reflection_range[0] <= wavelength <= self.reflection_range[1])
        in_refr = self.refraction_range is not None and (self.refraction_range[0] <= wavelength <= self.refraction_range[1])
        in_abs = self.absorption_range is not None and (self.absorption_range[0] <= wavelength <= self.absorption_range[1])
        return in_ref or in_refr or in_abs


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


class RectangularMirror(PlaneSurface):
    def __init__(self, center: np.ndarray, normal: np.ndarray,
                 width: float, height: float,
                 n_inside: float = 1.0,
                 reflection_range=(0, 10000)):   # ← теперь диапазон
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
            edge_radius=0.0,
            half_sizes=(half_u, half_v),
            face_tangents=(tangent1, tangent2),
            reflection_range=reflection_range   # зеркало
        )


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
                face_tangents=tangents,
                reflection_range = (0, np.inf), refraction_range = (0, np.inf)
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
                lens_origin=self.origin, lens_axis=self.axis_dir, edge_radius=edge_radius, reflection_range=(0, np.inf), refraction_range=(0, np.inf)
            )
        else:
            c1_world = self.origin + self.rotation @ np.array([v1_local + R1, 0, 0])
            self.front = SphereSurface(
                center=c1_world, radius=abs(R1), n_inside=n,
                lens_origin=self.origin, lens_axis=self.axis_dir,
                edge_radius=edge_radius, thickness=thickness,
                reflection_range=(0, np.inf), refraction_range=(0, np.inf)
            )

        # Задняя поверхность
        if R2 is None:
            p2_world = self.origin + self.rotation @ np.array([v2_local, 0, 0])
            n2_world = self.rotation @ np.array([1, 0, 0])
            self.back = PlaneSurface(
                point=p2_world, normal=n2_world, n_inside=n,
                lens_origin=self.origin, lens_axis=self.axis_dir, edge_radius=edge_radius, reflection_range=(0, np.inf), refraction_range=(0, np.inf)
            )
        else:
            c2_world = self.origin + self.rotation @ np.array([v2_local - R2, 0, 0])
            self.back = SphereSurface(
                center=c2_world, radius=abs(R2), n_inside=n,
                lens_origin=self.origin, lens_axis=self.axis_dir,
                edge_radius=edge_radius, thickness=thickness,
                reflection_range=(0, np.inf), refraction_range=(0, np.inf)
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

        # Слишком далеко – диафрагма не пересечена
        if dist_to_axis > self.outer_radius + 1e-6:
            return None

        # Попадание в отверстие или в непрозрачную часть?
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


def _trace_simple(ray: Ray, elements: List, max_bounces: int,
                  offset_distance: float) -> np.ndarray:
    path = [ray.origin]
    current_ray = ray
    current_n = ray.current_n

    for _ in range(max_bounces):
        best_t = float('inf')
        hit_obj = None
        for obj in elements:
            if hasattr(obj, 'is_active') and not obj.is_active(current_ray.wavelength):
                continue
            t = obj.intersect(current_ray)
            if t is not None and t < best_t:
                best_t = t
                hit_obj = obj

        if hit_obj is None:
            path.append(current_ray.origin + current_ray.direction * 500)
            break

        hit_point = current_ray.origin + current_ray.direction * best_t
        path.append(hit_point)

        # Поглощение?
        if hasattr(hit_obj, 'absorption_range') and hit_obj.absorption_range is not None:
            if current_ray.wavelength is None or (hit_obj.absorption_range[0] <= current_ray.wavelength <= hit_obj.absorption_range[1]):
                break

        # Разрешённые действия
        allow_reflection = False
        allow_refraction = False
        if hasattr(hit_obj, 'reflection_range') and hit_obj.reflection_range is not None:
            if current_ray.wavelength is None or (hit_obj.reflection_range[0] <= current_ray.wavelength <= hit_obj.reflection_range[1]):
                allow_reflection = True
        if hasattr(hit_obj, 'refraction_range') and hit_obj.refraction_range is not None:
            if current_ray.wavelength is None or (hit_obj.refraction_range[0] <= current_ray.wavelength <= hit_obj.refraction_range[1]):
                allow_refraction = True

        # Ничего не разрешено – проходим сквозь
        if not allow_reflection and not allow_refraction:
            current_ray = Ray(hit_point + offset_distance * current_ray.direction,
                              current_ray.direction,
                              energy=current_ray.energy, current_n=current_n,
                              color=current_ray.color,
                              wavelength=current_ray.wavelength,
                              energy_color_type=current_ray.energy_color_type)
            continue

        # Нормаль для отражения/преломления
        normal = hit_obj.get_normal(hit_point)
        dot = np.dot(normal, current_ray.direction)
        actual_normal = normal if dot < 0 else -normal

        if allow_refraction:
            n_next = hit_obj.n if abs(current_n - 1.0) < 1e-6 else 1.0
            refracted_dir = refract(current_ray.direction, normal, current_n, n_next)
            if refracted_dir is not None:
                current_n = n_next
                new_dir = refracted_dir
            else:  # полное внутреннее отражение
                allow_reflection = True
                allow_refraction = False
                # принудительно отражаем

        if allow_reflection:
            new_dir = (current_ray.direction - 2 * np.dot(current_ray.direction, actual_normal) * actual_normal)
            new_dir /= np.linalg.norm(new_dir)

        current_ray = Ray(hit_point + offset_distance * new_dir, new_dir,
                          energy=current_ray.energy, current_n=current_n,
                          color=current_ray.color,
                          wavelength=current_ray.wavelength,
                          energy_color_type=current_ray.energy_color_type)

    return np.array(path)


def trace_ray(ray: Ray, elements: List, mode: str = 'tree',
              max_depth: int = 10, min_energy: float = 0.01,
              offset_distance: float = 0.5):
    """
    Универсальная трассировка луча.

    mode:
        'simple' – без разделения энергии (аналог run_simulation).
        'tree'   – с разделением и учётом энергии (аналог trace_ray_tree).

    Возвращает:
        при mode='simple': массив точек траектории (np.ndarray)
        при mode='tree'  : список отрезков (p1, p2, energy)
    """
    if mode == 'simple':
        return _trace_simple(ray, elements, max_depth, offset_distance)
    elif mode == 'tree':
        segments = []
        _trace_recursive(ray, elements, max_depth, min_energy, segments,
                         total_limit=5000, offset_distance=offset_distance)
        return segments
    else:
        raise ValueError("mode must be 'simple' or 'tree'")


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


def update_rays(trajectories):
    """trajectories – список np.array траекторий (последовательностей точек)"""
    # Собираем все точки и индексы линий
    points = []
    lines = []
    offset = 0
    for traj in trajectories:
        n = len(traj)
        points.extend(traj)
        lines.append(np.hstack([n, np.arange(offset, offset + n)]))
        offset += n

    if not points:
        return
    points = np.array(points)
    lines = np.hstack(lines).astype(int)

    # Создаём новый PolyData и копируем в существующий
    new_pd = pv.PolyData(points, lines=lines)
    # Быстрое обновление (без удаления актора)
    plotter.actors["rays_actor"].mapper.dataset.copy_from(new_pd)


plotter = pv.Plotter()
plotter.set_background("black")
plotter.view_isometric()
plotter.enable_parallel_projection()
plotter.enable_terrain_style(mouse_wheel_zooms=True)
plotter.view_vector((0, 0, 1), viewup=(0, 1, 0))
plotter.add_axes(color="white")
