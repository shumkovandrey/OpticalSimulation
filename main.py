import random

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as R
from typing import List, Optional, Tuple

from numba import njit
from fast_math import *

import trimesh

from matplotlib.colors import to_rgb

pv.global_theme.allow_empty_mesh = True
# Глобальная константа – длина отрезка, которым луч уходит в бесконечность
RAY_INFINITY_DISTANCE = 100

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
    normal = np.array(normal, dtype=float)  # <-- гарантирует float
    normal /= np.linalg.norm(normal)
    if abs(normal[0]) < 0.9:
        arbitrary = np.array([1.0, 0.0, 0.0])
    else:
        arbitrary = np.array([0.0, 1.0, 0.0])
    t1 = np.cross(normal, arbitrary)
    t1 = t1.astype(float) / np.linalg.norm(t1)  # явное преобразование
    t2 = np.cross(normal, t1)
    t2 = t2.astype(float) / np.linalg.norm(t2)
    return t1, t2


# def fresnel_amplitudes(n1, n2, cos_i):
#     """
#     Возвращает комплексные амплитудные коэффициенты для s и p поляризаций:
#     (r_s, r_p, t_s, t_p).
#     cos_i – положительный косинус угла падения.
#     """
#     eta = n1 / n2
#     sin2_i = max(0.0, 1.0 - cos_i*cos_i)
#     sin2_t = (eta**2) * sin2_i
#     if sin2_t > 1.0:   # полное внутреннее отражение
#         # r_s = 1, r_p = 1 (фаза сдвигается, но для амплитуд модуль 1)
#         # t_s = 0, t_p = 0
#         return 1.0+0j, 1.0+0j, 0.0+0j, 0.0+0j
#     cos_t = np.sqrt(1.0 - sin2_t)
#
#     # Амплитудные коэффициенты (действительные для диэлектриков без поглощения)
#     r_s = (n1*cos_i - n2*cos_t) / (n1*cos_i + n2*cos_t)
#     r_p = (n2*cos_i - n1*cos_t) / (n2*cos_i + n1*cos_t)
#     t_s = 2.0 * n1 * cos_i / (n1*cos_i + n2*cos_t)
#     t_p = 2.0 * n1 * cos_i / (n2*cos_i + n1*cos_t)
#     # Фазы для проходящих волн? Обычно t положительные для прозрачных сред.
#     return r_s, r_p, t_s, t_p
#
# def fresnel_coeffs(n1, n2, cos_i):
#     r_s, r_p, _, _ = fresnel_amplitudes(n1, n2, cos_i)
#     R = 0.5 * (abs(r_s)**2 + abs(r_p)**2)
#     T = 1.0 - R
#     return R, T


def split_ray(ray: Ray, normal: np.ndarray, n_next: float, start_point: np.ndarray,
              allow_reflection: bool = True, allow_refraction: bool = True,
              offset_distance: float = 0.01,
              use_polarization_color: bool = False) -> List[Ray]:
    EPS = offset_distance

    cos_i = np.dot(normal, ray.direction)
    if cos_i > 0:
        normal = -normal
        cos_i = np.dot(normal, ray.direction)
    cos_i = -cos_i

    n1, n2 = ray.current_n, n_next

    # Локальный базис плоскости падения
    s_dir = np.cross(normal, ray.direction)
    if np.linalg.norm(s_dir) < 1e-10:
        s_dir = np.array([0.0, 1.0, 0.0])
    s_dir /= np.linalg.norm(s_dir)
    p_dir = np.cross(ray.direction, s_dir)
    p_dir /= np.linalg.norm(p_dir)

    if ray.polarization is not None:
        E_s = np.dot(ray.polarization, s_dir)
        E_p = np.dot(ray.polarization, p_dir)
    else:
        E_s = complex(1.0, 0.0)
        E_p = complex(0.0, 0.0)

    r_s, r_p, t_s, t_p = fresnel_amplitudes(n1, n2, cos_i)

    if not allow_reflection and not allow_refraction:
        return []
    if allow_reflection and not allow_refraction:
        r_s, r_p = -1.0 + 0j, 1.0 + 0j
        t_s, t_p = 0j, 0j
    elif not allow_reflection and allow_refraction:
        t_s, t_p = 1.0 + 0j, 1.0 + 0j
        r_s, r_p = 0j, 0j

    new_rays = []

    # Отражённый луч
    if allow_reflection and (abs(r_s) > 1e-9 or abs(r_p) > 1e-9):
        new_E_s = r_s * E_s
        new_E_p = r_p * E_p
        energy = abs(new_E_s)**2 + abs(new_E_p)**2
        if energy > 1e-9:
            reflected_dir = ray.direction - 2 * np.dot(ray.direction, normal) * normal
            new_pol = new_E_s * s_dir + new_E_p * p_dir
            new_ray = Ray(start_point + EPS * reflected_dir, reflected_dir, energy, n1,
                          color=ray.color, wavelength=ray.wavelength,
                          energy_color_type=ray.energy_color_type,
                          polarization=new_pol)
            if use_polarization_color:
                new_ray.update_color_from_polarization()
                print(f"Polarization color: {new_ray.color} for ray with energy {energy:.3f}")
            new_rays.append(new_ray)

    # Преломлённый луч
    if allow_refraction and (abs(t_s) > 1e-9 or abs(t_p) > 1e-9):
        new_E_s = t_s * E_s
        new_E_p = t_p * E_p
        energy = abs(new_E_s)**2 + abs(new_E_p)**2
        if energy > 1e-9:
            eta = n1 / n2
            cos_t = np.sqrt(max(0.0, 1.0 - (eta**2) * (1.0 - cos_i**2)))
            refracted_dir = eta * ray.direction + (eta * cos_i - cos_t) * normal
            refracted_dir /= np.linalg.norm(refracted_dir)
            new_pol = new_E_s * s_dir + new_E_p * p_dir
            new_ray = Ray(start_point + EPS * refracted_dir, refracted_dir, energy, n2,
                          color=ray.color, wavelength=ray.wavelength,
                          energy_color_type=ray.energy_color_type,
                          polarization=new_pol)
            if use_polarization_color:
                new_ray.update_color_from_polarization()
                print(f"Polarization color: {new_ray.color} for ray with energy {energy:.3f}")
            new_rays.append(new_ray)

    return new_rays


def trace_ray_tree(ray: Ray, elements: List, max_depth: int,
                   min_energy: float = 0.01, use_polarization_color=False) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Возвращает список отрезков в виде (p1, p2, energy).
    Глубина ограничена max_depth, лучи с энергией < min_energy отбрасываются.
    """
    segments = []
    _trace_recursive(ray, elements, max_depth, min_energy, segments, use_polarization_color=use_polarization_color)
    return segments


def _trace_recursive(ray, elements, depth, min_energy, segments,
                     total_limit=5000, offset_distance=0.01, use_polarization_color=False):
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
        p2 = ray.origin + ray.direction * RAY_INFINITY_DISTANCE
        segments.append((ray.origin, p2, ray.energy, ray.color))
        return

    hit_point = ray.origin + ray.direction * best_t
    segments.append((ray.origin, hit_point, ray.energy, ray.color))

    if isinstance(hit_obj, ThinLens):
        new_dir = hit_obj.thin_lens_deflection(ray.direction, hit_point)
        new_ray = Ray(hit_point + offset_distance * new_dir, new_dir,
                      energy=ray.energy, current_n=ray.current_n,
                      color=ray.color, wavelength=ray.wavelength,
                      energy_color_type=ray.energy_color_type)
        segments.append((hit_point, new_ray.origin, new_ray.energy, ray.color))
        _trace_recursive(new_ray, elements, depth - 1, min_energy, segments,
                         total_limit, offset_distance, use_polarization_color=use_polarization_color)
        return

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
        segments.append((hit_point, new_ray.origin, new_ray.energy, ray.color))   # соединительный отрезок
        _trace_recursive(new_ray, elements, depth-1, min_energy, segments,
                         total_limit, offset_distance, use_polarization_color=use_polarization_color)
        return

    n_next = hit_obj.n if abs(ray.current_n - 1.0) < 1e-6 else 1.0
    new_rays = split_ray(ray, hit_obj.get_normal(hit_point), n_next, hit_point,
                         allow_reflection=allow_reflection,
                         allow_refraction=allow_refraction,
                         offset_distance=offset_distance,
                         use_polarization_color=use_polarization_color)
    for new_ray in new_rays:
        # Соединительный отрезок от точки удара до старта нового луча
        segments.append((hit_point, new_ray.origin, new_ray.energy, ray.color))
        _trace_recursive(new_ray, elements, depth-1, min_energy, segments,
                         total_limit, offset_distance, use_polarization_color=use_polarization_color)


# ---------------------
# Классы элементов сцены
# ---------------------


class Ray:
    def __init__(self, origin, direction,
                 energy=1.0, current_n=1.0,
                 color="yellow",
                 energy_color_type=2,
                 wavelength=None,
                 polarization=None):          # ← теперь трёхмерный комплексный вектор
        self.origin = np.array(origin, dtype=float)
        self.direction = np.array(direction, dtype=float)
        self.direction /= np.linalg.norm(self.direction)
        self.energy = energy
        self.current_n = current_n
        self.color = color
        self.energy_color_type = energy_color_type
        self.wavelength = wavelength
        if polarization is not None:
            self.polarization = np.array(polarization, dtype=complex)
        else:
            self.polarization = None

    def update_color_from_polarization(self):
        if self.polarization is None:
            return
        E = self.polarization
        # Глобальные оси: Y – p-компонента, Z – s-компонента
        I_y = abs(E[1])**2
        I_z = abs(E[2])**2
        total = I_y + I_z
        if total < 1e-9:
            self.color = (1.0, 1.0, 1.0)
        else:
            r = I_y / total
            b = I_z / total
            self.color = (r, 0.0, b)


class RayPool:
    """Пул переиспользуемых лучей для снижения аллокаций."""
    def __init__(self, initial_size=100):
        self.pool = [self._create_blank() for _ in range(initial_size)]

    def _create_blank(self):
        return Ray(np.zeros(3), np.zeros(3))

    def acquire(self, origin, direction, energy=1.0, current_n=1.0,
                color="yellow", energy_color_type=2, wavelength=None,
                polarization=None):
        """Взять луч из пула и инициализировать поля."""
        if self.pool:
            ray = self.pool.pop()
        else:
            ray = Ray(np.zeros(3), np.zeros(3))
        # Заполняем все атрибуты
        ray.origin[:] = origin
        ray.direction[:] = direction
        ray.direction /= np.linalg.norm(ray.direction)
        ray.energy = energy
        ray.current_n = current_n
        ray.color = color
        ray.energy_color_type = energy_color_type
        ray.wavelength = wavelength
        if polarization is not None:
            ray.polarization = np.array(polarization, dtype=complex)
        else:
            ray.polarization = None
        return ray

    def release(self, ray: Ray):
        """Вернуть луч в пул."""
        self.pool.append(ray)


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

        # Временная точка, чтобы PyVista не ругался на пустой меш
        temp_points = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        temp_mesh = pv.PolyData(temp_points)
        temp_mesh.point_data["colors"] = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        self.actor = plotter.add_mesh(
            temp_mesh,
            scalars="colors",
            rgba=True,
            line_width=line_width,
            render_lines_as_tubes=False,
            name="RayCloud"
        )
        # Сразу очищаем геометрию – точка не будет видна
        self.actor.mapper.dataset.copy_from(pv.PolyData())

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
            p1, p2, energy = seg[0], seg[1], seg[2]
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

        for i, seg in enumerate(valid_segments):
            if len(seg) == 4:
                p1, p2, energy, color = seg
            else:  # обратная совместимость
                p1, p2, energy = seg
                color = None
            points.append(p1)
            points.append(p2)
            lines.append([2, offset, offset + 1])

            # Определяем цвет
            color = color if color is not None else (valid_colors[i] if valid_colors else self.default_color)

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
    def __init__(self, plotter, mode='tree', max_depth=6, min_energy=0.01,
                 offset_distance=0.5, use_polarization_color=False, pool=None, **cloud_kwargs):
        self.plotter = plotter
        self.mode = mode
        self.max_depth = max_depth
        self.min_energy = min_energy
        self.offset_distance = offset_distance
        self.rays = []
        self.elements = []
        self.emitters = []
        self.use_polarization_color = use_polarization_color
        self.pool = pool
        # Один‑единственный RayCloud на всё время жизни трейсера
        self.cloud = RayCloud(plotter, **cloud_kwargs)

    def add_ray(self, ray):
        self.rays.append(ray)

    def add_elements(self, *elements):
        for element in elements:
            self.elements.append(element)

    def add_emitter(self, emitter: BeamEmitter):
        if not hasattr(self, 'emitters'):
            self.emitters = []
        self.emitters.append(emitter)

    def set_mode(self, mode):
        if mode not in ('simple', 'tree'):
            raise ValueError("mode must be 'simple' or 'tree'")
        self.mode = mode

    def trace_all(self):
        if hasattr(self, 'emitters'):
            for emitter in self.emitters:
                for ray in emitter.emit():
                    self.rays.append(ray)

        if self.mode == 'simple':
            trajectories, colors = [], []
            for ray in self.rays:
                traj = trace_ray(ray, self.elements, mode='simple',
                                 max_depth=self.max_depth,
                                 offset_distance=self.offset_distance, use_polarization_color=self.use_polarization_color)
                trajectories.append(traj)
                colors.append(ray.color)
            return trajectories, colors, None
        else:  # tree
            all_segments, all_colors, all_types = [], [], []
            for ray in self.rays:
                segs = trace_ray(ray, self.elements, mode='tree',
                                 max_depth=self.max_depth,
                                 min_energy=self.min_energy,
                                 offset_distance=self.offset_distance, use_polarization_color=self.use_polarization_color)
                all_segments.extend(segs)
                all_colors.extend([ray.color] * len(segs))
                all_types.extend([ray.energy_color_type] * len(segs))
            return all_segments, all_colors, all_types

    def render(self):
        """Обновляет существующий RayCloud и очищает список лучей."""
        result, colors, types = self.trace_all()
        if self.mode == 'simple':
            self.cloud.update_from_trajectories(result, colors=colors)
        else:
            self.cloud.update_from_segments(result, base_colors=colors, energy_types=types)
        if self.pool:
            for ray in self.rays:
                self.pool.release(ray)
        self.rays.clear()
        self.elements.clear()
        return self.cloud

    def remove(self):
        """Удаляет облако с графика (если нужно полностью убрать лучи)."""
        self.plotter.remove_actor(self.cloud.actor)


class BeamEmitter:
    """
    Излучатель пучка лучей. Трансформируется (поворот, перемещение), может генерировать
    набор параллельных лучей или выдавать лучи из заданного пользователем списка.
    """
    def __init__(self, origin, direction=np.array([1.0, 0.0, 0.0]),
                 rotation_degrees=(0,0,0), pool=None,
                 num_rays=5, min_offset=-2.0, max_offset=2.0,
                 color="yellow", wavelength=550, energy_color_type=2,
                 energy=1.0, current_n=1.0):
        self.origin = np.asarray(origin, dtype=float)
        self.direction = np.asarray(direction, dtype=float)
        self.direction /= np.linalg.norm(self.direction)
        # Применяем начальный поворот
        rot = R.from_euler('xyz', rotation_degrees, degrees=True).as_matrix()
        self.direction = rot @ self.direction
        self.rotation_matrix = rot
        # Параметры генерации
        self.num_rays = num_rays
        self.min_offset = min_offset
        self.max_offset = max_offset
        self.color = color
        self.wavelength = wavelength
        self.energy_color_type = energy_color_type
        self.energy = energy
        self.current_n = current_n

        self.pool = pool

        # Пользовательские лучи (храним в локальной системе)
        self.custom_rays: List[Ray] = []
        self.use_custom = False   # если True, используются custom_rays вместо сетки

    def add_ray(self, ray: Ray):
        """Добавить пользовательский луч (в локальной системе излучателя)."""
        self.custom_rays.append(ray)
        self.use_custom = True

    def rotate(self, angles_deg):
        rot = R.from_euler('xyz', angles_deg, degrees=True).as_matrix()
        self.direction = rot @ self.direction
        self.rotation_matrix = rot @ self.rotation_matrix

    def translate(self, vec):
        self.origin += np.asarray(vec)

    def emit(self) -> List[Ray]:
        rays = []
        perp1, perp2 = get_tangents(self.direction)
        offsets = np.linspace(self.min_offset, self.max_offset, self.num_rays)
        for dy in offsets:
            world_origin = self.origin + dy * perp1
            if self.pool:
                ray = self.pool.acquire(origin=world_origin, direction=self.direction,
                                        energy=self.energy, current_n=self.current_n,
                                        color=self.color, wavelength=self.wavelength,
                                        energy_color_type=self.energy_color_type,
                                        polarization=None)
            else:
                ray = Ray(origin=world_origin, direction=self.direction,
                          energy=self.energy, current_n=self.current_n,
                          color=self.color, wavelength=self.wavelength,
                          energy_color_type=self.energy_color_type)
            rays.append(ray)
        return rays

    def get_mesh(self) -> pv.PolyData:
        """Маленькая стрелка для визуализации излучателя."""
        arrow = pv.Arrow(start=self.origin, direction=self.direction, scale=0.5)
        return arrow


class PlaneSurface:
    def __init__(self, point, rotation_degrees=(0,0,0), n_inside=1.0,
                 half_sizes=None, edge_radius=None,
                 reflection_range=None, refraction_range=None,
                 absorption_range=None,
                 lens_origin=None, lens_axis=None):
        self.point = np.array(point, dtype=float)
        base_normal = np.array([1.0, 0.0, 0.0])
        rot = R.from_euler('xyz', rotation_degrees, degrees=True).as_matrix()
        self.normal = rot @ base_normal
        self.n = n_inside

        # Автоматические lens_origin/lens_axis, если не заданы явно
        self.lens_origin = np.array(lens_origin, dtype=float) if lens_origin is not None else self.point.copy()
        self.lens_axis = np.array(lens_axis, dtype=float) if lens_axis is not None else self.normal.copy()
        self.lens_axis /= np.linalg.norm(self.lens_axis)

        # Прямоугольная апертура
        if half_sizes is not None:
            self.half_sizes = half_sizes
            self.edge_radius = 0.0
            self.face_tangents = get_tangents(self.normal)
        else:
            self.half_sizes = None
            self.face_tangents = None
            self.edge_radius = edge_radius if edge_radius is not None else 0.0

        self.reflection_range = reflection_range
        self.refraction_range = refraction_range
        self.absorption_range = absorption_range

    def rotate(self, angles_deg):
        rot = R.from_euler('xyz', angles_deg, degrees=True).as_matrix()
        # point и lens_origin остаются на месте
        self.normal = rot @ self.normal
        self.lens_axis = rot @ self.lens_axis
        if self.face_tangents is not None:
            self.face_tangents = (rot @ self.face_tangents[0], rot @ self.face_tangents[1])

    def translate(self, vec):
        self.point += np.asarray(vec)
        self.lens_origin += np.asarray(vec)

    def _slow_intersect(self, ray: Ray) -> Optional[float]:
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

    def intersect(self, ray: Ray) -> Optional[float]:
        use_rect = self.half_sizes is not None and self.face_tangents is not None
        tangents = None
        half = None
        if use_rect:
            tangents = np.array(self.face_tangents)
            half = np.array(self.half_sizes)
        t = plane_intersect(ray.origin, ray.direction, self.point, self.normal,
                            self.lens_origin, self.lens_axis, self.edge_radius,
                            half, tangents, use_rect)
        if t < 0.0:
            return None
        return t

    def get_normal(self, point: np.ndarray) -> np.ndarray:
        return self.normal

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

    def get_mesh(self) -> pv.PolyData:
        if self.half_sizes is not None and self.face_tangents is not None:
            # Прямоугольник
            t1, t2 = self.face_tangents
            hu, hv = self.half_sizes
            c = self.lens_origin
            p0 = c - hu * t1 - hv * t2
            p1 = c + hu * t1 - hv * t2
            p2 = c + hu * t1 + hv * t2
            p3 = c - hu * t1 + hv * t2
            vertices = np.array([p0, p1, p2, p3])
            faces = np.array([[3, 0, 1, 2], [3, 0, 2, 3]])
            return pv.PolyData(vertices, faces)
        else:
            # Круглое (диск) с edge_radius
            radius = self.edge_radius if self.edge_radius else 1.0
            disc = pv.Disc(center=(0, 0, 0), normal=(0, 0, 1), inner=0, outer=radius, c_res=64)
            # Поворот и перенос
            v_from = np.array([0., 0., 1.])
            v_to = self.normal
            # ... матрица поворота (используем calculate_rotation_matrix или аналогичную)
            rot = calculate_rotation_matrix(v_to)  # уже есть в main
            transform = np.eye(4)
            transform[:3, :3] = rot
            transform[:3, 3] = self.lens_origin
            return disc.transform(transform, inplace=False)


class SphereSurface:
    """
    Сферическая поверхность с ограничениями по радиусу апертуры и продольному
    положению (толщине) вдоль оптической оси линзы.
    """
    def __init__(self, center, radius, rotation_degrees=(0,0,0), n_inside=1.0,
                 edge_radius=None, thickness=0.0,
                 reflection_range=None, refraction_range=None,
                 absorption_range=None,
                 lens_origin=None, lens_axis=None):
        self.center = np.array(center, dtype=float)
        self.radius = radius
        base_axis = np.array([1.0, 0.0, 0.0])
        rot = R.from_euler('xyz', rotation_degrees, degrees=True).as_matrix()
        default_axis = rot @ base_axis

        # Явные lens_origin/lens_axis имеют приоритет
        self.lens_origin = np.array(lens_origin, dtype=float) if lens_origin is not None else self.center.copy()
        self.lens_axis = np.array(lens_axis, dtype=float) if lens_axis is not None else default_axis.copy()
        self.lens_axis = self.lens_axis.astype(float) / np.linalg.norm(self.lens_axis)

        self.n = n_inside
        self.edge_radius = edge_radius if edge_radius is not None else 0.0
        self.thickness = thickness
        self.reflection_range = reflection_range
        self.refraction_range = refraction_range
        self.absorption_range = absorption_range

    def rotate(self, angles_deg):
        rot = R.from_euler('xyz', angles_deg, degrees=True).as_matrix()
        v = self.center - self.lens_origin  # вектор от вершины к центру
        self.center = self.lens_origin + rot @ v
        self.lens_axis = rot @ self.lens_axis

    def translate(self, vec):
        self.center += np.asarray(vec)
        self.lens_origin += np.asarray(vec)

    def intersect(self, ray: Ray) -> Optional[float]:
        t = sphere_intersect(ray.origin, ray.direction, self.center, self.radius,
                             self.lens_origin, self.lens_axis, self.edge_radius, self.thickness)
        if t < 0.0:
            return None
        return t

    def _slow_intersect(self, ray: Ray) -> Optional[float]:
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

    def __init__(self, mesh, rotation_degrees=(0,0,0), translation=(0,0,0),
                 n_inside=1.0, reflection_range=None, refraction_range=None,
                 absorption_range=None):
        # Загрузка тримеша (как и раньше)
        if isinstance(mesh, str):
            self.trimesh_obj = trimesh.load(mesh)
            if isinstance(self.trimesh_obj, trimesh.Scene):
                self.trimesh_obj = trimesh.util.concatenate(
                    [g for g in self.trimesh_obj.geometry.values() if isinstance(g, trimesh.Trimesh)])
            if not isinstance(self.trimesh_obj, trimesh.Trimesh):
                raise TypeError("Файл не содержит треугольной сетки")
        elif isinstance(mesh, trimesh.Trimesh):
            self.trimesh_obj = mesh
        elif isinstance(mesh, pv.PolyData):
            verts, faces = mesh.points, mesh.faces.reshape(-1, 4)[:, 1:4]
            self.trimesh_obj = trimesh.Trimesh(vertices=verts, faces=faces)
        else:
            raise TypeError("mesh должен быть str, trimesh.Trimesh или pv.PolyData")

        # Применяем поворот и перенос
        rot_4x4 = np.eye(4)
        rot_4x4[:3, :3] = R.from_euler('xyz', rotation_degrees, degrees=True).as_matrix()
        self.trimesh_obj.apply_transform(rot_4x4)
        if translation is not None:
            self.trimesh_obj.apply_translation(translation)

        self.mesh = self.trimesh_obj
        self.intersector = trimesh.ray.ray_triangle.RayMeshIntersector(self.mesh)
        self.n = n_inside
        self.reflection_range = reflection_range
        self.refraction_range = refraction_range
        self.absorption_range = absorption_range
        self._last_hit_triangle_idx = None

    def is_active(self, wavelength):
        if wavelength is None:
            return True
        if self.reflection_range is None and self.refraction_range is None and self.absorption_range is None:
            return False
        in_ref = self.reflection_range is not None and (self.reflection_range[0] <= wavelength <= self.reflection_range[1])
        in_refr = self.refraction_range is not None and (self.refraction_range[0] <= wavelength <= self.refraction_range[1])
        in_abs = self.absorption_range is not None and (self.absorption_range[0] <= wavelength <= self.absorption_range[1])
        return in_ref or in_refr or in_abs

    def intersect(self, ray: Ray) -> Optional[float]:
        origins = np.array([ray.origin])
        directions = np.array([ray.direction])
        locations, _, tri_indices = self.intersector.intersects_location(
            origins, directions, multiple_hits=False)
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
            return self.mesh.face_normals[self._last_hit_triangle_idx]
        _, _, tri_idx = trimesh.proximity.closest_point(self.mesh, [point])
        return self.mesh.face_normals[tri_idx[0]]

    def get_mesh(self):
        verts = self.mesh.vertices
        faces = np.hstack([np.full((len(self.mesh.faces), 1), 3), self.mesh.faces])
        return pv.PolyData(verts, faces)

    def rotate(self, angles_deg):
        rot_4x4 = np.eye(4)
        rot_4x4[:3, :3] = R.from_euler('xyz', angles_deg, degrees=True).as_matrix()
        # Вращение вокруг локального центра (bounding box center)
        center = self.mesh.bounding_box.center_mass
        # Перенос в нуль, поворот, возврат
        T1 = np.eye(4)
        T1[:3, 3] = -center
        T2 = np.eye(4)
        T2[:3, 3] = center
        self.mesh.apply_transform(T2 @ rot_4x4 @ T1)
        self.intersector = trimesh.ray.ray_triangle.RayMeshIntersector(self.mesh)

    def translate(self, vec):
        self.mesh.apply_translation(vec)
        self.intersector = trimesh.ray.ray_triangle.RayMeshIntersector(self.mesh)


class AsphericSurface:
    def __init__(self, center, radius, conic_constant=0.0, aspheric_coeffs=None,
                 rotation_degrees=(0,0,0), n_inside=1.0,
                 edge_radius=None, thickness=0.0,
                 reflection_range=None, refraction_range=None,
                 absorption_range=None,
                 lens_origin=None, lens_axis=None):
        self.center = np.array(center, dtype=float)
        self.radius = radius
        self.k = conic_constant
        self.aspheric_coeffs = aspheric_coeffs if aspheric_coeffs else []

        base_axis = np.array([1.0, 0.0, 0.0])
        rot = R.from_euler('xyz', rotation_degrees, degrees=True).as_matrix()
        default_axis = rot @ base_axis

        self.lens_origin = np.array(lens_origin, dtype=float) if lens_origin is not None else self.center.copy()
        self.lens_axis = np.array(lens_axis, dtype=float) if lens_axis is not None else default_axis.copy()
        self.lens_axis /= np.linalg.norm(self.lens_axis)

        self.n = n_inside
        self.edge_radius = edge_radius if edge_radius is not None else 0.0
        self.thickness = thickness
        self.reflection_range = reflection_range
        self.refraction_range = refraction_range
        self.absorption_range = absorption_range

        self._t1, self._t2 = get_tangents(self.lens_axis)
        self._rot_local_to_world = np.column_stack([self.lens_axis, self._t1, self._t2])
        self._rot_world_to_local = self._rot_local_to_world.T

    def _world_to_local(self, point):
        return self._rot_world_to_local @ (point - self.lens_origin)

    def _local_to_world(self, local_point):
        return self.lens_origin + self._rot_local_to_world @ local_point

    def sag(self, r):
        c = 1.0 / self.radius if self.radius != 0 else 0.0
        if abs(c) < 1e-12:
            sag0 = np.zeros_like(r)
        else:
            discr = 1.0 - (1.0 + self.k) * c**2 * r**2
            safe_discr = np.maximum(0.0, discr)
            sag0 = np.where(discr >= 0,
                            (c * r**2) / (1.0 + np.sqrt(safe_discr)),
                            np.inf)
        sag_asp = np.zeros_like(r)
        for i, A in enumerate(self.aspheric_coeffs):
            sag_asp += A * r**(2 * (i + 1))
        return sag0 + sag_asp

    def sag_derivative(self, r):
        c = 1.0 / self.radius if self.radius != 0 else 0.0
        if abs(c) < 1e-12:
            dsag0 = np.zeros_like(r)
        else:
            discr = 1.0 - (1.0 + self.k) * c**2 * r**2
            safe_discr = np.maximum(0.0, discr)
            dsag0 = np.where(discr >= 0,
                             (c * r) / np.sqrt(safe_discr),
                             0.0)
        dsag_asp = np.zeros_like(r)
        for i, A in enumerate(self.aspheric_coeffs):
            power = 2 * (i + 1)
            dsag_asp += A * power * r**(power - 1)
        return dsag0 + dsag_asp

    @staticmethod
    @njit
    def _intersect_numba(origin, direction, radius, k, coeffs, edge_radius):
        # Плоская поверхность
        if abs(radius) > 1e8:
            if abs(direction[0]) < 1e-12:
                return -1.0
            t = -origin[0] / direction[0]
            if t <= 1e-6:
                return -1.0
            p = origin + t * direction
            if p[1] ** 2 + p[2] ** 2 > edge_radius ** 2:
                return -1.0
            return t

        # Опорная сфера
        R = radius
        a = direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2
        b = 2.0 * (origin[0] * direction[0] + origin[1] * direction[1] + origin[2] * direction[2]) - 2 * R * direction[
            0]
        c = origin[0] ** 2 + origin[1] ** 2 + origin[2] ** 2 - 2 * R * origin[0]
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return -1.0
        sqrt_disc = np.sqrt(disc)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)
        t_guess = t1 if t1 > 1e-6 else (t2 if t2 > 1e-6 else -1.0)
        if t_guess < 0.0:
            return -1.0

        # Ньютон
        t = t_guess
        for _ in range(50):
            p = origin + t * direction
            r = np.sqrt(p[1] ** 2 + p[2] ** 2)
            c = 1.0 / R
            discr = 1.0 - (1.0 + k) * c ** 2 * r ** 2
            if discr < 0.0:
                return -1.0
            sqrt_discr = np.sqrt(discr)
            sag = (c * r ** 2) / (1.0 + sqrt_discr)
            # Асферические члены
            for i, A in enumerate(coeffs):
                sag += A * r ** (2 * (i + 1))
            F = p[0] - sag
            if abs(F) < 1e-9:
                break
            # Производная sag
            if r < 1e-12:
                dsag = 0.0
                drdt = 0.0
            else:
                dsag = (c * r) / sqrt_discr
                for i, A in enumerate(coeffs):
                    power = 2 * (i + 1)
                    dsag += A * power * r ** (power - 1)
                drdt = (p[1] * direction[1] + p[2] * direction[2]) / r
            dFdt = direction[0] - dsag * drdt
            if abs(dFdt) < 1e-15:
                break
            t -= F / dFdt
            if t < 0.0:
                return -1.0
        else:
            # fallback – проверка начального приближения
            p_init = origin + t_guess * direction
            r_init = np.sqrt(p_init[1] ** 2 + p_init[2] ** 2)
            discr_init = 1.0 - (1.0 + k) * (1.0 / R) ** 2 * r_init ** 2
            if discr_init >= 0.0:
                sag_init = (1.0 / R * r_init ** 2) / (1.0 + np.sqrt(discr_init))
                if abs(p_init[0] - sag_init) < 1e-4:
                    t = t_guess
                else:
                    return -1.0
            else:
                return -1.0

        p_final = origin + t * direction
        if p_final[1] ** 2 + p_final[2] ** 2 > edge_radius ** 2:
            return -1.0
        return t

    def intersect(self, ray: Ray) -> Optional[float]:
        # Переводим луч в локальные координаты
        origin_loc = self._world_to_local(ray.origin)
        dir_loc = self._rot_world_to_local @ ray.direction
        # Вызываем ускоренную функцию
        t = self._intersect_numba(origin_loc.astype(np.float64),
                                  dir_loc.astype(np.float64),
                                  self.radius, self.k,
                                  np.array(self.aspheric_coeffs, dtype=np.float64),
                                  self.edge_radius)
        if t < 0.0:
            return None
        return float(t)

    def get_normal(self, point: np.ndarray) -> np.ndarray:
        p_loc = self._world_to_local(point)
        r = np.sqrt(p_loc[1]**2 + p_loc[2]**2)
        if r < 1e-12:
            normal_loc = np.array([1.0, 0.0, 0.0])
        else:
            dsag = self.sag_derivative(r)
            normal_loc = np.array([1.0, -dsag * p_loc[1] / r, -dsag * p_loc[2] / r])
        normal_loc /= np.linalg.norm(normal_loc)
        return self._rot_local_to_world @ normal_loc

    def get_mesh(self, n_radial=40, n_azimuth=80):
        rs = np.linspace(0, self.edge_radius, n_radial)
        phis = np.linspace(0, 2*np.pi, n_azimuth)
        r_grid, phi_grid = np.meshgrid(rs, phis)
        y_loc = r_grid * np.cos(phi_grid)
        z_loc = r_grid * np.sin(phi_grid)
        r = np.sqrt(y_loc**2 + z_loc**2)
        x_loc = self.sag(r)
        x_loc = np.where(np.isfinite(x_loc), x_loc, 0.0)
        grid = pv.StructuredGrid(x_loc, y_loc, z_loc)
        poly = grid.extract_surface(algorithm='dataset_surface')
        matrix = np.eye(4)
        matrix[:3, :3] = self._rot_local_to_world
        matrix[:3, 3] = self.lens_origin
        poly.transform(matrix, inplace=True)
        return poly

    def rotate(self, angles_deg):
        rot = R.from_euler('xyz', angles_deg, degrees=True).as_matrix()
        v = self.center - self.lens_origin
        self.center = self.lens_origin + rot @ v
        self.lens_axis = rot @ self.lens_axis
        self._t1, self._t2 = get_tangents(self.lens_axis)
        self._rot_local_to_world = np.column_stack([self.lens_axis, self._t1, self._t2])
        self._rot_world_to_local = self._rot_local_to_world.T

    def translate(self, vec):
        self.center += np.asarray(vec)
        self.lens_origin += np.asarray(vec)

    def is_active(self, wavelength):
        if wavelength is None: return True
        if self.reflection_range is None and self.refraction_range is None and self.absorption_range is None:
            return False
        in_ref = self.reflection_range is not None and (self.reflection_range[0] <= wavelength <= self.reflection_range[1])
        in_refr = self.refraction_range is not None and (self.refraction_range[0] <= wavelength <= self.refraction_range[1])
        in_abs = self.absorption_range is not None and (self.absorption_range[0] <= wavelength <= self.absorption_range[1])
        return in_ref or in_refr or in_abs


class RectangularScreen(PlaneSurface):
    def __init__(self, center, normal, width, height, absorption_range=(0,10000)):
        center = np.asarray(center, dtype=float)
        normal = np.asarray(normal, dtype=float)    # ← обязательно float
        half_u, half_v = width/2, height/2
        t1, t2 = get_tangents(normal)
        super().__init__(
            point=center, normal=normal, n_inside=1.0,
            lens_origin=center, lens_axis=normal,
            edge_radius=0.0,
            half_sizes=(half_u, half_v),
            face_tangents=(t1, t2),
            absorption_range=absorption_range
        )
        self.width, self.height = width, height

    def get_mesh(self) -> pv.PolyData:
        t1, t2 = self.face_tangents
        hu, hv = self.half_sizes
        c = self.lens_origin
        p0 = c - hu * t1 - hv * t2
        p1 = c + hu * t1 - hv * t2
        p2 = c + hu * t1 + hv * t2
        p3 = c - hu * t1 + hv * t2
        vertices = np.array([p0, p1, p2, p3])
        faces = np.array([[3, 0, 1, 2], [3, 0, 2, 3]])
        return pv.PolyData(vertices, faces)


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


class ThinLens:
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
    def __init__(self, origin, rotation_degrees=(0,0,0), R1=None, R2=None,
                 thickness=2.0, edge_radius=3.0, n=1.5,
                 reflection_range=None, refraction_range=(0, np.inf),
                 absorption_range=None):
        self.origin = np.array(origin, dtype=float)
        # Сохраняем углы Эйлера для возможного использования
        self.rotation_degrees = rotation_degrees
        # Вычисляем начальную оптическую ось
        base_axis = np.array([1.0, 0.0, 0.0])
        rot = R.from_euler('xyz', rotation_degrees, degrees=True).as_matrix()
        self.axis_dir = rot @ base_axis
        self.thickness = thickness
        self.edge_radius = edge_radius
        self.n = n
        self.R1, self.R2 = R1, R2

        # Спектральные диапазоны (применяются к обеим поверхностям)
        self.reflection_range = reflection_range
        self.refraction_range = refraction_range
        self.absorption_range = absorption_range

        # Построение поверхностей
        self._create_surfaces()

        # Расчёт оптических параметров (фокус и пр.)
        self._calc_optical_params()

    def _create_surfaces(self):
        """Пересоздаёт front и back на основе текущих origin, axis_dir, R1, R2."""
        # Матрица поворота от базовой оси (X) к axis_dir
        v_old = np.array([1.0, 0.0, 0.0])
        v_new = self.axis_dir
        if np.allclose(v_old, v_new):
            rot = np.eye(3)
        elif np.allclose(v_old, -v_new):
            rot = np.diag([-1, -1, 1])
        else:
            v = np.cross(v_old, v_new)
            s = np.linalg.norm(v)
            c = np.dot(v_old, v_new)
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            rot = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
        self.rotation = rot

        half = self.thickness / 2
        v1_local = -half
        v2_local = half

        # Передняя поверхность
        if self.R1 is None:
            p1 = self.origin + rot @ np.array([v1_local, 0, 0])
            n1 = rot @ np.array([-1, 0, 0])
            self.front = PlaneSurface(
                point=p1, normal=n1, n_inside=self.n,
                lens_origin=self.origin, lens_axis=self.axis_dir,
                edge_radius=self.edge_radius,
                reflection_range=self.reflection_range,
                refraction_range=self.refraction_range,
                absorption_range=self.absorption_range
            )
        else:
            c1 = self.origin + rot @ np.array([v1_local + self.R1, 0, 0])
            self.front = SphereSurface(
                center=c1, radius=abs(self.R1), n_inside=self.n,
                lens_origin=self.origin, lens_axis=self.axis_dir,
                edge_radius=self.edge_radius, thickness=self.thickness,
                reflection_range=self.reflection_range,
                refraction_range=self.refraction_range,
                absorption_range=self.absorption_range
            )

        # Задняя поверхность
        if self.R2 is None:
            p2 = self.origin + rot @ np.array([v2_local, 0, 0])
            n2 = rot @ np.array([1, 0, 0])
            self.back = PlaneSurface(
                point=p2, normal=n2, n_inside=self.n,
                lens_origin=self.origin, lens_axis=self.axis_dir,
                edge_radius=self.edge_radius,
                reflection_range=self.reflection_range,
                refraction_range=self.refraction_range,
                absorption_range=self.absorption_range
            )
        else:
            c2 = self.origin + rot @ np.array([v2_local - self.R2, 0, 0])
            self.back = SphereSurface(
                center=c2, radius=abs(self.R2), n_inside=self.n,
                lens_origin=self.origin, lens_axis=self.axis_dir,
                edge_radius=self.edge_radius, thickness=self.thickness,
                reflection_range=self.reflection_range,
                refraction_range=self.refraction_range,
                absorption_range=self.absorption_range
            )

    def _calc_optical_params(self):
        # (оставьте ваш существующий расчёт f_dist)
        r1_val = self.R1 if self.R1 else 1e10
        r2_val = -self.R2 if self.R2 else -1e10
        inv_f = (self.n - 1) * (1 / r1_val - 1 / r2_val +
                                ((self.n - 1) * self.thickness) / (self.n * r1_val * r2_val))
        self.f_dist = 1 / inv_f if abs(inv_f) > 1e-10 else float('inf')

    def rotate(self, angles_deg):
        """Поворот линзы вокруг её центра (origin)."""
        rot = R.from_euler('xyz', angles_deg, degrees=True).as_matrix()
        self.axis_dir = rot @ self.axis_dir
        self.axis_dir /= np.linalg.norm(self.axis_dir)
        self._create_surfaces()

    def translate(self, vec):
        self.origin += np.asarray(vec)
        self._create_surfaces()

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

        front_mesh = pv.StructuredGrid(x_front, y, z).extract_surface(algorithm='dataset_surface')
        back_mesh = pv.StructuredGrid(x_back, y, z).extract_surface(algorithm='dataset_surface')

        # Ободок (соединение краёв)
        rim_x = np.array([x_front[:, -1], x_back[:, -1]])
        rim_y = np.array([y[:, -1], y[:, -1]])
        rim_z = np.array([z[:, -1], z[:, -1]])
        rim_mesh = pv.StructuredGrid(rim_x, rim_y, rim_z).extract_surface(algorithm='dataset_surface')

        local_mesh = front_mesh.merge(back_mesh).merge(rim_mesh)

        # Перенос в мировые координаты
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.origin
        return local_mesh.transform(matrix, inplace=False)

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


class Aperture(PlaneSurface):
    """Диафрагма: непрозрачная плоскость с круглым отверстием."""

    class Aperture(PlaneSurface):
        def __init__(self, point, normal, aperture_radius, outer_radius=3.0,
                     absorption_range=(0, 10000), n_inside=1.0):
            super().__init__(
                point=point, normal=normal, n_inside=n_inside,
                lens_origin=point, lens_axis=normal,
                edge_radius=0.0,
                absorption_range=absorption_range
            )
            self.aperture_radius = aperture_radius
            self.outer_radius = outer_radius

        def get_mesh(self) -> pv.PolyData:
            disc = pv.Disc(center=(0, 0, 0), normal=(0, 0, 1),
                           inner=self.aperture_radius, outer=self.outer_radius, c_res=100)
            # Поворот к нормали
            v_from = np.array([0., 0., 1.])
            v_to = self.normal
            if np.allclose(v_from, v_to):
                rot = np.eye(3)
            elif np.allclose(v_from, -v_to):
                rot = -np.eye(3)
            else:
                v = np.cross(v_from, v_to)
                c = np.dot(v_from, v_to)
                K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                rot = np.eye(3) + K + (K @ K) * (1 / (1 + c))
            transform = np.eye(4)
            transform[:3, :3] = rot
            transform[:3, 3] = self.lens_origin
            return disc.transform(transform, inplace=False)

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
            path.append(current_ray.origin + current_ray.direction * RAY_INFINITY_DISTANCE)
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
            path.append(current_ray.origin + current_ray.direction * RAY_INFINITY_DISTANCE)
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
              offset_distance: float = 0.5, use_polarization_color=False):
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
                         total_limit=5000, offset_distance=offset_distance, use_polarization_color=use_polarization_color)
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
