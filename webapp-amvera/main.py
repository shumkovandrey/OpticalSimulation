import random

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as R
from typing import List, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

from numba import njit
from fast_math import *

import trimesh


import os
# os.environ['NUMBA_DISABLE_JIT'] = '1'


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


def split_ray(ray: Ray, normal: np.ndarray, n_next: float, start_point: np.ndarray,
              allow_reflection: bool = True, allow_refraction: bool = True,
              offset_distance: float = 0.01,
              use_polarization_color: bool = False,
              pool: Optional[RayPool] = None) -> List[Ray]:
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
        energy = abs(new_E_s) ** 2 + abs(new_E_p) ** 2
        if energy > 1e-9:
            reflected_dir = ray.direction - 2 * np.dot(ray.direction, normal) * normal
            new_pol = new_E_s * s_dir + new_E_p * p_dir
            if pool:
                # Передаем type(ray) в пул, чтобы он извлек объект нужного класса
                new_ray = pool.acquire(start_point + EPS * reflected_dir, reflected_dir,
                                       energy, n1, ray.color, ray.wavelength,
                                       ray.energy_color_type, new_pol, ray_class=type(ray))
            else:
                # Создаем экземпляр напрямую через тип текущего луча
                new_ray = type(ray)(start_point + EPS * reflected_dir, reflected_dir,
                                    energy, n1, ray.color, ray.wavelength,
                                    ray.energy_color_type, new_pol)
            if use_polarization_color:
                new_ray.update_color_from_polarization()
            new_rays.append(new_ray)

    # Преломлённый луч
    if allow_refraction and (abs(t_s) > 1e-9 or abs(t_p) > 1e-9):
        new_E_s = t_s * E_s
        new_E_p = t_p * E_p
        energy = abs(new_E_s) ** 2 + abs(new_E_p) ** 2
        if energy > 1e-9:
            eta = n1 / n2
            cos_t = np.sqrt(max(0.0, 1.0 - (eta ** 2) * (1.0 - cos_i ** 2)))
            refracted_dir = eta * ray.direction + (eta * cos_i - cos_t) * normal
            refracted_dir /= np.linalg.norm(refracted_dir)
            new_pol = new_E_s * s_dir + new_E_p * p_dir
            if pool:
                # Передаем type(ray) в пул
                new_ray = pool.acquire(start_point + EPS * refracted_dir, refracted_dir, energy, n2,
                                       color=ray.color, wavelength=ray.wavelength,
                                       energy_color_type=ray.energy_color_type,
                                       polarization=new_pol, ray_class=type(ray))
            else:
                new_ray = type(ray)(start_point + EPS * refracted_dir, refracted_dir, energy, n2,
                                    color=ray.color, wavelength=ray.wavelength,
                                    energy_color_type=ray.energy_color_type,
                                    polarization=new_pol)

            if use_polarization_color:
                new_ray.update_color_from_polarization()
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


# Вставить в глобальную область (после импортов, до определения классов)
@njit
def _sag_numba(r, R, k, coeffs):
    """Стрелка прогиба асферической поверхности в зависимости от радиуса r."""
    if abs(R) < 1e-12:  # плоский случай
        sag = 0.0
    else:
        c = 1.0 / R
        discr = 1.0 - (1.0 + k) * c**2 * r**2
        if discr < 0.0:
            return np.inf
        sqrt_discr = np.sqrt(discr)
        sag = (c * r**2) / (1.0 + sqrt_discr)
    for i, A in enumerate(coeffs):
        sag += A * r**(2 * (i + 1))
    return sag


def get_dispersion_n(base_n: float, ray: Ray) -> float:
    """
    Вычисляет показатель преломления.
    Если луч является экземпляром DispersiveRay, применяется формула Коши.
    Если это обычный Ray, возвращается неизменный базовый base_n.
    """
    # Если это НЕ DispersiveRay или у него нет длины волны — возвращаем базовый n объекта
    if not isinstance(ray, DispersiveRay) or ray.wavelength is None:
        return base_n

    # Переводим нанометры в микрометры для формулы Коши
    lambda_mkm = ray.wavelength / 1000.0

    # Эмпирические коэффициенты Коши для стекла типа К8 (BK7)
    A = base_n - 0.01  # Смещение, чтобы в центре спектра получался примерно base_n
    B = 0.005  # Коэффициент дисперсии

    return A + B / (lambda_mkm ** 2)


# ------------------------------------------------
# Единый контейнер для отрезка луча (результат трассировки)
# ------------------------------------------------
@dataclass(slots=True)
class Segment:
    """Один отрезок трассированного луча."""
    start: np.ndarray          # 3-вектор начала
    end: np.ndarray            # 3-вектор конца
    energy: float              # энергия (0..1)
    color: tuple               # RGB-цвет в диапазоне 0..1
    # energy_color_type: int     # режим привязки прозрачности к энергии (0/1/2)

    # Для удобства распаковки в старом стиле (опционально)
    def __iter__(self):
        return iter((self.start, self.end, self.energy, self.color))


# ------------------------------------------------
# Результат поиска пересечения луча с элементами сцены
# ------------------------------------------------
@dataclass(slots=True)
class HitInfo:
    """Информация о пересечении луча с объектом."""
    obj: Any                   # элемент сцены, с которым произошло пересечение
    t: float                   # расстояние от начала луча до точки удара
    point: np.ndarray          # точка удара (3D)
    normal: np.ndarray         # нормаль в точке удара
    allow_reflection: bool     # разрешено ли отражение на этой поверхности
    allow_refraction: bool     # разрешено ли преломление на этой поверхности
    absorbed: bool             # поглощается ли луч (полное поглощение)
    n_inside: float            # показатель преломления материала внутри объекта (1.0 если нет)
    is_thin_lens: bool = False # специальный флаг для тонкой линзы

    def __post_init__(self):
        # Гарантируем, что векторы имеют тип float64
        self.point = np.asarray(self.point, dtype=np.float64)
        self.normal = np.asarray(self.normal, dtype=np.float64)


# ------------------------------------------------------------
# Абстрактный базовый класс для всех режимов трассировки
# ------------------------------------------------------------
class TraceMode(ABC):
    """Общий интерфейс для любых алгоритмов трассировки."""
    def __init__(self, energy_color_type: int = 1, pool: Optional['RayPool'] = None):
        self.energy_color_type = energy_color_type
        self.pool = pool

    @abstractmethod
    def trace(self, ray: 'Ray', elements: List) -> List[Segment]:
        """
        Трассирует один луч через список элементов сцены и возвращает
        плоский список отрезков Segment, готовых к отрисовке.
        """
        ...


# ------------------------------------------------------------
# Режим "simple" – однолучевой последовательный проход
# ------------------------------------------------------------
class SimpleMode(TraceMode):
    """
    Последовательная трассировка без ветвления.
    На каждом шаге выбирается ровно одно продолжение (преломление имеет
    приоритет, если включён флаг prioritize_refraction).
    """
    def __init__(self,
                 max_bounces: int = 30,
                 offset_distance: float = 0.001,
                 prioritize_refraction: bool = True,
                 energy_color_type=1,
                 pool=None):
        """
        Параметры
        ---------
        max_bounces : int
            Максимальное число взаимодействий луча с поверхностями.
        offset_distance : float
            Минимальное смещение точки старта нового луча, чтобы избежать
            самопересечения с той же поверхностью.
        prioritize_refraction : bool
            Если True и доступно преломление, отражение подавляется (даже
            если поверхность допускает и то и другое). Полезно для
            моделирования линз без паразитных отражений.
        """
        super().__init__(energy_color_type, pool)

        self.max_bounces = max_bounces
        self.offset_distance = offset_distance
        self.prioritize_refraction = prioritize_refraction

    def trace(self, ray: 'Ray', elements: List) -> List[Segment]:
        return _trace_simple(
            ray,
            elements,
            max_bounces=self.max_bounces,
            offset_distance=self.offset_distance,
            prioritize_refraction=self.prioritize_refraction,
            pool=self.pool
        )


# ------------------------------------------------------------
# Режим "tree" – рекурсивное дерево лучей с учётом энергии
# ------------------------------------------------------------
class TreeMode(TraceMode):
    """
    Рекурсивная трассировка с ветвлением на отражённый и преломлённый лучи.
    Каждый порождённый луч обладает энергией, вычисленной по формулам Френеля.
    Лучи с энергией ниже порога отбрасываются.
    """
    def __init__(self,
                 max_depth: int = 20,
                 min_energy: float = 0.01,
                 offset_distance: float = 0.001,
                 use_polarization_color: bool = False,
                 total_limit: int = 5000,
                 energy_color_type: int = 1,
                 pool=None):
        """
        Параметры
        ---------
        max_depth : int
            Максимальная глубина рекурсии (количество ветвлений).
        min_energy : float
            Порог энергии, ниже которого лучи не обрабатываются.
        offset_distance : float
            Смещение для избежания самопересечений.
        use_polarization_color : bool
            Если True, цвет луча вычисляется по состоянию поляризации.
        total_limit : int
            Аварийное ограничение на общее количество отрезков во всех
            рекурсивных ветвях (защита от бесконечного роста).
        """
        super().__init__(energy_color_type, pool)

        self.max_depth = max_depth
        self.min_energy = min_energy
        self.offset_distance = offset_distance
        self.use_polarization_color = use_polarization_color
        self.total_limit = total_limit

    def trace(self, ray: 'Ray', elements: List) -> List[Segment]:
        return _trace_recursive(
            ray,
            elements,
            depth=self.max_depth,
            min_energy=self.min_energy,
            offset_distance=self.offset_distance,
            use_polarization_color=self.use_polarization_color,
            total_limit=self.total_limit,
            pool=self.pool
        )


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

        if np.linalg.norm(self.direction) < 1e-12:
            self.direction = np.array([1.0, 0.0, 0.0])  # или оставить нулевым, но тогда не использовать в трассировке
        else:
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


class DispersiveRay(Ray):
    """
    Класс луча, поддерживающий дисперсию (зависимость показателя преломления от длины волны).
    Wavelength передается в нанометрах (например, 550).
    """

    def __init__(self, origin, direction, wavelength=550.0, **kwargs):
        # Если в kwargs передан current_n, используем его, иначе рассчитываем для вакуума/воздуха (1.0)
        current_n = kwargs.pop('current_n', 1.0)

        super().__init__(
            origin=origin,
            direction=direction,
            wavelength=wavelength,
            current_n=current_n,
            **kwargs
        )

        # Автоматически красим луч в спектральный цвет в зависимости от длины волны (опционально)
        if self.color == "yellow" and wavelength is not None:
            self.color = self.wavelength_to_rgb(wavelength)

    @staticmethod
    def wavelength_to_rgb(wavelength_nm: float) -> Tuple[float, float, float]:
        """Утилита для перевода длины волны (нм) в приближенный RGB цвет (0..1) для красивой визуализации."""
        w = wavelength_nm
        if 380 <= w < 440:
            R, G, B = -(w - 440) / (440 - 380), 0.0, 1.0
        elif 440 <= w < 490:
            R, G, B = 0.0, (w - 440) / (490 - 440), 1.0
        elif 490 <= w < 510:
            R, G, B = 0.0, 1.0, -(w - 510) / (510 - 490)
        elif 510 <= w < 580:
            R, G, B = (w - 510) / (580 - 510), 1.0, 0.0
        elif 580 <= w < 645:
            R, G, B = 1.0, -(w - 645) / (645 - 580), 0.0
        elif 645 <= w <= 780:
            R, G, B = 1.0, 0.0, 0.0
        else:
            R, G, B = 1.0, 1.0, 1.0  # Белый для невидимого спектра

        # Интенсивность падает на краях видимого спектра
        factor = 1.0
        if 380 <= w < 420:
            factor = 0.3 + 0.7 * (w - 380) / (420 - 380)
        elif 700 < w <= 780:
            factor = 0.3 + 0.7 * (780 - w) / (780 - 700)

        return (R * factor, G * factor, B * factor)


class WhiteRay(Ray):
    """
    Класс луча белого света, полностью интегрированный в систему RayPool.
    До первого взаимодействия ведет себя как один луч белого цвета.
    """

    def __init__(self, origin, direction, num_spectral_bands=7, pool=None, **kwargs):
        kwargs['color'] = (1.0, 1.0, 1.0)
        super().__init__(origin, direction, **kwargs)
        self.spectral_rays = []

        # Если при самом первом создании передан пул, сразу готовим спектр
        if pool:
            self.reinit_spectral_rays(num_spectral_bands, pool)

    def reinit_spectral_rays(self, num_spectral_bands: int, pool: RayPool):
        """Пересобирает или обновляет массив спектральных лучей, забирая их из пула."""
        # На всякий случай очищаем старые (хотя release() делает это автоматически)
        self.spectral_rays.clear()

        wavelengths = np.linspace(400.0, 700.0, num_spectral_bands)
        sub_energy = self.energy / num_spectral_bands

        for wl in wavelengths:
            # Запрашиваем DispersiveRay напрямую из пула
            sub_ray = pool.acquire(
                origin=self.origin,
                direction=self.direction,
                energy=sub_energy,
                current_n=self.current_n,
                energy_color_type=self.energy_color_type,
                wavelength=wl,
                ray_class=DispersiveRay
            )
            self.spectral_rays.append(sub_ray)

    def update_position(self, origin, direction):
        """Синхронизация положений, если луч перемещается до удара."""
        self.origin[:] = origin
        self.direction[:] = direction
        for r in self.spectral_rays:
            r.origin[:] = origin
            r.direction[:] = direction


class RayPool:
    """Пул переиспользуемых лучей для снижения аллокаций (с поддержкой спектрального белого света)."""

    def __init__(self, initial_size=100):
        self.pools = {}
        self.initial_size = initial_size
        self._ensure_pool_exists(Ray)

    def _ensure_pool_exists(self, ray_class):
        """Создает внутреннюю корзину для нового типа луча, если её нет."""
        if ray_class not in self.pools:
            # Для базовых типов создаем пустые заготовки
            self.pools[ray_class] = []
            # Заполняем дефолтными объектами (для WhiteRay это делать не нужно заранее,
            # так как количество диапазонов num_spectral_bands инициализируется динамически)
            if ray_class is not WhiteRay:
                for _ in range(self.initial_size):
                    self.pools[ray_class].append(ray_class(np.zeros(3), np.array([1.0, 0.0, 0.0])))

    def acquire(self, origin, direction, energy=1.0, current_n=1.0,
                color="yellow", energy_color_type=2, wavelength=None,
                polarization=None, ray_class=Ray, num_spectral_bands=7):
        """
        Взять луч определенного класса из пула и инициализировать его поля.
        """
        self._ensure_pool_exists(ray_class)

        # Вытаскиваем из пула или создаем новый экземпляр
        if self.pools[ray_class]:
            ray = self.pools[ray_class].pop()
        else:
            if ray_class is WhiteRay:
                ray = WhiteRay(np.zeros(3), np.array([1.0, 0.0, 0.0]), num_spectral_bands=num_spectral_bands, pool=self)
            else:
                ray = ray_class(np.zeros(3), np.array([1.0, 0.0, 0.0]))

        # Инициализируем базовые атрибуты
        ray.origin[:] = origin
        ray.direction[:] = direction
        ray.direction /= np.linalg.norm(ray.direction)
        ray.energy = energy
        ray.current_n = current_n
        ray.energy_color_type = energy_color_type
        ray.wavelength = wavelength

        if polarization is not None:
            ray.polarization = np.array(polarization, dtype=complex)
        else:
            ray.polarization = None

        # Специфичная сборка для DispersiveRay
        if ray_class is DispersiveRay and wavelength is not None:
            # Игнорируем переданный аргумент color, так как для дисперсионного луча
            # цвет строго и монопольно диктуется его длиной волны (wavelength)
            ray.color = ray.wavelength_to_rgb(wavelength)
        elif ray_class is WhiteRay:
            ray.color = (1.0, 1.0, 1.0)  # Всегда белый до первого преломления
            ray.reinit_spectral_rays(num_spectral_bands, self)
        else:
            # Для обычных лучей оставляем переданный цвет
            ray.color = color

        return ray

    def release(self, ray: Ray):
        """Определяет тип луча и возвращает его (и его под-лучи) в нужную корзину пула."""
        if ray is None:
            return

        ray_class = type(ray)

        # Если возвращаем белый луч — сначала рекурсивно освобождаем его спектральные составляющие
        if ray_class is WhiteRay and hasattr(ray, 'spectral_rays'):
            for sub_ray in ray.spectral_rays:
                self.release(sub_ray)
            ray.spectral_rays.clear()

        if ray_class not in self.pools:
            self.pools[ray_class] = []
        self.pools[ray_class].append(ray)


class RayCloud:
    """
    Единый актор для отрисовки множества отрезков с индивидуальным цветом
    и прозрачностью, зависящей от энергии.
    """

    def __init__(self, plotter: pv.Plotter,
                 default_color="yellow",
                 line_width: float = 2.0,
                 min_alpha: float = 0.05,
                 gamma: float = 0.3):
        self.plotter = plotter
        self.default_color = default_color
        self.line_width = line_width
        self.min_alpha = min_alpha
        self.gamma = gamma

        # Пустая заготовка, чтобы PyVista не ругался
        temp_mesh = pv.PolyData(np.zeros((1, 3)))
        temp_mesh.point_data["colors"] = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        self.actor = plotter.add_mesh(
            temp_mesh,
            scalars="colors",
            rgba=True,
            line_width=line_width,
            render_lines_as_tubes=False,
            name="RayCloud"
        )
        self.actor.mapper.dataset.copy_from(pv.PolyData())  # очищаем

    # ---------- вспомогательные методы ----------
    @staticmethod
    def _to_rgb(color):
        from matplotlib.colors import to_rgb
        return np.array(to_rgb(color), dtype=np.float32)

    def _energy_to_alpha(self, energy: float, etype: int) -> float:
        if etype == 0:
            return 1.0
        elif etype == 1:
            return np.clip(energy, 0.0, 1.0)
        elif etype == 2:
            return max(self.min_alpha, energy ** self.gamma)
        else:
            return 1.0

    def _build_rgba(self, color, alpha: float) -> np.ndarray:
        rgb = self._to_rgb(color)
        return np.array([*rgb, alpha], dtype=np.float32)

    # ---------- основной метод обновления ----------
    def update(self, segments: List[Segment], energy_color_type: int = 1):
        """Принимает список Segment и обновляет геометрию."""
        if not segments:
            self.actor.mapper.dataset.copy_from(pv.PolyData())
            return

        points, lines, offset = [], [], 0
        rgba_list = []

        for seg in segments:
            p1, p2 = seg.start, seg.end
            if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
                continue
            points.append(p1)
            points.append(p2)
            lines.append([2, offset, offset + 1])

            alpha = self._energy_to_alpha(seg.energy, energy_color_type)
            rgba = self._build_rgba(seg.color, alpha)
            rgba_list.extend([rgba, rgba])
            offset += 2

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


class RayTracer:
    """
    Управляет лучами, элементами сцены и запускает трассировку через
    переданный объект TraceMode. Результат всегда – список Segment.
    """

    def __init__(self,
                 plotter: pv.Plotter,
                 mode = 'simple',          # строка или экземпляр TraceMode
                 pool: Optional[RayPool] = None,
                 **cloud_kwargs):
        """
        Параметры
        ---------
        plotter : pv.Plotter
        mode : str | TraceMode
            Режим трассировки. 'simple', 'tree' или готовый объект.
        pool : RayPool или None
            Пул для переиспользования лучей (будет задействован позже).
        **cloud_kwargs
            Параметры для внутреннего RayCloud (цвета, прозрачность и т.д.).
        """
        self.plotter = plotter
        self.rays = []
        self.elements = []
        self.emitters = []
        self.pool = pool

        # Нормализуем режим
        self.set_mode(mode)

        # Единое облако отрезков
        self.cloud = RayCloud(plotter, **cloud_kwargs)

    # -----------------------------------------------------------------
    # Удобное управление режимом
    # -----------------------------------------------------------------
    def set_mode(self, mode):
        """Задать режим трассировки строкой или объектом TraceMode."""
        if isinstance(mode, TraceMode):
            self._mode = mode
        elif mode == 'simple':
            self._mode = SimpleMode(pool=self.pool)
        elif mode == 'tree':
            self._mode = TreeMode(pool=self.pool)
        else:
            raise ValueError("mode must be 'simple', 'tree' or a TraceMode instance")

    @property
    def mode(self):
        """Текущий режим (объект TraceMode)."""
        return self._mode

    @mode.setter
    def mode(self, value):
        self.set_mode(value)

    # -----------------------------------------------------------------
    # Добавление объектов
    # -----------------------------------------------------------------
    def add_ray(self, ray: 'Ray'):
        self.rays.append(ray)

    def add_elements(self, *elements):
        for el in elements:
            self.elements.append(el)

    def add_emitter(self, emitter: 'BeamEmitter'):
        if not hasattr(self, 'emitters'):
            self.emitters = []
        self.emitters.append(emitter)

    # -----------------------------------------------------------------
    # Основной цикл трассировки
    # -----------------------------------------------------------------
    def trace_all(self) -> List[Segment]:
        """
        Прогоняет все зарегистрированные лучи (включая лучи от эмиттеров)
        через текущий режим трассировки.
        Возвращает плоский список Segment, готовый к визуализации.
        """
        # Сначала собираем лучи от эмиттеров (если есть)
        for emitter in self.emitters:
            for ray in emitter.emit():
                self.rays.append(ray)

        all_segments = []
        for ray in self.rays:
            segments = self._mode.trace(ray, self.elements)
            all_segments.extend(segments)

        return all_segments

    def render(self):
        segments = self.trace_all()
        energy_type = self._mode.energy_color_type
        self.cloud.update(segments, energy_color_type=energy_type)

        if self.pool:
            for ray in self.rays:
                self.pool.release(ray)
        self.rays.clear()
        return self.cloud

    def remove(self):
        """Удалить облако отрезков со сцены."""
        self.plotter.remove_actor(self.cloud.actor)


class BeamEmitter:
    """
    Излучатель пучка лучей. Трансформируется (поворот, перемещение), может генерировать
    набор параллельных лучей или выдавать лучи из заданного пользователем списка.
    Поддерживает обычные лучи, DispersiveRay и WhiteRay.
    """

    def __init__(self, origin, direction=np.array([1.0, 0.0, 0.0]),
                 rotation_degrees=(0, 0, 0), pool=None,
                 num_rays=5, min_offset=-2.0, max_offset=2.0,
                 color="yellow", wavelength=550, energy_color_type=2,
                 energy=1.0, current_n=1.0, ray_class=Ray,
                 num_spectral_bands=7,
                 spawn_as_spectral_pack=True):  # <-- НОВЫЙ ПАРАМЕТР: если True, WhiteRay сразу заменяется пучком
        self.origin = np.asarray(origin, dtype=float)

        # Сохраняем исходное базовое направление
        self.base_direction = np.asarray(direction, dtype=float)
        self.base_direction /= np.linalg.norm(self.base_direction)

        self.direction = self.base_direction.copy()

        # Задаем начальную матрицу
        self.rotation_matrix = np.eye(3)

        # Применяем начальный поворот
        self.rotate(rotation_degrees)

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
        self.custom_rays: List[Ray] = []
        self.use_custom = False
        self.ray_class = ray_class

        # Количество спектральных составляющих для WhiteRay
        self.num_spectral_bands = num_spectral_bands
        # Временное решение: разбивать ли белый луч на цветные сразу при создании
        self.spawn_as_spectral_pack = spawn_as_spectral_pack

    def add_ray(self, ray: Ray):
        """Добавить пользовательский луч (в локальной системе излучателя)."""
        self.custom_rays.append(ray)
        self.use_custom = True

    def rotate(self, angles_deg):
        # Рассчитываем чистую абсолютную матрицу поворота
        rot = R.from_euler('xyz', angles_deg, degrees=True).as_matrix()
        self.direction = rot @ self.base_direction
        self.rotation_matrix = rot

    def translate(self, vec):
        self.origin += np.asarray(vec)

    def emit(self) -> List[Ray]:
        rays = []
        local_perp1 = np.array([0.0, 1.0, 0.0])
        perp1 = self.rotation_matrix @ local_perp1
        perp1 /= np.linalg.norm(perp1)

        offsets = np.linspace(self.min_offset, self.max_offset, self.num_rays)

        for dy in offsets:
            world_origin = self.origin + dy * perp1

            # Если выбран класс WhiteRay и включена опция "сразу создавать пучок"
            if self.ray_class is WhiteRay and self.spawn_as_spectral_pack:
                wavelengths = np.linspace(400.0, 700.0, self.num_spectral_bands)
                sub_energy = self.energy / self.num_spectral_bands

                for wl in wavelengths:
                    if self.pool:
                        ray = self.pool.acquire(origin=world_origin,
                                                direction=self.direction,
                                                energy=sub_energy,
                                                current_n=self.current_n,
                                                color=self.color,
                                                wavelength=wl,
                                                energy_color_type=self.energy_color_type,
                                                polarization=None,
                                                ray_class=DispersiveRay)
                    else:
                        ray = DispersiveRay(origin=world_origin,
                                            direction=self.direction,
                                            energy=sub_energy,
                                            current_n=self.current_n,
                                            energy_color_type=self.energy_color_type,
                                            wavelength=wl)
                    rays.append(ray)
            else:
                # Стандартное создание одиночного луча (как у вас и было)
                if self.pool:
                    ray = self.pool.acquire(origin=world_origin,
                                            direction=self.direction,
                                            energy=self.energy,
                                            current_n=self.current_n,
                                            color=self.color,
                                            wavelength=self.wavelength,
                                            energy_color_type=self.energy_color_type,
                                            polarization=None,
                                            ray_class=self.ray_class,
                                            num_spectral_bands=self.num_spectral_bands)
                else:
                    if self.ray_class is WhiteRay:
                        ray = WhiteRay(origin=world_origin,
                                       direction=self.direction,
                                       num_spectral_bands=self.num_spectral_bands,
                                       energy=self.energy,
                                       current_n=self.current_n,
                                       energy_color_type=self.energy_color_type)
                    else:
                        ray = self.ray_class(origin=world_origin,
                                             direction=self.direction,
                                             energy=self.energy,
                                             current_n=self.current_n,
                                             color=self.color,
                                             wavelength=self.wavelength,
                                             energy_color_type=self.energy_color_type)
                rays.append(ray)
        return rays

    def get_mesh(self) -> pv.PolyData:
        """Маленькая стрелка для визуализации излучателя."""
        arrow = pv.Arrow(start=self.origin, direction=self.direction, scale=0.5)
        return arrow


class PlaneSurface:
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

        # НАЧАЛО ИЗМЕНЕНИЙ: Инициализация формы плоскости
        self.shape_type = shape_type  # "circle" или "rectangle"
        self.width = float(width)
        self.height = float(height)
        self.half_sizes = np.array([self.width / 2.0, self.height / 2.0]) if half_sizes is None else np.array(
            half_sizes)
        # КОНЕЦ ИЗМЕНЕНИЙ

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
        use_rect = (self.shape_type == "rectangle")
        if use_rect:
            tangents = np.array(self.face_tangents, dtype=np.float64)
            half = np.array([self.width / 2.0, self.height / 2.0], dtype=np.float64)
            # Если это прямоугольник, радиус круговой апертуры сбрасываем в 0.0,
            # чтобы ядро fast_math не выполняло круговую отсечку поверх прямоугольной
            r_aperture = 0.0
        else:
            tangents = np.zeros((2, 3), dtype=np.float64)
            half = np.zeros(2, dtype=np.float64)
            r_aperture = self.edge_radius

        t = plane_intersect(
            ray.origin, ray.direction, self.point, self.normal,
            self.lens_origin, self.lens_axis, r_aperture,
            half, tangents, use_rect
        )
        if t < 0.0:
            return None
        return float(t)

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
        in_ref = self.reflection_range is not None and self.reflection_range[0] is not None and self.reflection_range[
            1] is not None and (self.reflection_range[0] <= wavelength <= self.reflection_range[1])
        in_refr = self.refraction_range is not None and self.refraction_range[0] is not None and self.refraction_range[
            1] is not None and (self.refraction_range[0] <= wavelength <= self.refraction_range[1])
        in_abs = self.absorption_range is not None and self.absorption_range[0] is not None and self.absorption_range[
            1] is not None and (self.absorption_range[0] <= wavelength <= self.absorption_range[1])
        return in_ref or in_refr or in_abs

    def get_mesh(self) -> pv.PolyData:
        if self.shape_type == "rectangle":
            # ПРЯМОУГОЛЬНИК
            t1, t2 = self.face_tangents
            hu, hv = self.width / 2.0, self.height / 2.0
            c = self.lens_origin

            # Рассчитываем честные мировые координаты четырех углов
            p0 = c - hu * t1 - hv * t2
            p1 = c + hu * t1 - hv * t2
            p2 = c + hu * t1 + hv * t2
            p3 = c - hu * t1 + hv * t2

            vertices = np.array([p0, p1, p2, p3], dtype=np.float32)

            # ИСПРАВЛЕНИЕ: В PyVista массив граней должен начинаться с количества точек.
            # Для одного четырехугольника: [4, индекс0, индекс1, индекс2, индекс3]
            faces = np.array([4, 0, 1, 2, 3], dtype=np.int32)

            return pv.PolyData(vertices, faces)
        else:
            # КРУГЛЫЙ ДИСК
            radius = self.edge_radius if self.edge_radius else 1.0
            disc = pv.Disc(center=(0, 0, 0), normal=(1, 0, 0), inner=0, outer=radius, c_res=64)

            transform = np.eye(4)
            transform[:3, :3] = self.rotation_matrix
            transform[:3, 3] = self.lens_origin
            return disc.transform(transform, inplace=False)


class SphereSurface:
    """
    Сферическая поверхность с ограничениями по радиусу апертуры и продольному
    положению (толщине) вдоль оптической оси линзы.
    """
    def __init__(self, radius, rotation_degrees=(0,0,0), n_inside=1.0,
                 edge_radius=None, thickness=0.0,
                 reflection_range=None, refraction_range=None,
                 absorption_range=None,
                 lens_origin=None, lens_axis=None):
        """
        Параметры:
        radius : float
            Радиус кривизны. Положительный – выпуклая поверхность,
            отрицательный – вогнутая.
        rotation_degrees : tuple (3,)
            Углы Эйлера (в градусах) для поворота оптической оси.
        lens_origin : array-like (3,)
            Вершина поверхности (точка на оси, где поверхность пересекает ось).
            По умолчанию (0,0,0).
        """
        # Вершина
        if lens_origin is None:
            self.lens_origin = np.array([0.0, 0.0, 0.0])
        else:
            self.lens_origin = np.array(lens_origin, dtype=float)

        self.radius = radius

        # Оптическая ось – поворот базового вектора (1,0,0)
        if lens_axis is not None:
            self.lens_axis = np.array(lens_axis, dtype=float) / np.linalg.norm(lens_axis)
        else:
            base_axis = np.array([1.0, 0.0, 0.0])
            rot = R.from_euler('xyz', rotation_degrees, degrees=True).as_matrix()
            self.lens_axis = rot @ base_axis
            self.lens_axis /= np.linalg.norm(self.lens_axis)
        # Центр вычисляется однозначно:
        self.center = self.lens_origin - self.radius * self.lens_axis

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
        # t = sphere_intersect(ray.origin, ray.direction, self.center, self.radius,
        #                      self.lens_origin, self.lens_axis, self.edge_radius, self.thickness)
        # if t < 0.0:
        #     return None
        # return t
        return self._slow_intersect(ray)

    def _slow_intersect(self, ray: Ray) -> Optional[float]:
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

            # 1. Проверяем расстояние от ТОЧКИ УДАРА до ОПТИЧЕСКОЙ ОСИ (Апертура)
            vec_to_hit = hit_p - self.lens_origin
            projection = np.dot(vec_to_hit, self.lens_axis)
            dist_to_axis = np.linalg.norm(vec_to_hit - projection * self.lens_axis)

            # Принудительно приводим к float на случай, если из Trame просочилась строка
            r_max = float(self.edge_radius)

            if dist_to_axis <= r_max + 1e-5:
                # 2. Проверяем по глубине: точка должна лежать на нужной "чаше" сферы,
                # а не на ее противоположной зеркальной половине.
                # Для передней поверхности projection должен быть отрицательным/нулевым, для задней - положительным
                # Безопасная проверка: расстояние вдоль оси от вершины не должно превышать стрелку прогиба (sagitta)
                sagitta = abs(self.radius) - np.sqrt(max(0.0, abs(self.radius) ** 2 - r_max ** 2))

                if abs(projection) <= (sagitta + 1e-4):
                    valid_ts.append(t)

        return min(valid_ts) if valid_ts else None

    def get_normal(self, point: np.ndarray) -> np.ndarray:
        normal = (point - self.center) / self.radius
        return normal / np.linalg.norm(normal)

    def get_mesh(self) -> pv.PolyData:
        abs_radius = abs(self.radius)
        mesh = pv.Sphere(radius=abs_radius, center=(0, 0, 0),
                         phi_resolution=80, theta_resolution=80)
        sagitta = abs_radius - np.sqrt(max(0.0, abs_radius ** 2 - self.edge_radius ** 2))
        R = self.radius

        # Правильное отсечение нужной «чаши»
        if R > 0:
            # выпуклая: оставляем x >= R - sagitta (вершина в +R)
            mesh = mesh.clip(normal=[1, 0, 0], origin=[R - sagitta, 0, 0], invert=False)
        else:
            # вогнутая: оставляем x <= R + sagitta (вершина в отрицательной области R)
            mesh = mesh.clip(normal=[-1, 0, 0], origin=[R + sagitta, 0, 0], invert=False)

        # Поворот: локальную ось X направляем вдоль lens_axis (без отражения!)
        rot_matrix = calculate_rotation_matrix(self.lens_axis)

        # Мировой центр: вершина (R,0,0) должна совпасть с lens_origin
        world_center = self.lens_origin - self.radius * self.lens_axis

        matrix = np.eye(4)
        matrix[:3, :3] = rot_matrix
        matrix[:3, 3] = world_center
        return mesh.transform(matrix, inplace=False)

    def apply_transform(self, mat):
        R_mat = mat[:3, :3]
        t_mat = mat[:3, 3]
        self.lens_origin = R_mat @ self.lens_origin + t_mat
        self.lens_axis = R_mat @ self.lens_axis
        self.center = self.lens_origin - self.radius * self.lens_axis

    def is_active(self, wavelength):
        """Возвращает True, если поверхность должна взаимодействовать с данной длиной волны."""
        if wavelength is None:
            return True
        # Если ни один диапазон не задан, объект невидим (прозрачен) – нет взаимодействия
        if self.reflection_range is None and self.refraction_range is None and self.absorption_range is None:
            return False
        # Проверяем попадание хотя бы в один диапазон
        in_ref = self.reflection_range is not None and self.reflection_range[0] is not None and self.reflection_range[
            1] is not None and (self.reflection_range[0] <= wavelength <= self.reflection_range[1])
        in_refr = self.refraction_range is not None and self.refraction_range[0] is not None and self.refraction_range[
            1] is not None and (self.refraction_range[0] <= wavelength <= self.refraction_range[1])
        in_abs = self.absorption_range is not None and self.absorption_range[0] is not None and self.absorption_range[
            1] is not None and (self.absorption_range[0] <= wavelength <= self.absorption_range[1])
        return in_ref or in_refr or in_abs


class CylinderSurface:
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
        # НАЧАЛО ИЗМЕНЕНИЙ: Включаем или выключаем capping в меше PyVista
        cylinder = pv.Cylinder(center=self.center, direction=self.axis_dir,
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
        if self.capping: self._update_caps() # Пересчитываем крышк

    def is_active(self, wavelength):
        if wavelength is None:
            return True
        if self.reflection_range is None and self.refraction_range is None and self.absorption_range is None:
            return False
        in_ref = self.reflection_range is not None and self.reflection_range[0] is not None and self.reflection_range[
            1] is not None and (self.reflection_range[0] <= wavelength <= self.reflection_range[1])
        in_refr = self.refraction_range is not None and self.refraction_range[0] is not None and self.refraction_range[
            1] is not None and (self.refraction_range[0] <= wavelength <= self.refraction_range[1])
        in_abs = self.absorption_range is not None and self.absorption_range[0] is not None and self.absorption_range[
            1] is not None and (self.absorption_range[0] <= wavelength <= self.absorption_range[1])
        return in_ref or in_refr or in_abs


class MeshSurface:
    """
    Произвольная треугольная поверхность, загружаемая из файла или создаваемая из меша.
    Может быть зеркальной, преломляющей или поглощающей.
    """

    def __init__(self, mesh, rotation_degrees=(0, 0, 0), translation=(0, 0, 0),
                 n_inside=1.0, reflection_range=None, refraction_range=None,
                 absorption_range=None, scale_factors=(1.0, 1.0, 1.0)):
        # Загрузка тримеша
        if isinstance(mesh, str):
            self.trimesh_obj = trimesh.load(mesh)
            if isinstance(self.trimesh_obj, trimesh.Scene):
                self.trimesh_obj = trimesh.util.concatenate(
                    [g for g in self.trimesh_obj.geometry.values() if isinstance(g, trimesh.Trimesh)])
            if not isinstance(self.trimesh_obj, trimesh.Trimesh):
                raise TypeError("Файл не содержит треугольной сетки")
        elif isinstance(mesh, trimesh.Trimesh):
            self.trimesh_obj = mesh.copy()  # Копируем, чтобы не портить исходник
        elif isinstance(mesh, pv.PolyData):
            verts, faces = mesh.points, mesh.faces.reshape(-1, 4)[:, 1:4]
            self.trimesh_obj = trimesh.Trimesh(vertices=verts, faces=faces)
        else:
            raise TypeError("mesh должен быть str, trimesh.Trimesh или pv.PolyData")

        # СОХРАНЯЕМ ИСХОДНЫЕ ЭТАЛОННЫЕ НОРМАЛИ (до любых деформаций)
        # Они нужны, чтобы при изменении ползунков не накапливалась ошибка
        self._base_face_normals = self.trimesh_obj.face_normals.copy()

        # Применяем масштаб ДО поворотов и переносов
        if scale_factors is not None and not np.allclose(scale_factors, 1.0):
            # ВНИМАНИЕ: Нам нужно построить правильную диагональную матрицу 4x4
            scale_mat = np.eye(4)
            scale_mat[0, 0] = scale_factors[0]
            scale_mat[1, 1] = scale_factors[1]
            scale_mat[2, 2] = scale_factors[2]
            self.trimesh_obj.apply_transform(scale_mat)

            # Корректируем нормали под этот масштаб
            self._apply_optical_normals_scale(scale_factors)

        # Применяем поворот и перенос
        rot_4x4 = np.eye(4)
        rot_4x4[:3, :3] = R.from_euler('xyz', rotation_degrees, degrees=True).as_matrix()
        self.trimesh_obj.apply_transform(rot_4x4)

        # Корректируем нормали под поворот (просто умножаем на матрицу вращения)
        if 'face_normals' in self.trimesh_obj._cache:
            current_normals = self.trimesh_obj._cache['face_normals']
            rotated_normals = current_normals @ rot_4x4[:3, :3].T
            self.trimesh_obj._cache['face_normals'] = rotated_normals

        if translation is not None:
            self.trimesh_obj.apply_translation(translation)

        self.mesh = self.trimesh_obj
        self.intersector = trimesh.ray.ray_triangle.RayMeshIntersector(self.mesh)
        self.n = n_inside
        self.reflection_range = reflection_range
        self.refraction_range = refraction_range
        self.absorption_range = absorption_range
        self._last_hit_triangle_idx = None

    def _apply_optical_normals_scale(self, scale_factors):
        """Внутренний метод пересчета нормалей по законам оптики."""
        # Для нормалей используется транспонированная обратная матрица масштабных коэффициентов
        inv_scale = 1.0 / np.array(scale_factors, dtype=float)

        # Модифицируем базовые нормали
        scaled_normals = self._base_face_normals * inv_scale

        # Нормализуем каждый вектор нормали (приводим к длине 1)
        norms = np.linalg.norm(scaled_normals, axis=1, keepdims=True)
        # Защита от деления на 0
        norms = np.where(norms < 1e-12, 1.0, norms)
        correct_normals = scaled_normals / norms

        # Принудительно жестко записываем их в кэш trimesh.
        # Теперь trimesh будет брать их отсюда, а не считать геометрически!
        self.trimesh_obj._cache['face_normals'] = correct_normals

    def scale(self, scale_factors):
        """Применяет оптически корректное масштабирование к мешу относительно его центра."""
        # Строим правильную матрицу масштаба 4x4
        scale_mat = np.eye(4)
        scale_mat[0, 0] = scale_factors[0]
        scale_mat[1, 1] = scale_factors[1]
        scale_mat[2, 2] = scale_factors[2]

        center = self.mesh.bounding_box.center_mass
        T1 = np.eye(4)
        T1[:3, 3] = -center
        T2 = np.eye(4)
        T2[:3, 3] = center

        # Очищаем старый кэш, чтобы trimesh не сопротивлялся изменениям
        self.mesh._cache.clear()

        # Смещаем, скейлим вершины, возвращаем
        self.mesh.apply_transform(T2 @ scale_mat @ T1)

        # Накатываем корректные нормали, рассчитанные через инверсию масштаба
        self._apply_optical_normals_scale(scale_factors)

        # Перестраиваем BVH-дерево пересечений для лучей
        self.intersector = trimesh.ray.ray_triangle.RayMeshIntersector(self.mesh)

    def is_active(self, wavelength):
        if wavelength is None:
            return True
        if self.reflection_range is None and self.refraction_range is None and self.absorption_range is None:
            return False
        in_ref = self.reflection_range is not None and self.reflection_range[0] is not None and self.reflection_range[
            1] is not None and (self.reflection_range[0] <= wavelength <= self.reflection_range[1])
        in_refr = self.refraction_range is not None and self.refraction_range[0] is not None and self.refraction_range[
            1] is not None and (self.refraction_range[0] <= wavelength <= self.refraction_range[1])
        in_abs = self.absorption_range is not None and self.absorption_range[0] is not None and self.absorption_range[
            1] is not None and (self.absorption_range[0] <= wavelength <= self.absorption_range[1])
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
        """Возвращает строго нормализованный вектор нормали из кэша."""
        if self._last_hit_triangle_idx is not None:
            raw_normal = self.mesh.face_normals[self._last_hit_triangle_idx]
        else:
            _, _, tri_idx = trimesh.proximity.closest_point(self.mesh, [point])
            raw_normal = self.mesh.face_normals[tri_idx[0]]

        norm = np.linalg.norm(raw_normal)
        if norm < 1e-12:
            return raw_normal
        return raw_normal / norm

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

    def intersect(self, ray: Ray) -> Optional[float]:
        # Переводим луч в локальные координаты
        origin_loc = self._world_to_local(ray.origin)
        dir_loc = self._rot_world_to_local @ ray.direction
        # Нормализуем направление на всякий случай
        dir_loc = dir_loc / np.linalg.norm(dir_loc)

        sag_max = self.sag(self.edge_radius) if self.edge_radius > 0 else 0.0

        # Проверка: если луч почти параллелен плоскости и не попадает в апертуру
        # (быстрый отсев для производительности)
        if abs(dir_loc[0]) < 1e-9:
            # Луч перпендикулярен оси, может пересечь только край, но это редко
            # Пропускаем, чтобы не застревать
            pass

        t = self._intersect_numba(
            origin_loc.astype(np.float64),
            dir_loc.astype(np.float64),
            self.radius,
            self.k,
            np.array(self.aspheric_coeffs, dtype=np.float64),
            self.edge_radius,
            sag_max
        )
        if t < 0.0:
            return None
        return float(t)

    @staticmethod
    @njit
    def _intersect_numba(origin, direction, radius, k, coeffs, edge_radius, sag_max):
        oy, oz = origin[1], origin[2]
        dy, dz = direction[1], direction[2]
        a_cyl = dy * dy + dz * dz

        # 1. Луч строго параллелен локальной оси X (dy=0, dz=0)
        if a_cyl < 1e-12:
            r0 = np.sqrt(oy * oy + oz * oz)
            if r0 > edge_radius + 1e-6:
                return -1.0  # луч не попадает в апертуру
            if abs(direction[0]) < 1e-12:
                return -1.0  # луч перпендикулярен оси - пересечения нет
            # Явное решение: t = (sag(r0) - x0) / dx
            sag0 = _sag_numba(r0, radius, k, coeffs)
            t = (sag0 - origin[0]) / direction[0]
            if t <= 1e-6:
                return -1.0
            return t

        # 2. Общий случай: находим интервал пересечения с апертурой
        b_cyl = 2.0 * (oy * dy + oz * dz)
        c_cyl = oy * oy + oz * oz - edge_radius * edge_radius
        disc_cyl = b_cyl * b_cyl - 4.0 * a_cyl * c_cyl
        if disc_cyl < 0.0:
            return -1.0

        sqrt_disc_cyl = np.sqrt(disc_cyl)
        t1 = (-b_cyl - sqrt_disc_cyl) / (2.0 * a_cyl)
        t2 = (-b_cyl + sqrt_disc_cyl) / (2.0 * a_cyl)
        t1, t2 = min(t1, t2), max(t1, t2)

        if t2 <= 1e-6:
            return -1.0

        t_enter = max(0.0, t1)
        t_exit = t2
        if t_enter > t_exit:
            return -1.0

        # Определяем значения функции F(t) на границах
        x_enter = origin[0] + t_enter * direction[0]
        x_exit = origin[0] + t_exit * direction[0]
        r_enter = np.sqrt((origin[1] + t_enter * dy) ** 2 + (origin[2] + t_enter * dz) ** 2)
        r_exit = np.sqrt((origin[1] + t_exit * dy) ** 2 + (origin[2] + t_exit * dz) ** 2)

        F_enter = x_enter - _sag_numba(r_enter, radius, k, coeffs)
        F_exit = x_exit - _sag_numba(r_exit, radius, k, coeffs)

        t_root = -1.0

        # Поиск корня методом бисекции
        if abs(F_enter) < 1e-9:
            t_root = t_enter
        elif abs(F_exit) < 1e-9:
            t_root = t_exit
        elif F_enter * F_exit < 0.0:
            lo, hi = t_enter, t_exit
            flo, fhi = F_enter, F_exit
            for _ in range(60):
                mid = (lo + hi) * 0.5
                x_mid = origin[0] + mid * direction[0]
                r_mid = np.sqrt((origin[1] + mid * dy) ** 2 + (origin[2] + mid * dz) ** 2)
                fmid = x_mid - _sag_numba(r_mid, radius, k, coeffs)
                if abs(fmid) < 1e-9 or (hi - lo) < 1e-12:
                    t_root = mid
                    break
                if flo * fmid < 0.0:
                    hi = mid
                    fhi = fmid
                else:
                    lo = mid
                    flo = fmid
            if t_root < 0.0:
                t_root = (lo + hi) * 0.5
        else:
            # 3. Защита от ошибок округления на краю апертуры (edge_radius)
            # Расширяем диапазон на 0.01 и пробуем снова
            t_lo_try = max(0.0, t_enter - 0.01)
            t_hi_try = t_exit + 0.01

            x_lo = origin[0] + t_lo_try * direction[0]
            r_lo = np.sqrt((origin[1] + t_lo_try * dy) ** 2 + (origin[2] + t_lo_try * dz) ** 2)
            F_lo = x_lo - _sag_numba(r_lo, radius, k, coeffs)

            x_hi = origin[0] + t_hi_try * direction[0]
            r_hi = np.sqrt((origin[1] + t_hi_try * dy) ** 2 + (origin[2] + t_hi_try * dz) ** 2)
            F_hi = x_hi - _sag_numba(r_hi, radius, k, coeffs)

            if F_lo * F_hi < 0.0:
                lo, hi = t_lo_try, t_hi_try
                flo, fhi = F_lo, F_hi
                for _ in range(60):
                    mid = (lo + hi) * 0.5
                    x_mid = origin[0] + mid * direction[0]
                    r_mid = np.sqrt((origin[1] + mid * dy) ** 2 + (origin[2] + mid * dz) ** 2)
                    fmid = x_mid - _sag_numba(r_mid, radius, k, coeffs)
                    if abs(fmid) < 1e-9 or (hi - lo) < 1e-12:
                        t_root = mid
                        break
                    if flo * fmid < 0.0:
                        hi = mid
                        fhi = fmid
                    else:
                        lo = mid
                        flo = fmid
                if t_root < 0.0:
                    t_root = (lo + hi) * 0.5
            else:
                return -1.0

        if t_root < 0.0:
            return -1.0

        # 4. Финальная проверка
        p_final = origin + t_root * direction
        r_final = np.sqrt(p_final[1] ** 2 + p_final[2] ** 2)
        if r_final > edge_radius + 1e-6:
            return -1.0

        # Защита от пересечения с обратной стороной (допускаем погрешность 1e-6)
        sag_min = min(0.0, sag_max)
        sag_max_val = max(0.0, sag_max)
        if p_final[0] < sag_min - 1e-6 or p_final[0] > sag_max_val + 1e-6:
            return -1.0

        return t_root

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
        in_ref = self.reflection_range is not None and self.reflection_range[0] is not None and self.reflection_range[
            1] is not None and (self.reflection_range[0] <= wavelength <= self.reflection_range[1])
        in_refr = self.refraction_range is not None and self.refraction_range[0] is not None and self.refraction_range[
            1] is not None and (self.refraction_range[0] <= wavelength <= self.refraction_range[1])
        in_abs = self.absorption_range is not None and self.absorption_range[0] is not None and self.absorption_range[
            1] is not None and (self.absorption_range[0] <= wavelength <= self.absorption_range[1])
        return in_ref or in_refr or in_abs


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
        rotation_degrees – углы Эйлера поворота линзы,
        R1, R2      – радиусы кривизны передней и задней поверхностей (None = плоскость),
        thickness   – толщина вдоль оси по центру,
        edge_radius – радиус апертуры,
        n           – показатель преломления материала.
    """

    def __init__(self, origin, rotation_degrees=(0, 0, 0), R1=None, R2=None,
                 thickness=2.0, edge_radius=3.0, n=1.5,
                 reflection_range=None, refraction_range=(0, np.inf),
                 absorption_range=None):
        self.origin = np.array(origin, dtype=float)
        self.rotation_degrees = rotation_degrees
        self.rotation = R.from_euler('xyz', rotation_degrees, degrees=True).as_matrix()
        self.axis_dir = self.rotation @ np.array([1.0, 0.0, 0.0])
        self.thickness = thickness
        self.edge_radius = edge_radius
        self.n = n
        self.R1, self.R2 = R1, R2
        self.reflection_range = reflection_range
        self.refraction_range = refraction_range
        self.absorption_range = absorption_range

        self.debug_cylinder_actor = None

        self._create_surfaces()
        self._calc_optical_params()

    def _create_surfaces(self):
        r = float(self.edge_radius)

        # Минимальная технологическая толщина бокового ободка (чтобы линза не схлопывалась в ноль на краях)
        MIN_RIM_THICKNESS = 0.5  # или self.thickness * 0.1

        # 1. Вычисляем стрелки прогиба (sagitta) для передней и задней поверхностей
        sag1 = 0.0
        if self.R1 is not None:
            # Ограничиваем радиус, чтобы избежать деления на ноль / корня из отрицательного числа на полусфере
            effective_R1 = max(abs(self.R1), r + 1e-5)
            sag1 = effective_R1 - np.sqrt(effective_R1 ** 2 - r ** 2)
            if self.R1 < 0:
                sag1 = -sag1

        sag2 = 0.0
        if self.R2 is not None:
            effective_R2 = max(abs(self.R2), r + 1e-5)
            sag2 = effective_R2 - np.sqrt(effective_R2 ** 2 - r ** 2)
            if self.R2 < 0:
                sag2 = -sag2

        self.internal_thickness = self.thickness

        # Пересчитываем параметры на основе скорректированной внутренней толщины
        half = self.internal_thickness / 2
        edge_thickness = self.internal_thickness - (sag1 if self.R1 else 0.0) - (sag2 if self.R2 else 0.0)
        half_rim_length = edge_thickness / 2.0

        # Корректируем смещение центра ободка (для несимметричных крутых линз)
        rim_center_x = (sag1 - sag2) / 2.0 if (self.R1 and self.R2) else 0.0
        local_cylinder_center = np.array([rim_center_x, 0.0, 0.0])

        # Строим мировую матрицу трансформации 4x4
        mat = np.eye(4)
        mat[:3, :3] = self.rotation
        mat[:3, 3] = self.origin

        # 3. Инициализируем поверхности с учётом новой внутренней толщины `self.internal_thickness`
        if self.R1 is None:
            self.front = PlaneSurface(
                point=[-half, 0, 0], normal=[-1.0, 0.0, 0.0],
                n_inside=self.n, edge_radius=self.edge_radius,
                reflection_range=self.reflection_range, refraction_range=self.refraction_range,
                absorption_range=self.absorption_range
            )
        else:
            self.front = SphereSurface(
                radius=self.R1, n_inside=self.n, edge_radius=self.edge_radius,
                thickness=self.internal_thickness, reflection_range=self.reflection_range,
                refraction_range=self.refraction_range, absorption_range=self.absorption_range,
                lens_origin=[-half, 0, 0], lens_axis=[-1.0, 0.0, 0.0]
            )

        if self.R2 is None:
            self.back = PlaneSurface(
                point=[half, 0, 0], normal=[1.0, 0.0, 0.0],
                n_inside=self.n, edge_radius=self.edge_radius,
                reflection_range=self.reflection_range, refraction_range=self.refraction_range,
                absorption_range=self.absorption_range
            )
        else:
            self.back = SphereSurface(
                radius=self.R2, n_inside=self.n, edge_radius=self.edge_radius,
                thickness=self.internal_thickness, reflection_range=self.reflection_range,
                refraction_range=self.refraction_range, absorption_range=self.absorption_range,
                lens_origin=[half, 0, 0], lens_axis=[1.0, 0.0, 0.0]
            )

        # Боковой цилиндр встает ровно посередине между краями «чаш»
        self.cylinder = CylinderSurface(
            center=local_cylinder_center, axis_dir=[1.0, 0.0, 0.0], radius=self.edge_radius,
            half_length=half_rim_length, n_inside=self.n, reflection_range=self.reflection_range,
            refraction_range=self.refraction_range, absorption_range=self.absorption_range
        )

        # Применяем трансформации к мировым координатам
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

        # Используем внутреннюю скорректированную толщину вместо исходной
        v1_local = -self.internal_thickness / 2
        v2_local = self.internal_thickness / 2

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

        rim_x = np.array([x_front[:, -1], x_back[:, -1]])
        rim_y = np.array([y[:, -1], y[:, -1]])
        rim_z = np.array([z[:, -1], z[:, -1]])
        rim_mesh = pv.StructuredGrid(rim_x, rim_y, rim_z).extract_surface(algorithm='dataset_surface')

        local_mesh = front_mesh.merge(back_mesh).merge(rim_mesh)

        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.origin
        return local_mesh.transform(matrix, inplace=False)

    def _calc_optical_params(self):
        r1_val = self.R1 if self.R1 else 1e10
        r2_val = -self.R2 if self.R2 else -1e10
        inv_f = (self.n - 1) * (1 / r1_val - 1 / r2_val +
                                ((self.n - 1) * self.thickness) / (self.n * r1_val * r2_val))
        self.f_dist = 1 / inv_f if abs(inv_f) > 1e-10 else float('inf')

    def intersect(self, ray: Ray) -> Optional[float]:
        t_front = self.front.intersect(ray)
        t_back = self.back.intersect(ray)
        t_cyl = self.cylinder.intersect(ray)

        self._last_hit_surface = None
        best_t = None

        if t_front is not None:
            best_t = t_front
            self._last_hit_surface = self.front

        if t_back is not None and (best_t is None or t_back < best_t):
            best_t = t_back
            self._last_hit_surface = self.back

        if t_cyl is not None and (best_t is None or t_cyl < best_t):
            best_t = t_cyl
            self._last_hit_surface = self.cylinder

        return best_t

    def get_normal(self, point: np.ndarray) -> np.ndarray:
        if self._last_hit_surface is not None:
            return self._last_hit_surface.get_normal(point)
        return self.front.get_normal(point)

    def is_active(self, wavelength) -> bool:
        return (self.front.is_active(wavelength) or
                self.back.is_active(wavelength) or
                self.cylinder.is_active(wavelength))

    def rotate(self, angles_deg):
        rot = R.from_euler('xyz', angles_deg, degrees=True).as_matrix()
        self.axis_dir = rot @ self.axis_dir
        self.axis_dir /= np.linalg.norm(self.axis_dir)
        self.rotation = rot @ self.rotation
        self._create_surfaces()

    def translate(self, vec):
        self.origin += np.asarray(vec)
        self._create_surfaces()

    def get_surfaces(self) -> List:
        return [self.front, self.back, self.cylinder]

    def debug_draw_analytical_cylinder(self, plot: pv.Plotter, color="red", opacity=0.4):
        """
        Метод отладки. Добавляет на сцену реальный аналитический цилиндр
        из CylinderSurface, используемый в расчете трассировки лучей.
        """

        if self.debug_cylinder_actor:
            print("Clear")
            plot.remove_actor(self.debug_cylinder_actor)

        # Запрашиваем меш напрямую у объекта CylinderSurface
        cylinder_mesh = self.cylinder.get_mesh()

        # Добавляем на сцену как каркас (wireframe) или полупрозрачный объект
        self.debug_cylinder_actor = plot.add_mesh(
            cylinder_mesh,
            color=color,
            opacity=opacity,
            style="wireframe",
            line_width=2,
            name=f"debug_cyl"
        )

        # Отрисуем ребра-границы цилиндра для наглядности длины half_rim_length
        half_l = self.cylinder.half_length
        c = self.cylinder.center
        d = self.cylinder.axis_dir


class HyperbolicLens:
    """
    Плоско-гиперболическая (асферическая) линза в произвольной ориентации.
    Параметры:
        radius_of_curvature (R) - радиус кривизны при вершине
        n - показатель преломления
        f_target - фокусное расстояние
    Коническая константа k рассчитывается автоматически на основе геометрических параметров.
    """

    def __init__(self, origin, radius_of_curvature=None, rotation_degrees=(0, 0, 0),
                 thickness=2.0, edge_radius=3.0, n=1.5, f_target=10.0,
                 reflection_range=None, refraction_range=(0, np.inf),
                 absorption_range=None):
        self.origin = np.array(origin, dtype=float)
        self.rotation_degrees = rotation_degrees
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
        self.debug_cylinder_actor = None

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

        # Динамический аффинный трансформ для ядра асферики
        def front_apply_transform(m):
            R_mat = m[:3, :3]
            t_mat = m[:3, 3]
            self.front.lens_origin = R_mat @ self.front.lens_origin + t_mat
            self.front.lens_axis = R_mat @ self.front.lens_axis
            self.front.center = R_mat @ self.front.center + t_mat
            self.front._t1, self.front._t2 = get_tangents(self.front.lens_axis)
            self.front._rot_local_to_world = np.column_stack([self.front.lens_axis, self.front._t1, self.front._t2])
            self.front._rot_world_to_local = self.front._rot_local_to_world.T

        self.front.apply_transform = front_apply_transform

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

    def intersect(self, ray: Ray) -> Optional[float]:
        t_front = self.front.intersect(ray)
        t_back = self.back.intersect(ray)
        t_cyl = self.cylinder.intersect(ray)

        self._last_hit_surface = None
        best_t = None

        if t_front is not None:
            best_t = t_front
            self._last_hit_surface = self.front

        if t_back is not None and (best_t is None or t_back < best_t):
            best_t = t_back
            self._last_hit_surface = self.back

        if t_cyl is not None and (best_t is None or t_cyl < best_t):
            best_t = t_cyl
            self._last_hit_surface = self.cylinder

        return best_t

    def get_normal(self, point: np.ndarray) -> np.ndarray:
        if self._last_hit_surface is not None:
            return self._last_hit_surface.get_normal(point)
        return self.front.get_normal(point)

    def is_active(self, wavelength) -> bool:
        return (self.front.is_active(wavelength) or
                self.back.is_active(wavelength) or
                self.cylinder.is_active(wavelength))

    def rotate(self, angles_deg):
        self.rotation_degrees = tuple(np.array(self.rotation_degrees) + np.array(angles_deg))
        rot = R.from_euler('xyz', angles_deg, degrees=True).as_matrix()
        self.axis_dir = rot @ self.axis_dir
        self.axis_dir /= np.linalg.norm(self.axis_dir)
        self.rotation = rot @ self.rotation
        self._create_surfaces()

    def translate(self, vec):
        self.origin += np.asarray(vec)
        self._create_surfaces()

    def get_surfaces(self) -> List:
        return [self.front, self.back, self.cylinder]

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

        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.origin
        return local_mesh.transform(matrix, inplace=False)

    def debug_draw_analytical_cylinder(self, plot: pv.Plotter, color="red", opacity=0.4):
        actor_name = f"debug_cyl_{id(self)}"
        if actor_name in plot.actors:
            plot.remove_actor(actor_name)

        cylinder_mesh = self.cylinder.get_mesh()
        self.debug_cylinder_actor = plot.add_mesh(
            cylinder_mesh,
            color=color,
            opacity=opacity,
            style="wireframe",
            line_width=2,
            name=actor_name
        )


# --------------------------------
# Трассировка лучей и визуализация
# --------------------------------

def find_best_hit(ray: 'Ray', elements: List) -> Optional[HitInfo]:
    """
    Находит ближайшее пересечение луча с элементами сцены с учётом
    спектральной активности и возвращает HitInfo или None.
    """
    best_t = float('inf')
    best_hit = None

    for obj in elements:
        # Проверка активности по длине волны
        if hasattr(obj, 'is_active') and not obj.is_active(ray.wavelength):
            continue

        t = obj.intersect(ray)
        if t is not None and t < best_t:
            best_t = t
            hit_point = ray.origin + ray.direction * t
            normal = obj.get_normal(hit_point) if hasattr(obj, 'get_normal') else np.array([0., 0., 0.])

            # Определяем разрешённые действия на основе диапазонов
            allow_reflection = False
            allow_refraction = False
            absorbed = False
            n_inside = 1.0
            is_thin_lens = isinstance(obj, ThinLens)

            if hasattr(obj, 'absorption_range') and obj.absorption_range is not None:
                if ray.wavelength is None or (obj.absorption_range[0] <= ray.wavelength <= obj.absorption_range[1]):
                    absorbed = True

            if hasattr(obj, 'reflection_range') and obj.reflection_range is not None:
                if ray.wavelength is None or (obj.reflection_range[0] <= ray.wavelength <= obj.reflection_range[1]):
                    allow_reflection = True

            if hasattr(obj, 'refraction_range') and obj.refraction_range is not None:
                if ray.wavelength is None or (obj.refraction_range[0] <= ray.wavelength <= obj.refraction_range[1]):
                    allow_refraction = True

            if hasattr(obj, 'n'):
                n_inside = obj.n

            # Если объект полностью прозрачен (нет взаимодействия), пропускаем его
            if not is_thin_lens and not allow_reflection and not allow_refraction and not absorbed:
                continue

            best_hit = HitInfo(
                obj=obj,
                t=t,
                point=hit_point,
                normal=normal,
                allow_reflection=allow_reflection,
                allow_refraction=allow_refraction,
                absorbed=absorbed,
                n_inside=n_inside,
                is_thin_lens=is_thin_lens
            )

    return best_hit


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


def _trace_simple(ray: 'Ray',
                  elements: List,
                  max_bounces: int = 10,
                  offset_distance: float = 0.01,
                  prioritize_refraction: bool = True,
                  pool: Optional['RayPool'] = None) -> List[Segment]:
    """
    Однолучевая последовательная трассировка.
    При встрече с WhiteRay автоматически разветвляет симуляцию для спектра.
    """
    segments: List[Segment] = []

    if isinstance(ray, WhiteRay) and hasattr(ray, 'spectral_rays'):
        # 1. Находим пересечение для единого белого луча
        hit = find_best_hit(ray, elements)

        if hit is None:
            end_point = ray.origin + ray.direction * RAY_INFINITY_DISTANCE
            segments.append(Segment(ray.origin.copy(), end_point.copy(), ray.energy, ray.color))
            return segments

        # Добавляем начальный белый отрезок от источника до текущей поверхности
        segments.append(Segment(ray.origin.copy(), hit.point.copy(), ray.energy, ray.color))

        if hit.absorbed:
            return segments

        # Проверяем, какое действие приоритетно на этой поверхности
        allow_reflection = hit.allow_reflection
        allow_refraction = hit.allow_refraction
        if prioritize_refraction and allow_refraction:
            allow_reflection = False

        # --- СЛУЧАЙ А: ЧИСТОЕ ПРЕЛОМЛЕНИЕ (РАЗДЕЛЕНИЕ НА СПЕКТР) ---
        if allow_refraction:
            # Переводим каждый скрытый спектральный луч в точку удара и трассируем отдельно
            for disp_ray in ray.spectral_rays:
                disp_ray.origin[:] = hit.point.copy()

                # Считаем показатель преломления для конкретного цвета
                resolved_n_inside = get_dispersion_n(hit.n_inside, disp_ray)
                n_next = resolved_n_inside if abs(disp_ray.current_n - 1.0) < 1e-6 else 1.0

                refracted_dir = refract(disp_ray.direction, hit.normal, disp_ray.current_n, n_next)

                if refracted_dir is not None:
                    # Успешное преломление: цветной луч летит внутрь
                    disp_ray.origin[:] = hit.point + offset_distance * refracted_dir
                    disp_ray.direction[:] = refracted_dir
                    disp_ray.current_n = n_next
                else:
                    # Полное внутреннее отражение для этой компоненты
                    normal = hit.normal
                    if np.dot(normal, disp_ray.direction) > 0: normal = -normal
                    ref_dir = disp_ray.direction - 2 * np.dot(disp_ray.direction, normal) * normal
                    ref_dir /= np.linalg.norm(ref_dir)

                    disp_ray.origin[:] = hit.point + offset_distance * ref_dir
                    disp_ray.direction[:] = ref_dir

                # Пускаем получившийся DispersiveRay дальше по цепочке
                sub_segments = _trace_simple(
                    ray=disp_ray,
                    elements=elements,
                    max_bounces=max_bounces - 1,
                    offset_distance=offset_distance,
                    prioritize_refraction=prioritize_refraction,
                    pool=pool
                )
                segments.extend(sub_segments)
            return segments

        # --- СЛУЧАЙ Б: ЧИСТОЕ ОТРАЖЕНИЕ (ЛУЧ ОСТАЕТСЯ БЕЛЫМ) ---
        elif allow_reflection:
            normal = hit.normal
            if np.dot(normal, ray.direction) > 0:
                normal = -normal
            reflected_dir = ray.direction - 2 * np.dot(ray.direction, normal) * normal
            reflected_dir /= np.linalg.norm(reflected_dir)

            # Двигаем сам белый луч дальше как единый объект
            ray.origin[:] = hit.point + offset_distance * reflected_dir
            ray.direction[:] = reflected_dir

            # Важно: синхронизируем внутренние спектральные лучи, чтобы они отражались вместе с белым
            ray.update_position(ray.origin, ray.direction)

            # Продолжаем итерацию для белого луча
            sub_segments = _trace_simple(
                ray=ray,
                elements=elements,
                max_bounces=max_bounces - 1,
                offset_distance=offset_distance,
                prioritize_refraction=prioritize_refraction,
                pool=pool
            )
            segments.extend(sub_segments)
            return segments

        # На случай если объект полностью прозрачный
        else:
            ray.origin[:] = hit.point + offset_distance * ray.direction
            ray.update_position(ray.origin, ray.direction)
            sub_segments = _trace_simple(ray, elements, max_bounces - 1, offset_distance, prioritize_refraction, pool)
            segments.extend(sub_segments)
            return segments

    # ОСТАЛЬНОЙ ВАШ НЕИЗМЕНЕННЫЙ КОД ДЛЯ ОБЫЧНЫХ ЛУЧЕЙ (Ray и DispersiveRay)
    current_ray = ray
    current_n = ray.current_n
    current_from_pool = False

    for _ in range(max_bounces):
        hit = find_best_hit(current_ray, elements)

        if hit is None:
            end_point = current_ray.origin + current_ray.direction * RAY_INFINITY_DISTANCE
            segments.append(Segment(current_ray.origin.copy(), end_point.copy(),
                                    current_ray.energy, current_ray.color))
            break

        segments.append(Segment(current_ray.origin.copy(), hit.point.copy(),
                                current_ray.energy, current_ray.color))

        if hit.absorbed:
            break

        if hit.is_thin_lens:
            new_dir = hit.obj.thin_lens_deflection(current_ray.direction, hit.point)
            start = hit.point + offset_distance * new_dir

            if pool:
                if current_from_pool:
                    pool.release(current_ray)
                next_ray = pool.acquire(origin=start, direction=new_dir,
                                        energy=current_ray.energy, current_n=current_n,
                                        color=current_ray.color, wavelength=current_ray.wavelength,
                                        energy_color_type=current_ray.energy_color_type, ray_class=type(current_ray))
                current_from_pool = True
            else:
                next_ray = type(current_ray)(start, new_dir,
                                             energy=current_ray.energy, current_n=current_n,
                                             color=current_ray.color, wavelength=current_ray.wavelength,
                                             energy_color_type=current_ray.energy_color_type)
                current_from_pool = False
            current_ray = next_ray
            continue

        allow_reflection = hit.allow_reflection
        allow_refraction = hit.allow_refraction

        if prioritize_refraction and allow_refraction:
            allow_reflection = False

        if not allow_reflection and not allow_refraction:
            start = hit.point + offset_distance * current_ray.direction
            if pool:
                if current_from_pool:
                    pool.release(current_ray)
                next_ray = pool.acquire(origin=start, direction=current_ray.direction,
                                        energy=current_ray.energy, current_n=current_n,
                                        color=current_ray.color, wavelength=current_ray.wavelength,
                                        energy_color_type=current_ray.energy_color_type, ray_class=type(current_ray))
                current_from_pool = True
            else:
                next_ray = type(current_ray)(start, current_ray.direction,
                                             energy=current_ray.energy, current_n=current_n,
                                             color=current_ray.color, wavelength=current_ray.wavelength,
                                             energy_color_type=current_ray.energy_color_type)
                current_from_pool = False
            current_ray = next_ray
            continue

        if allow_refraction:
            resolved_n_inside = get_dispersion_n(hit.n_inside, current_ray)
            n_next = resolved_n_inside if abs(current_n - 1.0) < 1e-6 else 1.0
            refracted_dir = refract(current_ray.direction, hit.normal, current_n, n_next)
            if refracted_dir is not None:
                current_n = n_next
                new_dir = refracted_dir
                allow_reflection = False
            else:
                allow_reflection = True

        if allow_reflection:
            normal = hit.normal
            if np.dot(normal, current_ray.direction) > 0:
                normal = -normal
            new_dir = current_ray.direction - 2 * np.dot(current_ray.direction, normal) * normal
            new_dir /= np.linalg.norm(new_dir)

        start = hit.point + offset_distance * new_dir
        if pool:
            if current_from_pool:
                pool.release(current_ray)
            next_ray = pool.acquire(origin=start, direction=new_dir,
                                    energy=current_ray.energy, current_n=current_n,
                                    color=current_ray.color, wavelength=current_ray.wavelength,
                                    energy_color_type=current_ray.energy_color_type, ray_class=type(current_ray))
            current_from_pool = True
        else:
            next_ray = type(current_ray)(start, new_dir,
                                         energy=current_ray.energy, current_n=current_n,
                                         color=current_ray.color, wavelength=current_ray.wavelength,
                                         energy_color_type=current_ray.energy_color_type)
            current_from_pool = False
        current_ray = next_ray

    if current_from_pool and pool:
        pool.release(current_ray)

    return segments

def _trace_recursive(ray: 'Ray',
                     elements: List,
                     depth: int,
                     min_energy: float = 0.01,
                     offset_distance: float = 0.01,
                     use_polarization_color: bool = False,
                     total_limit: int = 5000,
                     pool: Optional[RayPool] = None) -> List[Segment]:
    """
    Рекурсивная трассировка с ветвлением (дерево лучей).
    Возвращает плоский список отрезков Segment.
    """
    segments = []

    def recurse(current_ray: 'Ray', d: int, from_pool: bool):
        nonlocal segments

        if len(segments) >= total_limit or d <= 0 or current_ray.energy < min_energy:
            return

        if isinstance(current_ray, WhiteRay):
            hit = find_best_hit(current_ray, elements)
            if hit is None:
                end = current_ray.origin + current_ray.direction * RAY_INFINITY_DISTANCE
                segments.append(Segment(current_ray.origin.copy(), end.copy(),
                                        current_ray.energy, current_ray.color))
                return

            # Добавляем общий белый отрезок до геометрии
            segments.append(Segment(current_ray.origin.copy(), hit.point.copy(),
                                    current_ray.energy, current_ray.color))

            if hit.absorbed:
                return

            # Определяем поведение (по умолчанию в линзах приоритет у преломления)
            # Вы можете адаптировать под ваши флаги (например, prioritize_refraction)
            allow_reflection = hit.allow_reflection
            allow_refraction = hit.allow_refraction

            # Расщепляем единый WhiteRay на спектральные лучи с физическим расчетом направления
            for disp_ray in current_ray.spectral_rays:
                if allow_refraction:
                    resolved_n_inside = get_dispersion_n(hit.n_inside, disp_ray)
                    n_next = resolved_n_inside if abs(disp_ray.current_n - 1.0) < 1e-6 else 1.0

                    refracted_dir = refract(disp_ray.direction, hit.normal, disp_ray.current_n, n_next)

                    if refracted_dir is not None:
                        disp_ray.origin[:] = hit.point + offset_distance * refracted_dir
                        disp_ray.direction[:] = refracted_dir
                        disp_ray.current_n = n_next
                    else:
                        # Полное внутреннее отражение для этой компоненты
                        normal = hit.normal
                        if np.dot(normal, disp_ray.direction) > 0:
                            normal = -normal
                        ref_dir = disp_ray.direction - 2 * np.dot(disp_ray.direction, normal) * normal
                        ref_dir /= np.linalg.norm(ref_dir)

                        disp_ray.origin[:] = hit.point + offset_distance * ref_dir
                        disp_ray.direction[:] = ref_dir

                elif allow_reflection:
                    normal = hit.normal
                    if np.dot(normal, disp_ray.direction) > 0:
                        normal = -normal
                    reflected_dir = disp_ray.direction - 2 * np.dot(disp_ray.direction, normal) * normal
                    reflected_dir /= np.linalg.norm(reflected_dir)

                    disp_ray.origin[:] = hit.point + offset_distance * reflected_dir
                    disp_ray.direction[:] = reflected_dir
                else:
                    # Если объект прозрачный для этого диапазона
                    disp_ray.origin[:] = hit.point + offset_distance * disp_ray.direction

                # Добавляем маленький шаг смещения в геометрию лучей, чтобы визуализация не рвалась
                segments.append(Segment(hit.point.copy(), disp_ray.origin.copy(), disp_ray.energy, disp_ray.color))

                # Запускаем рекурсию дальше, уменьшая глубину!
                recurse(disp_ray, d - 1, from_pool=False)
            return

        hit = find_best_hit(current_ray, elements)
        if hit is None:
            end = current_ray.origin + current_ray.direction * RAY_INFINITY_DISTANCE
            segments.append(Segment(current_ray.origin.copy(), end.copy(),
                                    current_ray.energy, current_ray.color))
            return

        segments.append(Segment(current_ray.origin.copy(), hit.point.copy(),
                                current_ray.energy, current_ray.color))

        if hit.absorbed:
            return

        if hit.is_thin_lens:
            new_dir = hit.obj.thin_lens_deflection(current_ray.direction, hit.point)
            start = hit.point + offset_distance * new_dir
            if pool:
                new_ray = pool.acquire(start, new_dir,
                                       energy=current_ray.energy,
                                       current_n=current_ray.current_n,
                                       color=current_ray.color,
                                       wavelength=current_ray.wavelength,
                                       energy_color_type=current_ray.energy_color_type)
            else:
                new_ray = Ray(start, new_dir,
                              energy=current_ray.energy,
                              current_n=current_ray.current_n,
                              color=current_ray.color,
                              wavelength=current_ray.wavelength,
                              energy_color_type=current_ray.energy_color_type)
            segments.append(Segment(hit.point.copy(), start.copy(), current_ray.energy, current_ray.color))
            recurse(new_ray, d - 1, from_pool=pool is not None)
            if pool:  # новый луч больше не нужен
                pool.release(new_ray)
            return

        # Прозрачный проход или отражение/преломление
        resolved_n_inside = get_dispersion_n(hit.n_inside, current_ray)
        n_next = resolved_n_inside if abs(current_ray.current_n - 1.0) < 1e-6 else 1.0

        if not hit.allow_reflection and not hit.allow_refraction:
            start = hit.point + offset_distance * current_ray.direction
            if pool:
                new_ray = pool.acquire(start, current_ray.direction,
                                       energy=current_ray.energy,
                                       current_n=current_ray.current_n,
                                       color=current_ray.color,
                                       wavelength=current_ray.wavelength,
                                       energy_color_type=current_ray.energy_color_type)
            else:
                new_ray = Ray(start, current_ray.direction,
                              energy=current_ray.energy,
                              current_n=current_ray.current_n,
                              color=current_ray.color,
                              wavelength=current_ray.wavelength,
                              energy_color_type=current_ray.energy_color_type)
            segments.append(Segment(hit.point.copy(), start.copy(), current_ray.energy, current_ray.color))
            recurse(new_ray, d - 1, from_pool=pool is not None)
            if pool:
                pool.release(new_ray)
            return

        # Вызов split_ray с пулом
        new_rays = split_ray(current_ray, hit.normal, n_next, hit.point,
                             allow_reflection=hit.allow_reflection,
                             allow_refraction=hit.allow_refraction,
                             offset_distance=offset_distance,
                             use_polarization_color=use_polarization_color,
                             pool=pool)
        for nr in new_rays:
            segments.append(Segment(hit.point.copy(), nr.origin.copy(), nr.energy, nr.color))
            recurse(nr, d - 1, from_pool=pool is not None)
            if pool:
                pool.release(nr)  # после обработки каждого порождённого луча

    recurse(ray, depth, from_pool=False)
    return segments

def trace_ray(ray: Ray, elements: List, mode: str = 'tree',
              max_depth: int = 10, min_energy: float = 0.01,
              offset_distance: float = 0.5, use_polarization_color=False):
    if mode == 'simple':
        return _trace_simple(ray, elements, max_depth, offset_distance,
                             prioritize_refraction=True)
    elif mode == 'tree':
        segments = []
        _trace_recursive(ray, elements, max_depth, min_energy, segments,
                         total_limit=5000, offset_distance=offset_distance,
                         use_polarization_color=use_polarization_color)
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
