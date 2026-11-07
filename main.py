import numpy as np
import pyvista as pv
from abc import ABC, abstractmethod
from typing import List, Optional
from scipy.spatial.transform import Rotation as R

from fast_math import (
    cylinder_intersect,
    refract,
    fresnel_amplitudes,
    aspheric_intersect_numba,
)

from constants import (
    RAY_INFINITY_DISTANCE,
    DEFAULT_OFFSET_DISTANCE,
    DEFAULT_MIN_ENERGY,
)

from utils.color import (
    to_rgb as _to_rgb_utils,
    energy_to_alpha,
)

from core.rays import Ray, DispersiveRay, WhiteRay, RayPool
from core.segments import Segment, HitInfo
from core.dispersion import get_dispersion_n

from surfaces.plane import PlaneSurface
from surfaces.sphere import SphereSurface
from surfaces.cylinder import CylinderSurface
from surfaces.aspheric import AsphericSurface
from surfaces.mesh import MeshSurface
from surfaces.thin_lens import ThinLens
from surfaces.universal_lens import UniversalLens
from surfaces.hyperbolic_lens import HyperbolicLens


pv.global_theme.allow_empty_mesh = True


def split_ray(ray: Ray, normal: np.ndarray, n_next: float, start_point: np.ndarray,
              allow_reflection: bool = True, allow_refraction: bool = True,
              offset_distance: float = DEFAULT_OFFSET_DISTANCE,
              pool: Optional[RayPool] = None) -> List[Ray]:
    EPS = offset_distance

    cos_i = np.dot(normal, ray.direction)
    if cos_i > 0:
        normal = -normal
        cos_i = np.dot(normal, ray.direction)
    cos_i = -cos_i

    n1, n2 = ray.current_n, n_next

    # Получаем комплексные амплитуды Френеля из fast_math.py
    r_s, r_p, t_s, t_p = fresnel_amplitudes(n1, n2, cos_i)

    # Вычисляем честный коэффициент отражения по энергии R
    # (квадрат модуля комплексной амплитуды отражения)
    R_s = abs(r_s) ** 2
    R_p = abs(r_p) ** 2
    R_coeff = 0.5 * (R_s + R_p)

    # Проверка на полное внутреннее отражение (ПВО) или явный запрет преломления
    eta = n1 / n2
    sin2_t = (eta ** 2) * (1.0 - cos_i ** 2)
    is_tir = (sin2_t >= 1.0)

    if is_tir:
        R_coeff = 1.0
        T_coeff = 0.0
        cos_t = 0.0
        allow_refraction = False
    else:
        cos_t = np.sqrt(1.0 - sin2_t)
        # Строгий закон сохранения энергии волнового фронта:
        T_coeff = 1.0 - R_coeff

    # Корректировка флагами объектов сцены
    if not allow_reflection and not allow_refraction:
        return []
    if allow_reflection and not allow_refraction:
        R_coeff, T_coeff = 1.0, 0.0
    elif not allow_reflection and allow_refraction:
        R_coeff, T_coeff = 0.0, 1.0

    # Направления дочерних лучей
    reflected_dir = ray.direction - 2 * np.dot(ray.direction, normal) * normal
    reflected_dir /= np.linalg.norm(reflected_dir)

    if not is_tir:
        refracted_dir = eta * ray.direction + (eta * cos_i - cos_t) * normal
        refracted_dir /= np.linalg.norm(refracted_dir)
    else:
        refracted_dir = reflected_dir.copy()

    # Локальный базис для поляризации
    s_dir = np.cross(normal, ray.direction)
    if np.linalg.norm(s_dir) < 1e-10:
        s_dir = np.array([0.0, 1.0, 0.0])
    s_dir /= np.linalg.norm(s_dir)
    p_dir = np.cross(ray.direction, s_dir)
    p_dir /= np.linalg.norm(p_dir)

    new_rays = []

    # 1. ОТРАЖЁННЫЙ ЛУЧ
    if allow_reflection and R_coeff > 1e-6:
        reflected_energy = ray.energy * R_coeff
        if reflected_energy > 1e-6:
            if pool:
                new_ray = pool.acquire(
                    origin=start_point + EPS * reflected_dir, direction=reflected_dir,
                    energy=reflected_energy, current_n=n1, color=ray.color,
                    energy_color_type=ray.energy_color_type, wavelength=ray.wavelength,
                )
            else:
                new_ray = type(ray)(
                    origin=start_point + EPS * reflected_dir, direction=reflected_dir,
                    energy=reflected_energy, current_n=n1, color=ray.color,
                    wavelength=ray.wavelength, energy_color_type=ray.energy_color_type,
                )
            new_rays.append(new_ray)

    # 2. ПРЕЛОМЛЕННЫЙ ЛУЧ
    if allow_refraction and T_coeff > 1e-6:
        refracted_energy = ray.energy * T_coeff
        if refracted_energy > 1e-6:
            if pool:
                new_ray = pool.acquire(
                    origin=start_point + EPS * refracted_dir, direction=refracted_dir,
                    energy=refracted_energy, current_n=n2, color=ray.color,
                    energy_color_type=ray.energy_color_type, wavelength=ray.wavelength,
                    ray_class=type(ray)
                )
            else:
                new_ray = type(ray)(
                    origin=start_point + EPS * refracted_dir, direction=refracted_dir,
                    energy=refracted_energy, current_n=n2, color=ray.color,
                    wavelength=ray.wavelength, energy_color_type=ray.energy_color_type,
                )
            new_rays.append(new_ray)

    return new_rays


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
                 min_energy: float = 0,
                 offset_distance: float = 0.001,
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
        total_limit : int
            Аварийное ограничение на общее количество отрезков во всех
            рекурсивных ветвях (защита от бесконечного роста).
        """
        super().__init__(energy_color_type, pool)

        self.max_depth = max_depth
        self.min_energy = min_energy
        self.offset_distance = offset_distance
        self.total_limit = total_limit

    def trace(self, ray: 'Ray', elements: List) -> List[Segment]:
        return _trace_recursive(
            ray,
            elements,
            depth=self.max_depth,
            min_energy=self.min_energy,
            offset_distance=self.offset_distance,
            total_limit=self.total_limit,
            pool=self.pool
        )


class RayCloud:
    """
    Единый актор для отрисовки множества отрезков с индивидуальным цветом
    и прозрачностью, зависящей от энергии.
    """

    def __init__(self, plotter: pv.Plotter,
                 default_color="yellow",
                 line_width: float = 2.0,
                 min_alpha: float = 0,
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

    def _build_rgba(self, color, alpha: float) -> np.ndarray:
        rgb = _to_rgb_utils(color)
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

            alpha = energy_to_alpha(
                seg.energy, energy_color_type,
                self.min_alpha, self.gamma
            )
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
        Прогоняет все зарегистрированные лучи через текущий режим трассировки.
        """
        # Сначала собираем лучи от эмиттеров
        for emitter in self.emitters:
            for ray in emitter.emit():
                self.rays.append(ray)

        # Распаковываем контейнеры белого света на отдельные DispersiveRay до симуляции
        unpacked_rays = []
        for ray in self.rays:
            if isinstance(ray, WhiteRay):
                # Энергия честно поделится внутри generate_rays по спектру!
                spectral = ray.generate_rays(pool=self.pool)
                unpacked_rays.extend(spectral)
            else:
                unpacked_rays.append(ray)

        # Перезаписываем список лучей симуляции
        self.rays = unpacked_rays

        all_segments = []
        for ray in self.rays:
            segments = self._mode.trace(ray, self.elements)
            all_segments.extend(segments)

        return all_segments

    def render(self):
        segments = self.trace_all()
        self.cloud.update(segments, energy_color_type=self._mode.energy_color_type)

        # Возвращаем все лучи (включая распакованные спектральные) обратно в пул
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

        # ИСПРАВЛЕНИЕ: Вместо слепого np.cross и условий с 0.9,
        # мы берем честный локальный вектор Y и трансформируем его матрицей поворота.
        # Это полностью убирает скачки на 26 градусах и заставляет работать поворот по X!
        local_perp1 = np.array([0.0, 1.0, 0.0])
        perp1 = self.rotation_matrix @ local_perp1
        perp1 /= np.linalg.norm(perp1)

        offsets = np.linspace(self.min_offset, self.max_offset, self.num_rays)

        for dy in offsets:
            # Теперь world_origin смещается строго перпендикулярно направлению выстрела
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
                # Стандартное создание одиночного луча
                if self.pool:
                    ray = self.pool.acquire(origin=world_origin,
                                            direction=self.direction,
                                            energy=self.energy,
                                            current_n=self.current_n,
                                            color=self.color,
                                            wavelength=self.wavelength,
                                            energy_color_type=self.energy_color_type,
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


def _trace_simple(ray: 'Ray',
                  elements: List,
                  max_bounces: int = 10,
                  offset_distance: float = DEFAULT_OFFSET_DISTANCE,
                  prioritize_refraction: bool = True,
                  pool: Optional['RayPool'] = None) -> List[Segment]:
    """
    Однолучевая последовательная трассировка.
    При встрече с WhiteRay автоматически разветвляет симуляцию для спектра.
    """
    segments: List[Segment] = []

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
                     min_energy: float = DEFAULT_MIN_ENERGY,
                     offset_distance: float = DEFAULT_OFFSET_DISTANCE,
                     total_limit: int = 5000,
                     pool: Optional[RayPool] = None) -> List[Segment]:
    """
    Рекурсивная трассировка дерева лучей без геометрических наложений отрезков.
    """
    segments = []

    def recurse(current_ray: 'Ray', d: int, from_pool: bool):
        nonlocal segments

        if len(segments) >= total_limit or d <= 0 or current_ray.energy < min_energy:
            return

        # 1. ОБРАБОТКА КОНТЕЙНЕРА WHITE_RAY ВНУТРИ ДЕРЕВА
        if isinstance(current_ray, WhiteRay):
            spectral_components = current_ray.generate_rays(pool=pool)
            for comp in spectral_components:
                recurse(comp, d, from_pool=(pool is not None))
                if pool:
                    pool.release(comp)
            return

        # Находим точку пересечения
        hit = find_best_hit(current_ray, elements)

        # Определяем цвет текущего луча для отрисовки его входящего отрезка
        ray_color = current_ray.color
        if isinstance(current_ray, DispersiveRay) and current_ray.wavelength is not None:
            ray_color = current_ray.wavelength_to_rgb(current_ray.wavelength)

        # 2. ОТРИСОВКА ВХОДЯЩЕГО ОТРЕЗКА (СТРОГО ОДИН РАЗ НА ШАГ РЕКУРСИИ)
        if hit is None:
            # Луч улетает в бесконечность — рисуем финальный отрезок с текущей энергией
            end = current_ray.origin + current_ray.direction * RAY_INFINITY_DISTANCE
            segments.append(Segment(current_ray.origin.copy(), end.copy(), current_ray.energy, ray_color))
            return
        else:
            # Луч долетел до объекта — рисуем отрезок пути ДО точки удара с текущей энергией
            segments.append(Segment(current_ray.origin.copy(), hit.point.copy(), current_ray.energy, ray_color))

        # Если объект поглотил луч, на этом шаг рекурсии для него завершен
        if hit.absorbed:
            return

        # 3. ОБРАБОТКА ТОНКОЙ ЛИНЗЫ
        if hit.is_thin_lens:
            new_dir = hit.obj.thin_lens_deflection(current_ray.direction, hit.point)
            start = hit.point + offset_distance * new_dir

            if pool:
                new_ray = pool.acquire(start, new_dir, energy=current_ray.energy,
                                       current_n=current_ray.current_n, color=current_ray.color,
                                       wavelength=current_ray.wavelength,
                                       energy_color_type=current_ray.energy_color_type)
            else:
                new_ray = Ray(start, new_dir, energy=current_ray.energy,
                              current_n=current_ray.current_n, color=current_ray.color,
                              wavelength=current_ray.wavelength, energy_color_type=current_ray.energy_color_type)

            recurse(new_ray, d - 1, from_pool=(pool is not None))
            if pool:
                pool.release(new_ray)
            return

        # 4. ВЗАИМОДЕЙСТВИЕ С ОПТИЧЕСКОЙ ПОВЕРХНОСТЬЮ (ФРЕНЕЛЬ / ПРОХОД)
        resolved_n_inside = get_dispersion_n(hit.n_inside, current_ray)
        n_next = resolved_n_inside if abs(current_ray.current_n - 1.0) < 1e-6 else 1.0

        if not hit.allow_reflection and not hit.allow_refraction:
            # Прозрачный проход (без изменения направления и энергии)
            start = hit.point + offset_distance * current_ray.direction
            if pool:
                new_ray = pool.acquire(start, current_ray.direction, energy=current_ray.energy,
                                       current_n=current_ray.current_n, color=current_ray.color,
                                       wavelength=current_ray.wavelength,
                                       energy_color_type=current_ray.energy_color_type)
            else:
                new_ray = Ray(start, current_ray.direction, energy=current_ray.energy,
                              current_n=current_ray.current_n, color=current_ray.color,
                              wavelength=current_ray.wavelength, energy_color_type=current_ray.energy_color_type)

            recurse(new_ray, d - 1, from_pool=(pool is not None))
            if pool:
                pool.release(new_ray)
            return

        # Расщепляем луч на отраженный и преломленный
        new_rays = split_ray(current_ray, hit.normal, n_next, hit.point,
                             allow_reflection=hit.allow_reflection,
                             allow_refraction=hit.allow_refraction,
                             offset_distance=offset_distance,
                             pool=pool)

        # 5.ЗАПУСК ПОДЛУЧЕЙ В РЕКУРСИЮ
        # Теперь мы НЕ добавляем здесь отрезки (они добавятся на следующем уровне рекурсии в начале шага #2)
        for nr in new_rays:
            recurse(nr, d - 1, from_pool=(pool is not None))
            if pool:
                pool.release(nr)

    recurse(ray, depth, from_pool=False)
    return segments


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