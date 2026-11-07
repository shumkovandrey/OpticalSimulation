# visualization.py
"""Визуализация: облако отрезков (RayCloud) и оркестратор трассировки (RayTracer)."""

from typing import List, Optional, TYPE_CHECKING

import numpy as np
import pyvista as pv

from core.rays import Ray, WhiteRay, RayPool
from core.segments import Segment
from tracing.modes import TraceMode, SimpleMode, TreeMode
from utils.color import to_rgb as _to_rgb_utils, energy_to_alpha

if TYPE_CHECKING:
    from emitters import BeamEmitter


class RayCloud:
    """Единый актор для отрисовки множества отрезков с индивидуальным цветом.

    Полностью эквивалентен RayCloud из main.py (шаг 5).
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
    """Управляет лучами, элементами сцены и запускает трассировку.

    Полностью эквивалентен RayTracer из main.py (шаг 5).
    """

    def __init__(self,
                 plotter: pv.Plotter,
                 mode='simple',  # строка или экземпляр TraceMode
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
