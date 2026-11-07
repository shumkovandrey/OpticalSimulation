"""Режимы трассировки: интерфейс TraceMode и реализации Simple/Tree."""

from abc import ABC, abstractmethod
from typing import List, Optional

from core.segments import Segment
from core.rays import Ray, RayPool


class TraceMode(ABC):
    """Общий интерфейс для алгоритмов трассировки."""
    def __init__(self, energy_color_type: int = 1, pool: Optional['RayPool'] = None):
        self.energy_color_type = energy_color_type
        self.pool = pool

    @abstractmethod
    def trace(self, ray: 'Ray', elements: List) -> List[Segment]:
        ...


class SimpleMode(TraceMode):
    """Последовательная трассировка без ветвления (см. main.py:SimpleMode)."""
    def __init__(self,
                 max_bounces: int = 30,
                 offset_distance: float = 0.001,
                 prioritize_refraction: bool = True,
                 energy_color_type: int = 1,
                 pool=None):
        super().__init__(energy_color_type, pool)
        self.max_bounces = max_bounces
        self.offset_distance = offset_distance
        self.prioritize_refraction = prioritize_refraction

    def trace(self, ray: 'Ray', elements: List) -> List[Segment]:
        # Импорт внутри метода, чтобы избежать цикла modes ← simple ← modes
        from tracing.simple import _trace_simple
        return _trace_simple(
            ray,
            elements,
            max_bounces=self.max_bounces,
            offset_distance=self.offset_distance,
            prioritize_refraction=self.prioritize_refraction,
            pool=self.pool,
        )


class TreeMode(TraceMode):
    """Рекурсивная трассировка с ветвлением по Френелю."""
    def __init__(self,
                 max_depth: int = 20,
                 min_energy: float = 0,
                 offset_distance: float = 0.001,
                 total_limit: int = 5000,
                 energy_color_type: int = 1,
                 pool=None):
        super().__init__(energy_color_type, pool)
        self.max_depth = max_depth
        self.min_energy = min_energy
        self.offset_distance = offset_distance
        self.total_limit = total_limit

    def trace(self, ray: 'Ray', elements: List) -> List[Segment]:
        from tracing.recursive import _trace_recursive
        return _trace_recursive(
            ray,
            elements,
            depth=self.max_depth,
            min_energy=self.min_energy,
            offset_distance=self.offset_distance,
            total_limit=self.total_limit,
            pool=self.pool,
        )
