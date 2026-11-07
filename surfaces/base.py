# surfaces/base.py
"""Базовые классы для оптических поверхностей и линз-сборок."""

import numpy as np


class BaseSurface:
    """Базовый класс для поверхностей с оптическими диапазонами.

    Предоставляет единую реализацию is_active, повторяющую поведение
    PlaneSurface/SphereSurface/CylinderSurface/AsphericSurface.
    Наследники обязаны установить в __init__ три атрибута:
    self.reflection_range, self.refraction_range, self.absorption_range.
    """

    @staticmethod
    def _in_range(rng, wavelength):
        return (
            rng is not None
            and rng[0] is not None
            and rng[1] is not None
            and rng[0] <= wavelength <= rng[1]
        )

    def is_active(self, wavelength) -> bool:
        """True, если поверхность взаимодействует с данной длиной волны."""
        if wavelength is None:
            return True
        if (self.reflection_range is None
                and self.refraction_range is None
                and self.absorption_range is None):
            return False
        return (
            self._in_range(self.reflection_range, wavelength)
            or self._in_range(self.refraction_range, wavelength)
            or self._in_range(self.absorption_range, wavelength)
        )


class LensAssembly(BaseSurface):
    """Базовая сборка линзы: front + back + cylinder.

    Ожидает, что наследник создаст в __init__ атрибуты
    self.front, self.back, self.cylinder и self._last_hit_surface = None.
    """

    def intersect(self, ray):
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
        return (self.front.is_active(wavelength)
                or self.back.is_active(wavelength)
                or self.cylinder.is_active(wavelength))

    def get_surfaces(self):
        return [self.front, self.back, self.cylinder]
