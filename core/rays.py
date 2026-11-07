"""Классы лучей: Ray, DispersiveRay, WhiteRay и пул переиспользования RayPool."""

import numpy as np
from typing import List, Tuple

from utils.color import wavelength_to_rgb as _wavelength_to_rgb_utils


class Ray:
    """Базовый луч: начало, направление, энергия, показатель преломления среды."""

    def __init__(
        self,
        origin,
        direction,
        energy=1.0,
        current_n=1.0,
        color="yellow",
        energy_color_type=2,
        wavelength=None,
    ):
        self.origin = np.array(origin, dtype=float)

        self.direction = np.array(direction, dtype=float)
        if np.linalg.norm(self.direction) < 1e-12:
            self.direction = np.array([1.0, 0.0, 0.0])
        else:
            self.direction /= np.linalg.norm(self.direction)

        self.energy = energy
        self.current_n = current_n
        self.color = color
        self.energy_color_type = energy_color_type
        self.wavelength = wavelength


class DispersiveRay(Ray):
    """Луч с длиной волны (нм): цвет и показатель преломления зависят от спектра."""

    def __init__(self, origin, direction, wavelength=550.0, **kwargs):
        current_n = kwargs.pop("current_n", 1.0)

        super().__init__(
            origin=origin,
            direction=direction,
            wavelength=wavelength,
            current_n=current_n,
            **kwargs,
        )

        if wavelength is not None:
            self.color = self.wavelength_to_rgb(wavelength)

    @staticmethod
    def wavelength_to_rgb(wavelength_nm: float) -> Tuple[float, float, float]:
        """Делегат utils.color.wavelength_to_rgb (сохранён для обратной совместимости)."""
        return _wavelength_to_rgb_utils(wavelength_nm)


class WhiteRay:
    """Контейнер для пучка спектральных лучей; сам в трассировке не участвует."""

    def __init__(self, origin, direction, num_spectral_bands=7, energy=1.0,
                 current_n=1.0, energy_color_type=2, **kwargs):
        self.origin = np.array(origin, dtype=float)
        self.direction = np.array(direction, dtype=float)
        self.direction /= np.linalg.norm(self.direction)
        self.energy = energy
        self.current_n = current_n
        self.energy_color_type = energy_color_type
        self.num_spectral_bands = num_spectral_bands
        self.spectral_rays = []

    def generate_rays(self, pool=None) -> List[DispersiveRay]:
        """Разворачивает белый луч в набор спектральных составляющих."""
        rays = []
        wavelengths = np.linspace(400.0, 700.0, self.num_spectral_bands)
        sub_energy = self.energy / self.num_spectral_bands

        for wl in wavelengths:
            if pool:
                sub_ray = pool.acquire(
                    origin=self.origin.copy(),
                    direction=self.direction.copy(),
                    energy=sub_energy,
                    current_n=self.current_n,
                    energy_color_type=self.energy_color_type,
                    wavelength=wl,
                    ray_class=DispersiveRay,
                )
            else:
                sub_ray = DispersiveRay(
                    origin=self.origin.copy(),
                    direction=self.direction.copy(),
                    energy=sub_energy,
                    current_n=self.current_n,
                    energy_color_type=self.energy_color_type,
                    wavelength=wl,
                )
            rays.append(sub_ray)
        return rays


class RayPool:
    """Пул переиспользуемых лучей для снижения аллокаций."""

    def __init__(self, initial_size=100):
        self.pools = {}
        self.initial_size = initial_size
        self._ensure_pool_exists(Ray)

    def _ensure_pool_exists(self, ray_class):
        """Создаёт внутреннюю корзину для нового типа луча, если её нет."""
        if ray_class not in self.pools:
            self.pools[ray_class] = []
            if ray_class is not WhiteRay:
                for _ in range(self.initial_size):
                    self.pools[ray_class].append(
                        ray_class(np.zeros(3), np.array([1.0, 0.0, 0.0]))
                    )

    def acquire(self, origin, direction, energy=1.0, current_n=1.0,
                color="yellow", energy_color_type=2, wavelength=None,
                ray_class=Ray, num_spectral_bands=7):
        """Взять луч определённого класса из пула и инициализировать его поля."""
        self._ensure_pool_exists(ray_class)

        if self.pools[ray_class]:
            ray = self.pools[ray_class].pop()
        else:
            if ray_class is WhiteRay:
                ray = WhiteRay(np.zeros(3), np.array([1.0, 0.0, 0.0]),
                               num_spectral_bands=num_spectral_bands, pool=self)
            else:
                ray = ray_class(np.zeros(3), np.array([1.0, 0.0, 0.0]))

        ray.origin[:] = origin
        ray.direction[:] = direction
        ray.direction /= np.linalg.norm(ray.direction)
        ray.energy = energy
        ray.current_n = current_n
        ray.energy_color_type = energy_color_type
        ray.wavelength = wavelength

        if ray_class is DispersiveRay and wavelength is not None:
            ray.color = ray.wavelength_to_rgb(wavelength)
        elif ray_class is WhiteRay:
            ray.color = (1.0, 1.0, 1.0)
            ray.num_spectral_bands = num_spectral_bands
        else:
            ray.color = color

        return ray

    def release(self, ray: Ray):
        """Возвращает луч в пул по его типу."""
        if ray is None:
            return
        ray_class = type(ray)
        if ray_class is WhiteRay:
            return
        if ray_class not in self.pools:
            self.pools[ray_class] = []
        self.pools[ray_class].append(ray)
