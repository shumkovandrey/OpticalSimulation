"""Излучатели пучков лучей."""

from typing import List

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation

from core.rays import DispersiveRay, Ray, WhiteRay


class BeamEmitter:
    """Излучатель пучка лучей: трансформация, генерация параллельных лучей.

    Полностью эквивалентен BeamEmitter из main.py (шаг 5).
    """

    def __init__(
        self,
        origin,
        direction=np.array([1.0, 0.0, 0.0]),
        rotation_degrees=(0, 0, 0),
        pool=None,
        num_rays=5,
        min_offset=-2.0,
        max_offset=2.0,
        color="yellow",
        wavelength=550,
        energy_color_type=2,
        energy=1.0,
        current_n=1.0,
        ray_class=Ray,
        num_spectral_bands=7,
        spawn_as_spectral_pack=True,
    ):
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
        """Повернуть лучи"""
        rot = Rotation.from_euler("xyz", angles_deg, degrees=True).as_matrix()
        self.direction = rot @ self.base_direction
        self.rotation_matrix = rot

    def translate(self, vec):
        """Переместить лучи"""
        self.origin += np.asarray(vec)

    def emit(self) -> List[Ray]:
        """Создать лучей"""
        rays = []

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
                        ray = self.pool.acquire(
                            origin=world_origin,
                            direction=self.direction,
                            energy=sub_energy,
                            current_n=self.current_n,
                            color=self.color,
                            wavelength=wl,
                            energy_color_type=self.energy_color_type,
                            ray_class=DispersiveRay,
                        )
                    else:
                        ray = DispersiveRay(
                            origin=world_origin,
                            direction=self.direction,
                            energy=sub_energy,
                            current_n=self.current_n,
                            energy_color_type=self.energy_color_type,
                            wavelength=wl,
                        )
                    rays.append(ray)
            else:
                # Стандартное создание одиночного луча
                if self.pool:
                    ray = self.pool.acquire(
                        origin=world_origin,
                        direction=self.direction,
                        energy=self.energy,
                        current_n=self.current_n,
                        color=self.color,
                        wavelength=self.wavelength,
                        energy_color_type=self.energy_color_type,
                        ray_class=self.ray_class,
                        num_spectral_bands=self.num_spectral_bands,
                    )
                else:
                    if self.ray_class is WhiteRay:
                        ray = WhiteRay(
                            origin=world_origin, direction=self.direction,
                            num_spectral_bands=self.num_spectral_bands,
                            energy=self.energy, current_n=self.current_n,
                            energy_color_type=self.energy_color_type,
                        )
                    else:
                        kwargs = dict(
                            origin=world_origin, direction=self.direction,
                            energy=self.energy, current_n=self.current_n,
                            color=self.color, energy_color_type=self.energy_color_type,
                        )
                        if self.ray_class is DispersiveRay:
                            kwargs["wavelength"] = self.wavelength
                        ray = self.ray_class(**kwargs)
                rays.append(ray)
        return rays

    def get_mesh(self) -> pv.PolyData:
        """Возвращается стрелка для визуализации излучателя."""
        arrow = pv.Arrow(start=(0.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0), scale=0.5)
        return arrow
