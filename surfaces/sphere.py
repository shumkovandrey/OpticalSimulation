from typing import Optional

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as R

from core.rays import Ray
from surfaces.base import BaseSurface


class SphereSurface(BaseSurface):
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
        # Вектор от центра базовой сферы до начала луча
        oc = ray.origin - self.center

        # Вычисление коэффициентов квадратного уравнения для пересечения луча со сферой
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius ** 2

        # Дискриминант квадратного уравнения
        disc = b ** 2 - 4 * a * c
        if disc < 0:
            # Дискриминант отрицательный: луч проходит мимо сферы
            return None

        # Нахождение двух возможных точек пересечения
        t1 = (-b - np.sqrt(disc)) / (2.0 * a)
        t2 = (-b + np.sqrt(disc)) / (2.0 * a)

        valid_ts = []
        # Фильтрация и проверка найденных точек
        for t in (t1, t2):
            if t <= 1e-6:
                # Исключаем пересечения позади луча или на его старте
                continue

            # Вычисляем 3D-координаты потенциальной точки пересечения
            hit_p = ray.origin + ray.direction * t

            # Проверка по радиусу апертуры
            # Находим вектор от центра объекта до точки пересечения
            vec_to_hit = hit_p - self.lens_origin

            # Проекция этого вектора на оптическую ось линзы (смещение по глубине)
            projection = np.dot(vec_to_hit, self.lens_axis)

            # Вычисляем кратчайшее расстояние от точки пересечения до оси направления объекта
            dist_to_axis = np.linalg.norm(vec_to_hit - projection * self.lens_axis)

            r_max = float(self.edge_radius)

            # Проверяем, входит ли точка в радиальные границы линзы
            if dist_to_axis <= r_max + 1e-5:
                # Проверка на принадлежность точки правильной полусфере
                # Т.к. сферическая поверхность это сегмент, а не полная сфера, точка должна лежать на нужной части, а не на её противоположной половине.
                # Вычисляем максимальную толщину сегмента линзы
                sagitta = abs(self.radius) - np.sqrt(max(0.0, abs(self.radius) ** 2 - r_max ** 2))

                # Проверяем, что смещение точки пересечения вдоль оси не превышает толщину нужной части
                if abs(projection) <= (sagitta + 1e-4):
                    valid_ts.append(t)  # Точка прошла обе проверки, добавляем в список валидных

        # Возвращаем ближайшее корректное пересечение, если такое есть
        return min(valid_ts) if valid_ts else None

    def get_normal(self, point: np.ndarray) -> np.ndarray:
        normal = (point - self.center) / self.radius
        return normal / np.linalg.norm(normal)

    def get_mesh(self) -> pv.PolyData:
        abs_radius = abs(self.radius)

        # 1. Создаем сферу. ВНИМАНИЕ: Для PyVista центр в (0,0,0) - это центр кривизны!
        mesh = pv.Sphere(radius=abs_radius, center=(0, 0, 0),
                         phi_resolution=80, theta_resolution=80)

        # 2. Вычисляем стрелку прогиба (высоту сегмента чаши)
        sagitta = abs_radius - np.sqrt(max(0.0, abs_radius ** 2 - self.edge_radius ** 2))

        # 3. Обрезаем нужную чашу в локальных координатах PyVista
        if self.radius > 0:
            # Выпуклая: вершина сферы находится на её правом краю в точке (+R, 0, 0)
            mesh = mesh.clip(normal=[1, 0, 0], origin=[abs_radius - sagitta, 0, 0], invert=False)

            # СДВИГ К ВЕРШИНЕ: Чтобы локальный ноль меша (0,0,0) стал ВЕРШИНОЙ чаши,
            # мы должны сместить всю геометрию влево на величину радиуса R!
            mesh.translate([-abs_radius, 0, 0], inplace=True)
        else:
            # Вогнутая: вершина сферы находится на её левом краю в точке (-R, 0, 0)
            mesh = mesh.clip(normal=[-1, 0, 0], origin=[-abs_radius + sagitta, 0, 0], invert=False)

            # СДВИГ К ВЕРШИНЕ: Смещаем геометрию вправо на величину радиуса R,
            # чтобы вершина вогнутой чаши оказалась точно в локальном нуле (0,0,0)
            mesh.translate([abs_radius, 0, 0], inplace=True)

        # Теперь локальный ноль (0,0,0) меша — это ВЕРШИНА поверхности.
        # Когда Траме применит user_matrix(param_pos), вершина меша встанет точно в координаты ползунков,
        # идеально совпав с физическим self.lens_origin на сервере!
        return mesh

    def apply_transform(self, mat):
        R_mat = mat[:3, :3]
        t_mat = mat[:3, 3]
        self.lens_origin = R_mat @ self.lens_origin + t_mat
        self.lens_axis = R_mat @ self.lens_axis
        self.center = self.lens_origin - self.radius * self.lens_axis
