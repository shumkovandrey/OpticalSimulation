import numpy as np

import pyvista as pv

import random

import scipy.spatial.transform as st # Для удобного вращения
from scipy.spatial.transform import Rotation as R


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


class Ray:
    def __init__(self, origin, direction):
        self.origin = np.array(origin, dtype=float)
        self.direction = np.array(direction, dtype=float)
        self.direction /= np.linalg.norm(self.direction)


class OpticalElement:
    def intersect(self, ray):
        """Возвращает дистанцию t до точки пересечения или None"""
        pass

    def get_normal(self, point):
        """Возвращает нормаль в точке пересечения"""
        pass

    def interact(self, ray_dir, normal, n1, n2):
        """Логика взаимодействия (преломление/отражение)"""
        pass


class PlaneSurface:
    def __init__(self, point, normal, n_inside, lens_origin, lens_axis, edge_radius):
        self.point = np.array(point, dtype=float)
        self.normal = np.array(normal, dtype=float)
        self.normal /= np.linalg.norm(self.normal)
        self.n = n_inside
        self.lens_origin = np.array(lens_origin, dtype=float)
        self.lens_axis = np.array(lens_axis, dtype=float)
        self.lens_axis /= np.linalg.norm(self.lens_axis)
        self.edge_radius = edge_radius

    def intersect(self, ray):
        dot_dn = np.dot(ray.direction, self.normal)
        if abs(dot_dn) < 1e-6:
            return None

        t = np.dot(self.point - ray.origin, self.normal) / dot_dn
        if t > 1e-6:
            hit_p = ray.origin + ray.direction * t

            # Проверка: попадает ли точка в радиус линзы в 3D
            vec_to_hit = hit_p - self.lens_origin
            projection = np.dot(vec_to_hit, self.lens_axis)
            dist_to_axis = np.linalg.norm(vec_to_hit - projection * self.lens_axis)

            if dist_to_axis <= (self.edge_radius + 1e-6):
                return t
        return None

    def get_normal(self, point):
        return self.normal

    def interact(self, ray_dir, normal, n1, n2):
        # Используем ту же логику преломления, что и в SphereSurface
        eta = n1 / n2
        cos_i = np.dot(normal, ray_dir)
        actual_normal = normal
        if cos_i > 0:
            actual_normal = -normal
            cos_i = np.dot(actual_normal, ray_dir)

        cos_i = -cos_i
        sin2_t = eta ** 2 * (1.0 - cos_i ** 2)
        if sin2_t > 1.0:  # Полное внутреннее отражение
            return ray_dir - 2 * np.dot(ray_dir, actual_normal) * actual_normal

        cos_t = np.sqrt(1.0 - sin2_t)
        return eta * ray_dir + (eta * cos_i - cos_t) * actual_normal


# class PlaneSurface(OpticalElement):
#     def __init__(self, point, normal, n_inside=1.0, is_mirror=False):
#         self.point = np.array(point, dtype=float)
#         self.normal = np.array(normal, dtype=float)
#         self.normal /= np.linalg.norm(self.normal)
#         self.n = n_inside
#         self.is_mirror = is_mirror
#
#     def intersect(self, ray):
#         dot_dn = np.dot(ray.direction, self.normal)
#         if abs(dot_dn) < 1e-6:
#             return None
#         t = np.dot(self.point - ray.origin, self.normal) / dot_dn
#         return t if t > 1e-6 else None
#
#     def get_normal(self, point):
#         return self.normal
#
#     def interact(self, ray_dir, normal, n1, n2):
#         if self.is_mirror:
#             # Отражение: R = D - 2(D·N)N
#             return ray_dir - 2 * np.dot(ray_dir, normal) * normal
#         else:
#             # Преломление (Снеллиус)
#             eta = n1 / n2
#             cos_i = -np.dot(normal, ray_dir)
#             sin2_t = eta ** 2 * (1.0 - cos_i ** 2)
#             if sin2_t > 1.0:  # Полное внутреннее
#                 return ray_dir - 2 * np.dot(ray_dir, normal) * normal
#             cos_t = np.sqrt(1.0 - sin2_t)
#             return eta * ray_dir + (eta * cos_i - cos_t) * normal

pass

# class SphereSurface:
#     def __init__(self, center, radius, n_inside=1.5, x_min=-np.inf, x_max=np.inf, edge_radius=np.inf):
#         """
#         center: [x, y, z] центра сферы
#         radius: радиус сферы
#         n_inside: показатель преломления материала за поверхностью
#         x_min, x_max: границы по оси X, где поверхность физически существует
#         edge_radius: максимальное удаление от оптической оси (радиус самой линзы)
#         """
#         self.center = np.array(center, dtype=float)
#         self.radius = radius
#         self.n = n_inside
#         self.x_min = x_min
#         self.x_max = x_max
#         self.edge_radius = edge_radius
#
#     def intersect(self, ray):
#         oc = ray.origin - self.center
#         a = np.dot(ray.direction, ray.direction)
#         b = 2.0 * np.dot(oc, ray.direction)
#         c = np.dot(oc, oc) - self.radius**2
#
#         discriminant = b**2 - 4*a*c
#         if discriminant < 0: return None
#
#         t1 = (-b - np.sqrt(discriminant)) / (2.0 * a)
#         t2 = (-b + np.sqrt(discriminant)) / (2.0 * a)
#
#         valid_ts = []
#         for t in [t1, t2]:
#             if t > 1e-6:
#                 hit_p = ray.origin + ray.direction * t
#
#                 # 1. СТРОГИЙ РАДИУС (с крошечным допуском для стабильности)
#                 dist_sq = hit_p[1] ** 2 + hit_p[2] ** 2
#                 in_radius = dist_sq <= (self.edge_radius ** 2 + 1e-5)
#
#                 # 2. СТРОГИЙ X
#                 # Важно: hit_p[0] - это координата X точки пересечения
#                 in_x = (self.x_min - 1e-5) <= hit_p[0] <= (self.x_max + 1e-5)
#                 # in_x = 36 <= hit_p[0] <= 41.35
#
#                 if in_x and in_radius:
#                     valid_ts.append(t)
#
#         return min(valid_ts) if valid_ts else None
#
#     def get_normal(self, point):
#         """Нормаль всегда направлена ОТ центра сферы (наружу)"""
#         normal = (point - self.center) / self.radius
#         return normal / np.linalg.norm(normal)
#
#     def interact(self, ray_dir, normal, n_from, n_to):
#         """Закон Снеллиуса с защитой от неправильного направления нормали"""
#         eta = n_from / n_to
#
#         # Убеждаемся, что нормаль направлена навстречу лучу
#         cos_i = np.dot(normal, ray_dir)
#         actual_normal = normal
#         if cos_i > 0:
#             actual_normal = -normal
#             cos_i = np.dot(actual_normal, ray_dir)
#
#         cos_i = -cos_i
#         sin2_t = eta ** 2 * (1.0 - cos_i ** 2)
#
#         if sin2_t > 1.0:  # Полное внутреннее отражение
#             return ray_dir - 2 * np.dot(ray_dir, actual_normal) * actual_normal
#
#         cos_t = np.sqrt(1.0 - sin2_t)
#         return eta * ray_dir + (eta * cos_i - cos_t) * actual_normal


class SphereSurface:
    def __init__(self, center, radius, n_inside, lens_origin, lens_axis, edge_radius, thickness):
        self.center = np.array(center, dtype=float)
        self.radius = radius
        self.n = n_inside
        self.lens_origin = np.array(lens_origin, dtype=float)
        self.lens_axis = np.array(lens_axis, dtype=float)
        self.lens_axis /= np.linalg.norm(self.lens_axis)
        self.edge_radius = edge_radius
        self.thickness = thickness

    def intersect(self, ray):
        # Расчет пересечения
        oc = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius ** 2

        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return None

        t1 = (-b - np.sqrt(discriminant)) / (2.0 * a)
        t2 = (-b + np.sqrt(discriminant)) / (2.0 * a)

        valid_ts = []
        for t in [t1, t2]:
            if t > 1e-6:
                hit_p = ray.origin + ray.direction * t

                # Векторная проверка границ
                vec_to_hit = hit_p - self.lens_origin
                projection = np.dot(vec_to_hit, self.lens_axis)
                dist_to_axis = np.linalg.norm(vec_to_hit - projection * self.lens_axis)

                in_radius = dist_to_axis <= (self.edge_radius + 1e-6)
                in_thickness = abs(projection) <= (self.thickness / 2 + 5.0)

                if in_radius and in_thickness:
                    valid_ts.append(t)

        return min(valid_ts) if valid_ts else None

    def get_normal(self, point):
        normal = (point - self.center) / self.radius
        return normal / np.linalg.norm(normal)

    def interact(self, ray_dir, normal, n1, n2):
        # Закон Снеллиуса
        eta = n1 / n2
        cos_i = np.dot(normal, ray_dir)

        actual_normal = normal
        if cos_i > 0:
            actual_normal = -normal
            cos_i = np.dot(actual_normal, ray_dir)

        cos_i = -cos_i
        sin2_t = eta ** 2 * (1.0 - cos_i ** 2)

        if sin2_t > 1.0:  # Внутреннее отражение
            return ray_dir - 2 * np.dot(ray_dir, actual_normal) * actual_normal

        cos_t = np.sqrt(1.0 - sin2_t)
        return eta * ray_dir + (eta * cos_i - cos_t) * actual_normal


class Screen(PlaneSurface):
    def __init__(self, point, normal, size=20):
        # Передаем все необходимые аргументы в базовый класс PlaneSurface
        # n_inside=1.0 (воздух), чтобы луч не преломлялся при касании экрана
        super().__init__(
            point=point,
            normal=normal,
            n_inside=1.0,
            lens_origin=point,  # Центр экрана
            lens_axis=normal,   # Направление экрана
            edge_radius=size/2  # Ограничение размера
        )
        self.size = size

    def interact(self, ray_dir, normal, n1, n2):
        # Возвращаем None, чтобы run_simulation понял: луч поглощен
        return None


class BoxPrism:
    def __init__(self, origin, size_x, size_y, size_z, n=1.5):
        self.origin = np.array(origin, dtype=float)
        self.size = np.array([size_x, size_y, size_z], dtype=float)
        self.n = n
        self.surfaces = []

        # Конфигурация 6 граней: [нормаль, смещение_от_центра, размеры_грани]
        # Нормали направлены НАРУЖУ от центра
        face_configs = [
            ([1, 0, 0], size_x / 2, [size_y, size_z]),  # Передняя (+X)
            ([-1, 0, 0], size_x / 2, [size_y, size_z]),  # Задняя (-X)
            ([0, 1, 0], size_y / 2, [size_x, size_z]),  # Правая (+Y)
            ([0, -1, 0], size_y / 2, [size_x, size_z]),  # Левая (-Y)
            ([0, 0, 1], size_z / 2, [size_x, size_y]),  # Верхняя (+Z)
            ([0, 0, -1], size_z / 2, [size_x, size_y])  # Нижняя (-Z)
        ]

        for norm, offset, edge_dims in face_configs:
            # 1. Рассчитываем центр конкретной грани
            face_center = self.origin + np.array(norm) * offset

            # 2. Рассчитываем радиус ограничения (диагональ грани)
            r_limit = np.sqrt(edge_dims[0] ** 2 + edge_dims[1] ** 2) / 2

            # 3. Создаем грань, передавая ВСЕ необходимые аргументы
            surf = PlaneSurface(
                point=face_center,
                normal=norm,
                n_inside=n,
                lens_origin=self.origin,  # Центр всего блока
                lens_axis=norm,  # Для грани это её нормаль
                edge_radius=r_limit
            )

            self.surfaces.append(surf)

    def get_surfaces(self):
        return self.surfaces

    def get_mesh(self):
        # Визуальная модель в PyVista
        return pv.Cube(center=self.origin,
                       x_length=self.size[0],
                       y_length=self.size[1],
                       z_length=self.size[2])

    def rotate(self, angles_deg):
        # angles_deg = [X, Y, Z] - углы поворота
        rot = R.from_euler('xyz', angles_deg, degrees=True).as_matrix()

        for surf in self.surfaces:
            # Поворачиваем точку относительно центра
            local_pos = surf.point - self.origin
            surf.point = self.origin + rot @ local_pos
            # Поворачиваем нормаль грани
            surf.normal = rot @ surf.normal


class UniversalLens:
    # def __init__(self, center_x, R1, R2, thickness, edge_radius, n=1.5, show_axis=True):
    #     self.center_x = center_x
    #     self.R1 = R1
    #     self.R2 = R2
    #     self.thickness = thickness
    #     self.edge_radius = edge_radius
    #     self.n = n
    #     self.show_axis = show_axis
    #
    #     # 1. Координаты вершин на оптической оси
    #     v1_x = self.center_x - self.thickness / 2
    #     v2_x = self.center_x + self.thickness / 2
    #
    #     # 2. Функция для расчета X-координаты края сферы (на высоте edge_radius)
    #     def get_edge_x(vertex_x, radius):
    #         if radius is None or abs(radius) > 1e5:
    #             return vertex_x
    #         # Сагитта (глубина прогиба): s = R - sqrt(R^2 - h^2)
    #         sagitta = abs(radius) - np.sqrt(max(0, radius ** 2 - self.edge_radius ** 2))
    #         # Если R > 0 (выпуклая), край правее вершины. Если R < 0 (вогнутая), край левее.
    #         # Но для ВТОРОЙ поверхности (R2) знаки работают зеркально
    #         return vertex_x + sagitta if radius > 0 else vertex_x - sagitta
    #
    #     # Рассчитываем X краев для обеих поверхностей
    #     # Для R2 инвертируем знак, так как она "смотрит" в другую сторону
    #     edge1_x = get_edge_x(v1_x, self.R1)
    #     # Для задней поверхности: если R2 > 0 (выпуклая), край уходит ВЛЕВО от вершины
    #     edge2_x = get_edge_x(v2_x, -self.R2 if self.R2 else None)
    #
    #     # 3. Находим ГЛОБАЛЬНЫЕ границы всей линзы
    #     # Берем min и max из всех четырех точек (2 вершины + 2 края)
    #     self.min_x = min(v1_x, v2_x, edge1_x, edge2_x) - 0.1
    #     self.max_x = max(v1_x, v2_x, edge1_x, edge2_x) + 0.1
    #
    #     # --- Создание ПЕРЕДНЕЙ поверхности ---
    #     if R1 is None or abs(R1) > 1e5:  # Плоская
    #         self.front = PlaneSurface(point=[v1_x, 0, 0], normal=[-1, 0, 0],
    #                                   n_inside=n)
    #         s1_sagitta = 0
    #     else:
    #         # Для SphereSurface: R > 0 (выпуклая), R < 0 (вогнутая)
    #         # Центр сферы c1 = вершина + радиус (с учетом знака)
    #         c1 = v1_x + R1
    #         # Рассчитываем "вылет" (сагитту) края линзы для установки границ X
    #         s1_sagitta = R1 - np.sign(R1) * np.sqrt(max(0, R1 ** 2 - edge_radius ** 2))
    #
    #         # Границы по X для этой сферы (с запасом, чтобы покрыть всю дугу)
    #         x_min_f = min(v1_x, v1_x + s1_sagitta) - 0.1
    #         x_max_f = max(v1_x, v1_x + s1_sagitta) + 0.1
    #
    #         self.front = SphereSurface(center=[c1, 0, 0], radius=abs(R1), n_inside=n,
    #                                    x_min=self.min_x, x_max=self.max_x, edge_radius=edge_radius)
    #
    #     # --- Создание ВТОРОЙ поверхности ---
    #     if R2 is None or abs(R2) > 1e5:  # Плоская
    #         self.back = PlaneSurface(point=[v2_x, 0, 0], normal=[1, 0, 0],
    #                                  n_inside=n)
    #         s2_sagitta = 0
    #     else:
    #         # Центр сферы c2 = вершина - радиус (чтобы знаки R работали как в оптике)
    #         c2 = v2_x - R2
    #         s2_sagitta = R2 - np.sign(R2) * np.sqrt(max(0, R2 ** 2 - edge_radius ** 2))
    #
    #         x_min_b = min(v2_x, v2_x - s2_sagitta) - 0.1
    #         x_max_b = max(v2_x, v2_x - s2_sagitta) + 0.1
    #
    #         self.back = SphereSurface(center=[c2, 0, 0], radius=abs(R2), n_inside=n,
    #                                   x_min=self.min_x, x_max=self.max_x, edge_radius=edge_radius)
    #
    #     # 2. РАСЧЕТ ОПТИЧЕСКИХ ПАРАМЕТРОВ (Формула толстой линзы)
    #     # Для формулы: r1 и r2 (радиус второй берется с инверсией знака)
    #     r1_val = R1 if R1 else 1e10
    #     r2_val = -R2 if R2 else -1e10
    #
    #     inv_f = (n - 1) * (1 / r1_val - 1 / r2_val + ((n - 1) * thickness) / (n * r1_val * r2_val))
    #
    #     if abs(inv_f) < 1e-10:
    #         self.f_dist = float('inf')
    #     else:
    #         self.f_dist = 1 / inv_f
    #
    #     # Расчет положения главной плоскости H2 для корректного draw_axis
    #     self.h2 = -(self.f_dist * (n - 1) * thickness) / (n * r1_val)

    def __init__(self, origin, axis_dir, R1, R2, thickness, edge_radius, n=1.5):
        self.R1 = R1
        self.R2 = R2
        self.thickness = thickness
        self.edge_radius = edge_radius
        self.n = n
        self.show_axis = True

        self.origin = np.array(origin, dtype=float)
        self.axis_dir = np.array(axis_dir, dtype=float)
        self.axis_dir /= np.linalg.norm(self.axis_dir)

        # Создаем матрицу вращения, которая переводит [1,0,0] в axis_dir
        # Это позволит нам легко вращать меш и лучи
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

        # Пересчитываем центры сфер в локальных координатах (относительно 0)
        v1_local = -thickness / 2
        v2_local = thickness / 2

        # 2. Создание ПЕРЕДНЕЙ поверхности
        if R1 is None:
            # Если R1 None — создаем плоскость в мировых координатах
            # Точка плоскости = центр линзы + поворот * (смещение по X)
            p1_world = self.origin + self.rotation @ np.array([v1_local, 0, 0])
            n1_world = self.rotation @ np.array([-1, 0, 0])  # Нормаль наружу (влево)
            self.front = PlaneSurface(
                point=p1_world, normal=n1_world, n_inside=n,
                lens_origin=self.origin, lens_axis=self.axis_dir, edge_radius=edge_radius
            )
        else:
            # Если R1 есть — создаем сферу
            c1_world = self.origin + self.rotation @ np.array([v1_local + R1, 0, 0])
            self.front = SphereSurface(
                center=c1_world, radius=abs(R1), n_inside=n,
                lens_origin=self.origin, lens_axis=self.axis_dir,
                edge_radius=edge_radius, thickness=thickness
            )

        # 3. Создание ЗАДНЕЙ поверхности
        if R2 is None:
            p2_world = self.origin + self.rotation @ np.array([v2_local, 0, 0])
            n2_world = self.rotation @ np.array([1, 0, 0])  # Нормаль наружу (вправо)
            self.back = PlaneSurface(
                point=p2_world, normal=n2_world, n_inside=n,
                lens_origin=self.origin, lens_axis=self.axis_dir, edge_radius=edge_radius
            )
        else:
            c2_world = self.origin + self.rotation @ np.array([v2_local - R2, 0, 0])
            self.back = SphereSurface(
                center=c2_world, radius=abs(R2), n_inside=n,
                lens_origin=self.origin, lens_axis=self.axis_dir,
                edge_radius=edge_radius, thickness=thickness
            )

    def get_surfaces(self):
        return [self.front, self.back]

    def _generate_local_mesh(self):
        # 1. Сетка параметров
        rs = np.linspace(0, self.edge_radius, 30)
        phis = np.linspace(0, 2 * np.pi, 60)
        r_grid, phi_grid = np.meshgrid(rs, phis)

        # Локальные Y и Z (остаются такими же)
        y = r_grid * np.cos(phi_grid)
        z = r_grid * np.sin(phi_grid)

        # Локальные вершины на оси X (относительно нуля)
        v1_local = -self.thickness / 2
        v2_local = self.thickness / 2

        # 2. Вычисляем локальный X для каждой поверхности
        def get_local_x(R, v_x, r_vals, is_front):
            if R is not None and abs(R) < 1e5:
                # Локальный центр сферы на оси X
                # Для front: c = v1 + R. Для back: c = v2 - R.
                c_x = v_x + R if is_front else v_x - R

                dx = np.sqrt(np.maximum(0, abs(R) ** 2 - r_vals ** 2))

                # Определяем, какую сторону сферы брать (выпуклость/вогнутость)
                # Если R > 0 (выпуклая для front), берем c_x - dx
                if is_front:
                    return c_x - dx if R > 0 else c_x + dx
                else:
                    return c_x + dx if R > 0 else c_x - dx
            else:
                # Для плоской поверхности локальный X — это просто координата вершины
                return np.full_like(r_vals, v_x)

        x_front = get_local_x(self.R1, v1_local, r_grid, True)
        x_back = get_local_x(self.R2, v2_local, r_grid, False)

        # 3. Создаем меши (в локальном пространстве)
        front_mesh = pv.StructuredGrid(x_front, y, z).extract_surface()
        back_mesh = pv.StructuredGrid(x_back, y, z).extract_surface()

        # 4. Создаем ободок
        rim_x = np.array([x_front[:, -1], x_back[:, -1]])
        rim_y = np.array([y[:, -1], y[:, -1]])
        rim_z = np.array([z[:, -1], z[:, -1]])
        rim_mesh = pv.StructuredGrid(rim_x, rim_y, rim_z).extract_surface()

        return front_mesh.merge(back_mesh).merge(rim_mesh)

    def get_mesh(self):
        # Получаем меш в локальных координатах
        local_mesh = self._generate_local_mesh()

        # Создаем 4x4 матрицу трансформации
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation  # Матрица вращения из __init__
        matrix[:3, 3] = self.origin  # Вектор смещения (центр линзы)

        # Переносим локальный меш в мировые координаты
        return local_mesh.transform(matrix)

    # def get_mesh(self):
    #     # 1. Создаем сетку параметров: радиус (r) и угол (phi)
    #     # r идет от 0 до edge_radius, phi от 0 до 2*pi
    #     rs = np.linspace(0, self.edge_radius, 30)
    #     phis = np.linspace(0, 2 * np.pi, 60)
    #     r_grid, phi_grid = np.meshgrid(rs, phis)
    #
    #     # Координаты Y и Z для всех точек поверхности
    #     y = r_grid * np.cos(phi_grid)
    #     z = r_grid * np.sin(phi_grid)
    #
    #     # 2. Вычисляем X для каждой поверхности
    #     def get_surface_x(surface, r_vals):
    #         if isinstance(surface, SphereSurface):
    #             # Формула сферы: (x - cx)^2 + r^2 = R^2  => x = cx ± sqrt(R^2 - r^2)
    #             # Выбираем знак так, чтобы точка была на нужной стороне сферы
    #             dx = np.sqrt(np.maximum(0, surface.radius ** 2 - r_vals ** 2))
    #             # Если R > 0 (выпуклая), для front берем cx - dx, для back берем cx + dx
    #             if surface.center[0] > self.center_x:  # центр справа
    #                 return surface.center[0] - dx
    #             else:  # центр слева
    #                 return surface.center[0] + dx
    #         else:
    #             # Для плоскости X постоянен
    #             return np.full_like(r_vals, surface.point[0])
    #
    #     x_front = get_surface_x(self.front, r_grid)
    #     x_back = get_surface_x(self.back, r_grid)
    #
    #     # 3. Создаем меши поверхностей
    #     # Передняя "крышка"
    #     front_mesh = pv.StructuredGrid(x_front, y, z).extract_surface()
    #     # Задняя "крышка"
    #     back_mesh = pv.StructuredGrid(x_back, y, z).extract_surface()
    #
    #     # 4. Создаем "боковину" (ободок)
    #     # Соединяем края передней и задней поверхностей
    #     rim_x = np.array([x_front[:, -1], x_back[:, -1]])
    #     rim_y = np.array([y[:, -1], y[:, -1]])
    #     rim_z = np.array([z[:, -1], z[:, -1]])
    #     rim_mesh = pv.StructuredGrid(rim_x, rim_y, rim_z).extract_surface()
    #
    #     # Объединяем всё в один объект
    #     return front_mesh.merge(back_mesh).merge(rim_mesh)

    def draw_axis(self, plotter, length=100):
        if not self.show_axis:
            return

        # 1. Основная линия оси
        axis_start = [self.center_x - length / 2, 0, 0]
        axis_stop = [self.center_x + length / 2, 0, 0]
        axis_line = pv.Line(axis_start, axis_stop)
        plotter.add_mesh(axis_line, color="white", line_width=1, opacity=0.5)

        # 2. Если линза плоская (f_dist = inf), выходим
        if np.isinf(self.f_dist) or abs(self.f_dist) > 1000:
            return

        # 3. Расчет BFL (расстояние от задней вершины до фокуса)
        # d - толщина, n - показатель преломления, r1 - радиус первой поверхности
        d = self.thickness
        n = self.n
        f = self.f_dist
        # Используем r1_val для формулы (если плоскость, то очень большое число)
        r1_val = self.R1 if self.R1 else 1e10

        # Формула заднего фокусного отрезка
        bfl = f * (1 - ((n - 1) * d) / (n * r1_val))

        # Координата задней вершины (правый край линзы)
        back_vertex_x = self.center_x + d / 2

        # Реальная координата фокуса на сцене
        f_mark_x = back_vertex_x + bfl

        # 4. Отрисовка меток (F и -F)
        # Для симметрии рассчитаем и передний фокус (FFL)
        r2_val = -self.R2 if self.R2 else -1e10
        ffl = f * (1 + ((n - 1) * d) / (n * r2_val))
        front_vertex_x = self.center_x - d / 2
        f_front_x = front_vertex_x - ffl

        marks = [(f_mark_x, "F"), (f_front_x, "-F")]

        for x_pos, txt in marks:
            # Вертикальная засечка
            mark = pv.Line([x_pos, -0.4, 0], [x_pos, 0.4, 0])
            plotter.add_mesh(mark, color="red", line_width=3)

            # Подпись
            plotter.add_point_labels([x_pos, 0.6, 0], [txt],
                                     font_size=12, text_color="yellow",
                                     shape=None, show_points=False)


def run_simulation(start_ray, elements, max_bounces=10):
    path = [start_ray.origin]
    current_ray = start_ray
    current_n = 1.0  # Воздух снаружи

    # print(f"--- Запуск луча из {start_ray.origin} ---")

    for i in range(max_bounces):
        best_t = float('inf')
        hit_obj = None

        for obj in elements:
            t = obj.intersect(current_ray)
            if t and t < best_t:
                best_t = t
                hit_obj = obj

        if hit_obj:
            hit_point = current_ray.origin + current_ray.direction * best_t
            path.append(hit_point)

            # Если это экран, он вернет None в interact
            normal = hit_obj.get_normal(hit_point)
            next_n = hit_obj.n if abs(current_n - 1.0) < 1e-6 else 1.0

            new_dir = hit_obj.interact(current_ray.direction, normal, current_n, next_n)

            if new_dir is None:  # ЛУЧ ОСТАНОВЛЕН ЭКРАНОМ
                break

            current_ray = Ray(hit_point, new_dir)
            current_n = next_n
            last_hit_obj = hit_obj
        else:
            # print(f"  Удар {i + 1}: Промах (луч улетел)")
            path.append(current_ray.origin + current_ray.direction * 50)
            break

    return np.array(path)


def visualize_scene(plotter, trajectory_list, elements, lenses=None):
    """
    plotter: объект pv.Plotter()
    trajectory_list: список массивов траекторий (для пучка лучей)
    elements: список всех поверхностей (для зеркал и пр.)
    lenses: список объектов класса Lens (чтобы рисовать их красиво)
    """

    # 1. Отрисовка всех лучей из пучка
    for traj in trajectory_list:
        path = pv.PolyData(traj)
        lines = np.hstack(([len(traj)], range(len(traj))))
        path.lines = lines
        plotter.add_mesh(path, color="yellow", line_width=2, render_lines_as_tubes=True)
        # Отрисовываем маленькие сферы в каждой точке изменения пути
        points_mesh = pv.PolyData(traj)
        plotter.add_mesh(points_mesh, color="purple", point_size=10, render_points_as_spheres=True)

    # 2. Отрисовка линз как целых объектов
    drawn_surfaces = set()
    if lenses:
        for lens in lenses:
            plotter.add_mesh(lens.get_mesh(), color="cyan", opacity=0.75, smooth_shading=True)
            # Запоминаем, какие поверхности мы уже отрисовали в составе линз
            drawn_surfaces.add(lens.front)
            drawn_surfaces.add(lens.back)

    # 3. Отрисовка остальных элементов (зеркал и одиночных поверхностей)
    for obj in elements:
        if obj in drawn_surfaces:
            continue  # Пропускаем сферы, которые уже нарисованы как линзы

        if isinstance(obj, PlaneSurface):
            plane = pv.Plane(center=obj.point, direction=obj.normal, i_size=5, j_size=5)
            color = "white" if obj.is_mirror else "lightblue"
            plotter.add_mesh(plane, color=color, opacity=0.5)

        elif isinstance(obj, SphereSurface):
            # Если это одиночная сфера (не линза)
            sphere = pv.Sphere(radius=obj.radius, center=obj.center)
            plotter.add_mesh(sphere, color="grey", opacity=0.2)

    plotter.add_axes()
    plotter.show()


# # --- НАСТРОЙКА СЦЕНЫ ---
# plotter = pv.Plotter()
# plotter.set_background("black")
#
# # 1. Создаем объект
# prism_block = BoxPrism(origin=[0, 0, 0], size_x=10, size_y=20, size_z=15, n=1.5)
#
# # 2. Задаем углы поворота [X, Y, Z]
# angles = [15, 0, 30] # Наклон на 15 градусов по X и поворот на 30 по Z
# prism_block.rotate(angles)
#
# # 3. Синхронизируем визуальную модель (Mesh)
# prism_mesh = prism_block.get_mesh()
# # В PyVista порядок поворотов должен совпадать
# prism_mesh.rotate_x(angles[0], inplace=True)
# prism_mesh.rotate_y(angles[1], inplace=True)
# prism_mesh.rotate_z(angles[2], inplace=True)
#
# plotter.add_mesh(prism_mesh, color="cyan", opacity=0.3)
#
# # 2. Создаем экран (Screen)
# # Ставим его справа (X=40), развернув к лучам
# screen_pos = np.array([40.0, -10.0, 0.0])
# prism_screen = Screen(point=screen_pos, normal=[-1, 0, 0], size=40)
#
# # Визуализация экрана
# screen_mesh = pv.Plane(center=prism_screen.point, direction=prism_screen.normal,
#                        i_size=40, j_size=40)
# plotter.add_mesh(screen_mesh, color="white", opacity=1)
#
# # 3. Настройка источника и спектра лучей
# colors = ["red", "orange", "yellow", "green", "cyan", "blue", "violet"]
# n_values = np.linspace(1.50, 1.62, len(colors))  # Разные n для каждого цвета
#
# source_pos = np.array([-30.0, 8.0, 0.0])
# target_point = np.array([-5.0, 0.0, 0.0])
# direction = (target_point - source_pos) / np.linalg.norm(target_point - source_pos)
#
# # 4. Цикл симуляции
# for i in range(len(colors)):
#     ray = Ray(origin=source_pos, direction=direction)
#
#     # СБОРКА СИСТЕМЫ: Призма + Экран
#     # Получаем актуальные (повернутые) грани призмы
#     prism_surfaces = prism_block.get_surfaces()
#
#     # Обновляем показатель преломления для текущего цвета
#     for surf in prism_surfaces:
#         surf.n = n_values[i]
#
#     # Объединяем в одну систему для run_simulation
#     full_system = [*prism_surfaces, prism_screen]
#
#     # Запускаем трассировку (max_bounces=5 хватит для прохода через призму до экрана)
#     trajectory = run_simulation(ray, full_system, max_bounces=6)
#
#     # ОТРИСОВКА ЛУЧА
#     path = pv.PolyData(trajectory)
#     lines = np.hstack(([len(trajectory)], range(len(trajectory))))
#     path.lines = lines
#     plotter.add_mesh(path, color=colors[i], opacity=0.9, line_width=3, label=f"n={n_values[i]:.2f}")
#
#     # Точки попадания (маркеры)
#     if len(trajectory) > 2:
#         # Рисуем все точки кроме начала и конца
#         hits = pv.PolyData(trajectory[1:])
#         # plotter.add_mesh(hits, color="red", point_size=10, render_points_as_spheres=True)


# 1. Инициализация сцены
plotter = pv.Plotter()
plotter.set_background("black")

# --- ЛИНЗА 1: Собирающая (Лежит на оси X) ---
l1 = UniversalLens(
    origin=[-20, 0, 0],
    axis_dir=[1, 0, 0],
    R1=5, R2=5, thickness=0.75, edge_radius=1.75, n=1.5
)

# --- ЛИНЗА 2: Рассеивающая (Стоит вертикально, ось вдоль Z) ---
l2 = UniversalLens(
    origin=[0, 0, 0],
    axis_dir=[0, 0, 1],
    R1=-15, R2=-15, thickness=2, edge_radius=8, n=1.5
)

# --- ЛИНЗА 3: Мениск (Наклонен под 45 градусов в плоскости XY) ---
l3 = UniversalLens(
    origin=[20, 10, 0],
    axis_dir=[1, 1, 0],
    R1=10, R2=20, thickness=3, edge_radius=6, n=1.5
)

# --- ЛИНЗА 4: Плоско-выпуклая (Наклонена в пространстве) ---
l4 = UniversalLens(
    origin=[0, 20, 10],
    axis_dir=[1, 0, 0], # Смотрит по диагонали Y-Z
    R1=None,              # Первая поверхность - ПЛОСКАЯ
    R2=None,               # Вторая - выпуклая
    thickness=4,
    edge_radius=8,
    n=1.5
)

all_lenses = [l1]

# 2. Симуляция лучей для каждой линзы
for i in range(len(all_lenses)):
    lens = all_lenses[i]
    # Отрисовка меша линзы
    plotter.add_mesh(lens.get_mesh(), color="cyan", opacity=0.3, smooth_shading=True)

    # Генерируем пучок лучей, направленный вдоль оси текущей линзы
    # Мы берем локальный старт и поворачиваем его в мировые координаты
    y_offsets = np.linspace(-10, 10, 40)
    for off in y_offsets:
        # Локальный вектор: старт за 15 единиц до линзы, смещение по "высоте"
        local_start = np.array([-15, random.uniform(-1, 1), random.uniform(-1, 1)])
        local_dir = np.array([0, 0, 0])

        # Перевод в мировые координаты
        world_start = lens.origin + lens.rotation @ local_start
        world_dir = local_dir + lens.axis_dir

        ray = Ray(origin=world_start, direction=world_dir)
        # В систему передаем поверхности только этой линзы для чистоты теста
        trajectory = run_simulation(ray, lens.get_surfaces(), max_bounces=5)

        # Отрисовка луча
        path = pv.PolyData(trajectory)
        path.lines = np.hstack(([len(trajectory)], range(len(trajectory))))
        plotter.add_mesh(path, color=["red", "blue", "yellow", "orange"][i], opacity=0.1, line_width=2)

        # Точки столкновения
        if len(trajectory) > 2:
            hits = pv.PolyData(trajectory[1:-1])
            plotter.add_mesh(hits, color="red", point_size=10, render_points_as_spheres=True)



plotter.show_grid(color="white")
plotter.view_xy()
plotter.enable_parallel_projection()
plotter.enable_terrain_style(mouse_wheel_zooms=True)
plotter.show()
