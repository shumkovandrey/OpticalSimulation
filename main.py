import numpy as np

import pyvista as pv


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


class PlaneSurface(OpticalElement):
    def __init__(self, point, normal, n_inside=1.0, is_mirror=False):
        self.point = np.array(point, dtype=float)
        self.normal = np.array(normal, dtype=float)
        self.normal /= np.linalg.norm(self.normal)
        self.n = n_inside
        self.is_mirror = is_mirror

    def intersect(self, ray):
        dot_dn = np.dot(ray.direction, self.normal)
        if abs(dot_dn) < 1e-6:
            return None
        t = np.dot(self.point - ray.origin, self.normal) / dot_dn
        return t if t > 1e-6 else None

    def get_normal(self, point):
        return self.normal

    def interact(self, ray_dir, normal, n1, n2):
        if self.is_mirror:
            # Отражение: R = D - 2(D·N)N
            return ray_dir - 2 * np.dot(ray_dir, normal) * normal
        else:
            # Преломление (Снеллиус)
            eta = n1 / n2
            cos_i = -np.dot(normal, ray_dir)
            sin2_t = eta ** 2 * (1.0 - cos_i ** 2)
            if sin2_t > 1.0:  # Полное внутреннее
                return ray_dir - 2 * np.dot(ray_dir, normal) * normal
            cos_t = np.sqrt(1.0 - sin2_t)
            return eta * ray_dir + (eta * cos_i - cos_t) * normal


# class SphereSurface(OpticalElement):
#     def __init__(self, center, radius, n_inside=1.5, x_min=-np.inf, x_max=np.inf, edge_radius=np.inf):
#         super().__init__()
#         self.center = np.array(center, dtype=float)
#         self.radius = radius
#         self.n = n_inside
#         self.x_limit = (x_min, x_max)
#         self.edge_radius = edge_radius # Добавляем ограничение по высоте
#
#     def intersect(self, ray):
#         oc = ray.origin - self.center
#         a = np.dot(ray.direction, ray.direction)
#         b = 2.0 * np.dot(oc, ray.direction)
#         c = np.dot(oc, oc) - self.radius ** 2
#         discriminant = b ** 2 - 4 * a * c
#
#         if discriminant < 0: return None
#
#         t1 = (-b - np.sqrt(discriminant)) / (2.0 * a)
#         t2 = (-b + np.sqrt(discriminant)) / (2.0 * a)
#
#         valid_ts = []
#         for t in [t1, t2]:
#             if t > 1e-6:
#                 # Точка, где луч теоретически касается сферы
#                 hit_p = ray.origin + ray.direction * t
#
#                 # 1. Проверка: попадает ли точка в "тело" линзы по высоте?
#                 # Рассчитываем расстояние от точки до оптической оси (оси X)
#                 dist_from_axis = np.sqrt(hit_p[1] ** 2 + hit_p[2] ** 2)
#
#                 if dist_from_axis <= self.edge_radius:
#                     # 2. Проверка по X (уже должна быть у тебя)
#                     if self.x_min <= hit_p[0] <= self.x_max:
#                         valid_ts.append(t)
#
#         return min(valid_ts) if valid_ts else None
#
#         # for t in sorted([t1, t2]):
#         #     if t > 1e-6:
#         #         hit_point = ray.origin + ray.direction * t
#         #         # ПРОВЕРКА: попадает ли точка в "тело" линзы по оси X?
#         #         if self.x_limit[0] <= hit_point[0] <= self.x_limit[1]:
#         #             return t
#         # return None
#
#     def get_normal(self, point):
#         # Нормаль всегда от центра сферы наружу
#         return (point - self.center) / self.radius
#
#     def interact(self, ray_dir, normal, n_from, n_to):
#         # Коэффициент преломления
#         eta = n_from / n_to
#
#         # Убеждаемся, что нормаль направлена навстречу лучу
#         # Если мы выходим из линзы, нормаль и луч смотрят в одну сторону,
#         # поэтому для формулы Снеллиуса нормаль нужно инвертировать
#         cos_i = np.dot(normal, ray_dir)
#         actual_normal = normal
#         if cos_i > 0:
#             actual_normal = -normal
#             cos_i = np.dot(actual_normal, ray_dir)
#
#         cos_i = -cos_i  # Теперь cos_i положительный
#
#         sin2_t = eta ** 2 * (1.0 - cos_i ** 2)
#         if sin2_t > 1.0:  # Полное внутреннее отражение
#             return ray_dir - 2 * np.dot(ray_dir, actual_normal) * actual_normal
#
#         cos_t = np.sqrt(1.0 - sin2_t)
#         return eta * ray_dir + (eta * cos_i - cos_t) * actual_normal


class SphereSurface:
    def __init__(self, center, radius, n_inside=1.5, x_min=-np.inf, x_max=np.inf, edge_radius=np.inf):
        """
        center: [x, y, z] центра сферы
        radius: радиус сферы
        n_inside: показатель преломления материала за поверхностью
        x_min, x_max: границы по оси X, где поверхность физически существует
        edge_radius: максимальное удаление от оптической оси (радиус самой линзы)
        """
        self.center = np.array(center, dtype=float)
        self.radius = radius
        self.n = n_inside
        self.x_min = x_min
        self.x_max = x_max
        self.edge_radius = edge_radius

    def intersect(self, ray):
        oc = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius**2

        discriminant = b**2 - 4*a*c
        if discriminant < 0: return None

        t1 = (-b - np.sqrt(discriminant)) / (2.0 * a)
        t2 = (-b + np.sqrt(discriminant)) / (2.0 * a)

        valid_ts = []
        for t in [t1, t2]:
            if t > 1e-6:
                hit_p = ray.origin + ray.direction * t

                # 1. СТРОГИЙ РАДИУС (с крошечным допуском для стабильности)
                dist_sq = hit_p[1] ** 2 + hit_p[2] ** 2
                in_radius = dist_sq <= (self.edge_radius ** 2 + 1e-5)

                # 2. СТРОГИЙ X
                # Важно: hit_p[0] - это координата X точки пересечения
                in_x = (self.x_min - 1e-5) <= hit_p[0] <= (self.x_max + 1e-5)
                # in_x = 36 <= hit_p[0] <= 41.35

                if in_x and in_radius:
                    valid_ts.append(t)

        return min(valid_ts) if valid_ts else None

    def get_normal(self, point):
        """Нормаль всегда направлена ОТ центра сферы (наружу)"""
        normal = (point - self.center) / self.radius
        return normal / np.linalg.norm(normal)

    def interact(self, ray_dir, normal, n_from, n_to):
        """Закон Снеллиуса с защитой от неправильного направления нормали"""
        eta = n_from / n_to

        # Убеждаемся, что нормаль направлена навстречу лучу
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


class Screen(PlaneSurface):
    def __init__(self, point, normal, size=20):
        # По умолчанию n=1.0 (как у воздуха), чтобы не было преломления перед остановкой
        super().__init__(point, normal, n_inside=1.0)
        self.size = size

    def interact(self, ray_dir, normal, n1, n2):
        # Экран поглощает луч, возвращаем тот же вектор,
        # но в run_simulation мы добавим флаг остановки
        return None


class UniversalLens:
    def __init__(self, center_x, R1, R2, thickness, edge_radius, n=1.5, show_axis=True):
        self.center_x = center_x
        self.R1 = R1
        self.R2 = R2
        self.thickness = thickness
        self.edge_radius = edge_radius
        self.n = n
        self.show_axis = show_axis

        # 1. Координаты вершин на оптической оси
        v1_x = self.center_x - self.thickness / 2
        v2_x = self.center_x + self.thickness / 2

        # 2. Функция для расчета X-координаты края сферы (на высоте edge_radius)
        def get_edge_x(vertex_x, radius):
            if radius is None or abs(radius) > 1e5:
                return vertex_x
            # Сагитта (глубина прогиба): s = R - sqrt(R^2 - h^2)
            sagitta = abs(radius) - np.sqrt(max(0, radius ** 2 - self.edge_radius ** 2))
            # Если R > 0 (выпуклая), край правее вершины. Если R < 0 (вогнутая), край левее.
            # Но для ВТОРОЙ поверхности (R2) знаки работают зеркально
            return vertex_x + sagitta if radius > 0 else vertex_x - sagitta

        # Рассчитываем X краев для обеих поверхностей
        # Для R2 инвертируем знак, так как она "смотрит" в другую сторону
        edge1_x = get_edge_x(v1_x, self.R1)
        # Для задней поверхности: если R2 > 0 (выпуклая), край уходит ВЛЕВО от вершины
        edge2_x = get_edge_x(v2_x, -self.R2 if self.R2 else None)

        # 3. Находим ГЛОБАЛЬНЫЕ границы всей линзы
        # Берем min и max из всех четырех точек (2 вершины + 2 края)
        self.min_x = min(v1_x, v2_x, edge1_x, edge2_x) - 0.1
        self.max_x = max(v1_x, v2_x, edge1_x, edge2_x) + 0.1

        # --- Создание ПЕРЕДНЕЙ поверхности ---
        if R1 is None or abs(R1) > 1e5:  # Плоская
            self.front = PlaneSurface(point=[v1_x, 0, 0], normal=[-1, 0, 0],
                                      n_inside=n)
            s1_sagitta = 0
        else:
            # Для SphereSurface: R > 0 (выпуклая), R < 0 (вогнутая)
            # Центр сферы c1 = вершина + радиус (с учетом знака)
            c1 = v1_x + R1
            # Рассчитываем "вылет" (сагитту) края линзы для установки границ X
            s1_sagitta = R1 - np.sign(R1) * np.sqrt(max(0, R1 ** 2 - edge_radius ** 2))

            # Границы по X для этой сферы (с запасом, чтобы покрыть всю дугу)
            x_min_f = min(v1_x, v1_x + s1_sagitta) - 0.1
            x_max_f = max(v1_x, v1_x + s1_sagitta) + 0.1

            self.front = SphereSurface(center=[c1, 0, 0], radius=abs(R1), n_inside=n,
                                       x_min=self.min_x, x_max=self.max_x, edge_radius=edge_radius)

        # --- Создание ВТОРОЙ поверхности ---
        if R2 is None or abs(R2) > 1e5:  # Плоская
            self.back = PlaneSurface(point=[v2_x, 0, 0], normal=[1, 0, 0],
                                     n_inside=n)
            s2_sagitta = 0
        else:
            # Центр сферы c2 = вершина - радиус (чтобы знаки R работали как в оптике)
            c2 = v2_x - R2
            s2_sagitta = R2 - np.sign(R2) * np.sqrt(max(0, R2 ** 2 - edge_radius ** 2))

            x_min_b = min(v2_x, v2_x - s2_sagitta) - 0.1
            x_max_b = max(v2_x, v2_x - s2_sagitta) + 0.1

            self.back = SphereSurface(center=[c2, 0, 0], radius=abs(R2), n_inside=n,
                                      x_min=self.min_x, x_max=self.max_x, edge_radius=edge_radius)

        # 2. РАСЧЕТ ОПТИЧЕСКИХ ПАРАМЕТРОВ (Формула толстой линзы)
        # Для формулы: r1 и r2 (радиус второй берется с инверсией знака)
        r1_val = R1 if R1 else 1e10
        r2_val = -R2 if R2 else -1e10

        inv_f = (n - 1) * (1 / r1_val - 1 / r2_val + ((n - 1) * thickness) / (n * r1_val * r2_val))

        if abs(inv_f) < 1e-10:
            self.f_dist = float('inf')
        else:
            self.f_dist = 1 / inv_f

        # Расчет положения главной плоскости H2 для корректного draw_axis
        self.h2 = -(self.f_dist * (n - 1) * thickness) / (n * r1_val)

    def get_surfaces(self):
        return [self.front, self.back]

    # def get_mesh(self):
    #     # 1. Расчет точной высоты цилиндра (расстояние между краями поверхностей)
    #     def get_edge_x(surface, radius_val, is_front):
    #         if isinstance(surface, SphereSurface):
    #             # Расстояние от центра сферы до края по оси X
    #             # (Теорема Пифагора: x^2 + r^2 = R^2)
    #             sagitta_x = np.sqrt(max(0, surface.radius ** 2 - self.edge_radius ** 2))
    #             # Если поверхность выпуклая (центр "внутри"), край смещен от центра
    #             # Если вогнутая (центр "снаружи"), край тоже смещен.
    #             # Направление зависит от того, фронтальная это сторона или задняя.
    #             if is_front:
    #                 return surface.center[0] - sagitta_x if self.R1 > 0 else surface.center[0] + sagitta_x
    #             else:
    #                 return surface.center[0] + sagitta_x if self.R2 > 0 else surface.center[0] - sagitta_x
    #         return surface.point[0]
    #
    #     x_edge_front = get_edge_x(self.front, self.R1, True)
    #     x_edge_back = get_edge_x(self.back, self.R2, False)
    #
    #     # Реальная габаритная высота линзы на краях
    #     h_total = abs(x_edge_back - x_edge_front)
    #     # Центр цилиндра должен быть посередине между краями
    #     center_h = (x_edge_front + x_edge_back) / 2
    #
    #     # 2. Создаем цилиндр-заготовку точного размера
    #     cylinder = pv.Cylinder(center=(center_h, 0, 0), direction=(1, 0, 0),
    #                            radius=self.edge_radius, height=h_total + 0.01)
    #     lens_body = cylinder.extract_surface(algorithm='dataset_surface').triangulate()
    #
    #     # 3. Создаем меши сфер
    #     s1 = pv.Sphere(radius=self.front.radius, center=self.front.center,
    #                    phi_resolution=80, theta_resolution=80).triangulate()
    #     s2 = pv.Sphere(radius=self.back.radius, center=self.back.center,
    #                    phi_resolution=80, theta_resolution=80).triangulate()
    #
    #     # 4. Логика формирования формы
    #     if isinstance(self.front, SphereSurface):
    #         if self.R1 > 0:
    #             lens_body = lens_body.boolean_intersection(s1)
    #         else:
    #             lens_body = lens_body.boolean_difference(s1)
    #     else:
    #         lens_body = lens_body.clip(normal=[-1, 0, 0], origin=self.front.point, invert=False)
    #
    #     if isinstance(self.back, SphereSurface):
    #         if self.R2 > 0:
    #             lens_body = lens_body.boolean_intersection(s2)
    #         else:
    #             lens_body = lens_body.boolean_difference(s2)
    #     else:
    #         lens_body = lens_body.clip(normal=[1, 0, 0], origin=self.back.point, invert=False)
    #
    #     return lens_body
    # def get_mesh(self):
    #     # Проверяем, является ли каждая сторона выпуклой (R > 0)
    #     # Если R is None, это не выпуклая сторона
    #     r1_is_convex = self.R1 > 0 if self.R1 is not None else False
    #     r2_is_convex = self.R2 > 0 if self.R2 is not None else False
    #
    #     # Если обе стороны выпуклые - линза собирающая (h = thickness)
    #     # Во всех остальных случаях (вогнутая или плоско-выпуклая) даем запас
    #     if r1_is_convex and r2_is_convex:
    #         h = self.thickness
    #     else:
    #         h = self.thickness + 2.0  # Запаса в 2.0 обычно хватает, чтобы не плодить фантомы
    #     # h = self.thickness if self.R1 > 0 and self.R2 > 0 else self.thickness + 5.0
    #     cylinder = pv.Cylinder(center=(self.center_x, 0, 0), direction=(1, 0, 0),
    #                            radius=self.edge_radius, height=h)
    #     # Переводим в PolyData и делаем сетку замкнутой (Triangulate)
    #     lens_body = cylinder.extract_surface().triangulate()
    #
    #     # 2. Создаем меши сфер
    #     # Разрешение 60-80 достаточно для гладкости
    #
    #     # 3. Логика формирования формы
    #     # Обрезаем переднюю сторону
    #     if isinstance(self.front, SphereSurface):
    #         s1 = pv.Sphere(radius=self.front.radius, center=self.front.center,
    #                        phi_resolution=80, theta_resolution=80).triangulate()
    #         if self.R1 > 0: # Выпуклая: пересекаем с цилиндром
    #             lens_body = lens_body.boolean_intersection(s1)
    #         else: # Вогнутая: вычитаем сферу из цилиндра
    #             lens_body = lens_body.boolean_difference(s1)
    #     else:
    #         # Плоская: просто режем плоскостью
    #         # lens_body = lens_body.clip(normal=[-1, 0, 0], origin=self.front.point, invert=False)
    #         pass
    #
    #     # Обрезаем заднюю сторону
    #     if isinstance(self.back, SphereSurface):
    #         s2 = pv.Sphere(radius=self.back.radius, center=self.back.center,
    #                        phi_resolution=80, theta_resolution=80).triangulate()
    #         if self.R2 > 0: # Выпуклая
    #             lens_body = lens_body.boolean_intersection(s2)
    #         else: # Вогнутая
    #             lens_body = lens_body.boolean_difference(s2)
    #     else:
    #         # lens_body = lens_body.clip(normal=[1, 0, 0], origin=self.back.point, invert=False)
    #         pass
    #
    #     return lens_body

    def get_mesh(self):
        # 1. Создаем сетку параметров: радиус (r) и угол (phi)
        # r идет от 0 до edge_radius, phi от 0 до 2*pi
        rs = np.linspace(0, self.edge_radius, 30)
        phis = np.linspace(0, 2 * np.pi, 60)
        r_grid, phi_grid = np.meshgrid(rs, phis)

        # Координаты Y и Z для всех точек поверхности
        y = r_grid * np.cos(phi_grid)
        z = r_grid * np.sin(phi_grid)

        # 2. Вычисляем X для каждой поверхности
        def get_surface_x(surface, r_vals):
            if isinstance(surface, SphereSurface):
                # Формула сферы: (x - cx)^2 + r^2 = R^2  => x = cx ± sqrt(R^2 - r^2)
                # Выбираем знак так, чтобы точка была на нужной стороне сферы
                dx = np.sqrt(np.maximum(0, surface.radius ** 2 - r_vals ** 2))
                # Если R > 0 (выпуклая), для front берем cx - dx, для back берем cx + dx
                if surface.center[0] > self.center_x:  # центр справа
                    return surface.center[0] - dx
                else:  # центр слева
                    return surface.center[0] + dx
            else:
                # Для плоскости X постоянен
                return np.full_like(r_vals, surface.point[0])

        x_front = get_surface_x(self.front, r_grid)
        x_back = get_surface_x(self.back, r_grid)

        # 3. Создаем меши поверхностей
        # Передняя "крышка"
        front_mesh = pv.StructuredGrid(x_front, y, z).extract_surface()
        # Задняя "крышка"
        back_mesh = pv.StructuredGrid(x_back, y, z).extract_surface()

        # 4. Создаем "боковину" (ободок)
        # Соединяем края передней и задней поверхностей
        rim_x = np.array([x_front[:, -1], x_back[:, -1]])
        rim_y = np.array([y[:, -1], y[:, -1]])
        rim_z = np.array([z[:, -1], z[:, -1]])
        rim_mesh = pv.StructuredGrid(rim_x, rim_y, rim_z).extract_surface()

        # Объединяем всё в один объект
        return front_mesh.merge(back_mesh).merge(rim_mesh)

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
    plotter.set_background("black")

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


# 1. Настройка плоттера
plotter = pv.Plotter()
plotter.set_background("black")

# 2. Оптические элементы
# Линза в центре (X=0)
lens = UniversalLens(center_x=0, R1=15, R2=15, thickness=5.5, edge_radius=8, n=1.5)

# Зеркало 1: стоит после линзы, разворачивает луч на 90 градусов вбок (в плоскости XY)
mirror1 = PlaneSurface(point=[15, 0, 0], normal=[-1, 1, 0], is_mirror=True)

# Зеркало 2: принимает отраженный луч и уводит его ВВЕРХ (вдоль оси Z)
mirror2 = PlaneSurface(point=[15, 15, 0], normal=[0, -1, 1], is_mirror=True)

# Экран: ловит луч наверху
screen = Screen(point=[15, 15, 15], normal=[0, 0, -1], size=10)

system = [*lens.get_surfaces(), mirror1, mirror2, screen]

# 3. Источник света (небольшой пучок)
for y_off in np.linspace(-1, 1, 5):
    ray = Ray(origin=[-20, y_off, 0], direction=[1, 0, 0])
    trajectory = run_simulation(ray, system, max_bounces=10)

    # Визуализация луча
    path = pv.PolyData(trajectory)
    lines = np.hstack(([len(trajectory)], range(len(trajectory))))
    path.lines = lines
    plotter.add_mesh(path, color="yellow", line_width=2, render_lines_as_tubes=True)

    # Точки удара
    if len(trajectory) > 2:
        hits = pv.PolyData(trajectory[1:-1])
        plotter.add_mesh(hits, color="red", point_size=10)

# 4. Визуализация моделей
# Линза
plotter.add_mesh(lens.get_mesh(), color="cyan", opacity=0.3)
# Зеркало 1
m1_mesh = pv.Plane(center=mirror1.point, direction=mirror1.normal, i_size=10, j_size=10)
plotter.add_mesh(m1_mesh, color="silver", opacity=0.8)
# Зеркало 2
m2_mesh = pv.Plane(center=mirror2.point, direction=mirror2.normal, i_size=10, j_size=10)
plotter.add_mesh(m2_mesh, color="silver", opacity=0.8)
# Экран
s_mesh = pv.Plane(center=screen.point, direction=screen.normal, i_size=10, j_size=10)
plotter.add_mesh(s_mesh, color="white", opacity=0.5)

# Настройка камеры для 3D вида
plotter.camera_position = [(-50, -50, 50), (10, 10, 0), (0, 0, 1)]
plotter.add_axes()


# # 1. Настройка параметров
# F = 20.0
# n = 1.5
# thickness = 5.5
# edge_r = 10.0
#
# # Рассчитываем радиус для двояковыпуклой линзы под фокус F
# # Используем упрощенную формулу R = 2*F*(n-1) для оценки
# R_val = 2 * F * (n - 1)
#
# # 2. Создание линз
# # Первая линза в X=0
# lens1 = UniversalLens(center_x=0, R1=R_val, R2=R_val, thickness=thickness, edge_radius=edge_r, n=n)
#
# # Вторая линза на расстоянии 2F от первой (X = 40)
# lens2 = UniversalLens(center_x=2 * F, R1=R_val, R2=R_val, thickness=thickness, edge_radius=edge_r, n=n)
#
# system = [*lens1.get_surfaces(), *lens2.get_surfaces()]
#
# # 3. Настройка источника
# # Слева от первой линзы на 0.5F (X = -10) и на высоте (Y = 3)
# source_pos = [-0.5 * F, 3.0, 0]
#
# # 1. Создаем экран в X = 80 (например)
# screen = Screen(point=[80, 0, 0], normal=[-1, 0, 0], size=30)
#
# # 2. Добавляем его в систему К ПОСЛЕДНИМ
# system.append(screen)
#
# # Отрисовка экрана как белого прямоугольника
# screen_mesh = pv.Plane(center=screen.point, direction=screen.normal,
#                        i_size=screen.size, j_size=screen.size)
#
# # 4. Визуализация
# plotter = pv.Plotter()
# plotter.set_background("black")
#
# # Отрисовка линз и их осей
# for l in [lens1, lens2]:
#     plotter.add_mesh(l.get_mesh(), color="cyan", opacity=0.3)
#     l.draw_axis(plotter, length=100)
#
# # Запуск расходящегося пучка из точки источника
# num_rays = 11
# angles = np.linspace(-0.15, 0.15, num_rays)
#
# for angle in angles:
#     # Направление луча из точки под углом
#     direction = [np.cos(angle), np.sin(angle), 0]
#     ray = Ray(origin=source_pos, direction=direction)
#
#     # Трассировка через всю систему
#     trajectory = run_simulation(ray, system, max_bounces=10)
#
#     # Отрисовка луча
#     path = pv.PolyData(trajectory)
#     lines = np.hstack(([len(trajectory)], range(len(trajectory))))
#     path.lines = lines
#     plotter.add_mesh(path, color="yellow", line_width=1.5, opacity=0.8)
#
# # Метка источника
# plotter.add_mesh(pv.Sphere(radius=0.3, center=source_pos), color="white", render_points_as_spheres=True)
# plotter.add_point_labels([source_pos], ["Источник (0.5F)"], font_size=12)
# plotter.add_mesh(screen_mesh, color="white", opacity=0.5, label="Экран")
#
# plotter.view_xy()
# plotter.enable_parallel_projection()
# plotter.enable_terrain_style(mouse_wheel_zooms=True)
# plotter.show()


# # 1. Настройка окружения
# plotter = pv.Plotter(shape=(2, 2))  # Сетка 2x2 для разных видов
# plotter.set_background("black")
#
# # 2. Список тестовых линз
# test_cases = [
#     {
#         "title": "Плоско-выпуклая (R1=None, R2=15)",
#         "lens": UniversalLens(center_x=0, R1=None, R2=15, thickness=4, edge_radius=8, n=1.5),
#         "pos": (0, 0)
#     },
#     {
#         "title": "Двояковогнутая (R1=-15, R2=-15)",
#         "lens": UniversalLens(center_x=0, R1=-15, R2=-15, thickness=2, edge_radius=8, n=1.5),
#         "pos": (0, 1)
#     },
#     {
#         "title": "Положительный мениск (R1=10, R2=20)",
#         "lens": UniversalLens(center_x=0, R1=10, R2=20, thickness=3, edge_radius=8, n=1.5),
#         "pos": (1, 0)
#     },
#     {
#         "title": "Крутая собирающая (R1=10, R2=10)",
#         "lens": UniversalLens(center_x=0, R1=10, R2=10, thickness=6, edge_radius=8, n=1.5),
#         "pos": (1, 1)
#     }
# ]
#
# # 3. Отрисовка
# for case in test_cases:
#     plotter.subplot(*case["pos"])
#     lens = case["lens"]
#
#     # Визуализация меша (параметрического)
#     try:
#         mesh = lens.get_mesh()
#         plotter.add_mesh(mesh, color="cyan", opacity=0.5, smooth_shading=True, show_edges=True)
#     except Exception as e:
#         plotter.add_text(f"Ошибка отрисовки: {e}", font_size=10, color="red")
#
#     # Отрисовка лучей для проверки преломления
#     system = lens.get_surfaces()
#     for y in np.linspace(-10, 10, 30):
#         ray = Ray(origin=[-15, y, 0], direction=[1, 0, 0])
#         traj = run_simulation(ray, system, max_bounces=5)
#
#         path = pv.PolyData(traj)
#         path.lines = np.hstack(([len(traj)], range(len(traj))))
#         plotter.add_mesh(path, color="yellow", line_width=2)
#
#     plotter.add_text(case["title"], font_size=12)
#     plotter.add_axes()
#     plotter.view_xy()
#
# plotter.link_views()  # Синхронное вращение всех окон
# plotter.show()


# # 1. Настройка сцены
# plotter = pv.Plotter()
# plotter.set_background("black")
#
# # 2. Параметры микроскопа
# # Объектив (сильная линза, маленький фокус)
# obj_x = 0
# objective = UniversalLens(center_x=obj_x, R1=10, R2=10, thickness=4, edge_radius=6, n=1.6)
#
# # Окуляр (линза побольше, фокус подбираем для комфортного просмотра)
# # Расстояние между линзами (тубус) выберем так, чтобы окуляр стоял за фокусом объектива
# eye_x = 35
# eyepiece = UniversalLens(center_x=eye_x, R1=20, R2=20, thickness=5.5, edge_radius=10, n=1.5)
#
# system = [*objective.get_surfaces(), *eyepiece.get_surfaces()]
#
# # 3. Предмет (Источник света в одной точке)
# # Помещаем предмет чуть дальше переднего фокуса объектива
# object_pos = [obj_x - objective.f_dist - 2.5, 0.5, 0]
#
# # Создаем расходящийся пучок от одной точки предмета
# num_rays = 50
# angles = np.linspace(-0.1, 0.1, num_rays)
#
# for angle in angles:
#     direction = [np.cos(angle), np.sin(angle), 0]
#     ray = Ray(origin=object_pos, direction=direction)
#
#     # Трассировка через всю систему (увеличим количество отскоков и длину)
#     trajectory = run_simulation(ray, system, max_bounces=10)
#
#     # Отрисовка луча
#     path = pv.PolyData(trajectory)
#     lines = np.hstack(([len(trajectory)], range(len(trajectory))))
#     path.lines = lines
#     plotter.add_mesh(path, color="yellow", line_width=1.5, opacity=0.7)
#
#     # Маркеры столкновений
#     if len(trajectory) > 2:
#         hits = pv.PolyData(trajectory[1:-1])
#         plotter.add_mesh(hits, color="red", point_size=8, render_points_as_spheres=True)
#
# # 4. Визуализация элементов
# # Объектив
# plotter.add_mesh(objective.get_mesh(), color="cyan", opacity=0.3)
# objective.draw_axis(plotter, length=40)
#
# # Окуляр
# plotter.add_mesh(eyepiece.get_mesh(), color="lightblue", opacity=0.3)
# eyepiece.draw_axis(plotter, length=60)
#
# # Подписи
# plotter.add_point_labels([object_pos], ["Предмет"], font_size=12, text_color="white")
# plotter.add_point_labels([[obj_x, 8, 0], [eye_x, 12, 0]], ["Объектив", "Окуляр"], font_size=14)
#
# plotter.view_xy()
# plotter.show()

plotter.view_xy()
plotter.enable_parallel_projection()
plotter.enable_terrain_style(mouse_wheel_zooms=True)
plotter.show()
