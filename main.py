import numpy as np

import pyvista as pv


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


class SphereSurface(OpticalElement):
    def __init__(self, center, radius, n_inside=1.5, x_min=-np.inf, x_max=np.inf):
        super().__init__()
        self.center = np.array(center, dtype=float)
        self.radius = radius
        self.n = n_inside
        self.x_limit = (x_min, x_max) # Допустимый диапазон по X

    def intersect(self, ray):
        oc = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius ** 2
        discriminant = b ** 2 - 4 * a * c

        if discriminant < 0: return None

        t1 = (-b - np.sqrt(discriminant)) / (2.0 * a)
        t2 = (-b + np.sqrt(discriminant)) / (2.0 * a)

        for t in sorted([t1, t2]):
            if t > 1e-6:
                hit_point = ray.origin + ray.direction * t
                # ПРОВЕРКА: попадает ли точка в "тело" линзы по оси X?
                if self.x_limit[0] <= hit_point[0] <= self.x_limit[1]:
                    return t
        return None

    def get_normal(self, point):
        # Нормаль всегда от центра сферы наружу
        return (point - self.center) / self.radius

    def interact(self, ray_dir, normal, n_from, n_to):
        # Коэффициент преломления
        eta = n_from / n_to

        # Убеждаемся, что нормаль направлена навстречу лучу
        # Если мы выходим из линзы, нормаль и луч смотрят в одну сторону,
        # поэтому для формулы Снеллиуса нормаль нужно инвертировать
        cos_i = np.dot(normal, ray_dir)
        actual_normal = normal
        if cos_i > 0:
            actual_normal = -normal
            cos_i = np.dot(actual_normal, ray_dir)

        cos_i = -cos_i  # Теперь cos_i положительный

        sin2_t = eta ** 2 * (1.0 - cos_i ** 2)
        if sin2_t > 1.0:  # Полное внутреннее отражение
            return ray_dir - 2 * np.dot(ray_dir, actual_normal) * actual_normal

        cos_t = np.sqrt(1.0 - sin2_t)
        return eta * ray_dir + (eta * cos_i - cos_t) * actual_normal


class Lens:
    def __init__(self, center_x, radius1, radius2, thickness, n=1.5, show_axis=True):
        self.center_x = center_x
        self.thickness = thickness
        self.n = n
        self.show_axis = show_axis

        # Левая вершина линзы в точке (center_x - thickness/2)
        # Чтобы поверхность была выпуклой влево, центр сферы должен быть ПРАВЕЕ вершины
        c1 = (center_x - thickness / 2) + radius1

        # Правая вершина линзы в точке (center_x + thickness/2)
        # Чтобы поверхность была выпуклой вправо, центр сферы должен быть ЛЕВЕЕ вершины
        c2 = (center_x + thickness / 2) - radius2

        self.front = SphereSurface(center=[c1, 0, 0], radius=radius1, n_inside=n)
        self.back = SphereSurface(center=[c2, 0, 0], radius=radius2, n_inside=n)

        left_v = center_x - thickness / 2
        right_v = center_x + thickness / 2

        self.front = SphereSurface(center=[c1, 0, 0], radius=radius1, n_inside=n,
                                   x_min=left_v, x_max=right_v)
        self.back = SphereSurface(center=[c2, 0, 0], radius=radius2, n_inside=n,
                                  x_min=left_v, x_max=right_v)

        # Расчет фокусного расстояния (формула для толстой линзы)
        # R1 и R2 для формулы (R2 отрицательный для выпуклой стороны)
        r1, r2 = radius1, -radius2
        inv_f = (n - 1) * (1 / r1 - 1 / r2 + ((n - 1) * thickness) / (n * r1 * r2))
        self.f_dist = 1 / inv_f

    def get_surfaces(self):
        # ВАЖНО: порядок в списке определяет, с чем луч столкнется первым
        # Сортируем поверхности по их реальному положению на оси X
        return [self.front, self.back]

    def get_mesh(self):
        """Создает меш через пересечение двух сфер"""
        s1 = pv.Sphere(radius=self.front.radius, center=self.front.center, phi_resolution=100, theta_resolution=100)
        s2 = pv.Sphere(radius=self.back.radius, center=self.back.center, phi_resolution=100, theta_resolution=100)
        # Оставляем только общую часть
        return s1.boolean_intersection(s2)

    def draw_axis(self, plotter, length=300):
        if not self.show_axis: return

        # Основная линия оси
        start = [self.center_x - length / 2, 0, 0]
        stop = [self.center_x + length / 2, 0, 0]
        axis_line = pv.Line(start, stop)
        plotter.add_mesh(axis_line, color="white", line_width=1, label="Оптическая ось")

        # Метки фокусов: F, 2F, 3F и т.д.
        for m in range(-10, 10):
            fx = self.center_x + m * self.f_dist
            label = f"{abs(m)}F"
            if m < 0: label = "-" + label

            # Рисуем черточку фокуса
            mark = pv.Line([fx, -0.2, 0], [fx, 0.2, 0])
            plotter.add_mesh(mark, color="red", line_width=2)
            plotter.add_point_labels([fx, 0.3, 0], [label], font_size=12, text_color="white")


class UniversalLens:
    def __init__(self, center_x, R1, R2, thickness, edge_radius, n=1.5, show_axis=True):
        """
        R > 0: выпуклая
        R < 0: вогнутая
        R is None: плоская
        """
        self.center_x = center_x
        self.thickness = thickness
        self.n = n
        self.edge_radius = edge_radius
        self.show_axis = show_axis

        self.R1 = R1
        self.R2 = R2

        # Координаты вершин линзы
        v1 = center_x - thickness / 2
        v2 = center_x + thickness / 2

        margin = abs(R1) if R1 else 5

        # --- Создание ПЕРВОЙ поверхности ---
        if R1 is None or R1 == 0:
            self.front = PlaneSurface(point=[v1, 0, 0], normal=[-1, 0, 0], n_inside=n)
        else:
            # Для выпуклой R1 > 0 центр впереди, для вогнутой R1 < 0 центр сзади
            c1 = v1 + R1
            self.front = SphereSurface(center=[c1, 0, 0], radius=abs(R1), n_inside=n, x_min=v1-margin, x_max=v2+margin)

        # --- Создание ВТОРОЙ поверхности ---
        if R2 is None or R2 == 0:
            self.back = PlaneSurface(point=[v2, 0, 0], normal=[1, 0, 0], n_inside=n)
        else:
            # Для выпуклой R2 > 0 центр сзади (v2 - R2), для вогнутой R2 < 0 центр впереди
            c2 = v2 - R2
            self.back = SphereSurface(center=[c2, 0, 0], radius=abs(R2), n_inside=n, x_min=v1-margin, x_max=v2+margin)

        # Расчет фокусного расстояния (общая формула)
        r1_val = R1 if R1 else 1e10  # Бесконечность для плоскости
        r2_val = -R2 if R2 else -1e10
        inv_f = (n - 1) * (1 / r1_val - 1 / r2_val + ((n - 1) * thickness) / (n * r1_val * r2_val))
        self.f_dist = 1 / inv_f if abs(inv_f) > 1e-10 else float('inf')

    def get_surfaces(self):
        return [self.front, self.back]

    def get_mesh(self):
        # 1. Создаем базовую заготовку - цилиндр
        # Его высота должна быть чуть больше thickness, чтобы сферы его прорезали
        cylinder = pv.Cylinder(center=(self.center_x, 0, 0), direction=(1, 0, 0),
                               radius=self.edge_radius, height=self.thickness + 2.0)
        # Переводим в PolyData и делаем сетку замкнутой (Triangulate)
        lens_body = cylinder.extract_surface().triangulate()

        # 2. Создаем меши сфер
        # Разрешение 60-80 достаточно для гладкости
        s1 = pv.Sphere(radius=self.front.radius, center=self.front.center,
                       phi_resolution=80, theta_resolution=80).triangulate()
        s2 = pv.Sphere(radius=self.back.radius, center=self.back.center,
                       phi_resolution=80, theta_resolution=80).triangulate()

        # 3. Логика формирования формы
        # Обрезаем переднюю сторону
        if isinstance(self.front, SphereSurface):
            if self.R1 > 0: # Выпуклая: пересекаем с цилиндром
                lens_body = lens_body.boolean_intersection(s1)
            else: # Вогнутая: вычитаем сферу из цилиндра
                lens_body = lens_body.boolean_difference(s1)
        else:
            # Плоская: просто режем плоскостью
            lens_body = lens_body.clip(normal=[-1, 0, 0], origin=self.front.point, invert=False)

        # Обрезаем заднюю сторону
        if isinstance(self.back, SphereSurface):
            if self.R2 > 0: # Выпуклая
                lens_body = lens_body.boolean_intersection(s2)
            else: # Вогнутая
                lens_body = lens_body.boolean_difference(s2)
        else:
            lens_body = lens_body.clip(normal=[1, 0, 0], origin=self.back.point, invert=False)

        return lens_body

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

    print(f"--- Запуск луча из {start_ray.origin} ---")

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
            print(f"  Удар {i + 1}: Поверхность id={id(hit_obj)} типа {type(hit_obj).__name__} в точке {hit_point}")

            normal = hit_obj.get_normal(hit_point)

            # ОПРЕДЕЛЯЕМ КУДА ИДЕМ:
            # Если мы были в воздухе (1.0), то идем в стекло (obj.n)
            # Если мы были в стекле (obj.n), то выходим в воздух (1.0)
            if abs(current_n - 1.0) < 1e-6:
                next_n = hit_obj.n
            else:
                next_n = 1.0

            new_dir = hit_obj.interact(current_ray.direction, normal, current_n, next_n)

            path.append(hit_point)
            current_ray = Ray(hit_point, new_dir)
            current_n = next_n  # ОБЯЗАТЕЛЬНО обновляем текущую среду
        else:
            print(f"  Удар {i + 1}: Промах (луч улетел)")
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
            plotter.add_mesh(lens.get_mesh(), color="cyan", opacity=0.5, smooth_shading=True)
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

# 2. Создаем "галерею" линз
# Каждую линзу смещаем по оси X, чтобы они не мешали друг другу
lenses = [
    # # Двояковыпуклая (Собирающая)
    # {"obj": UniversalLens(center_x=-40, R1=15, R2=15, thickness=5, edge_radius=7, n=1.5),
    #  "label": "Biconvex (Collecting)"},
    #
    # # Двояковогнутая (Рассеивающая)
    # {"obj": UniversalLens(center_x=0, R1=-15, R2=-15, thickness=2, edge_radius=7, n=1.5),
    #  "label": "Biconcave (Diverging)"},

    # Мениск (Выпукло-вогнутая)
    {"obj": UniversalLens(center_x=40, R1=10, R2=-20, thickness=3, edge_radius=7, n=1.5),
     "label": "Meniscus"}
]

# 3. Цикл отрисовки каждой линзы и её лучей
for item in lenses:
    lens = item["obj"]

    # Отрисовка меша линзы
    plotter.add_mesh(lens.get_mesh(), color="cyan", opacity=0.4, smooth_shading=True)

    # Отрисовка оптической оси и фокусов
    lens.draw_axis(plotter, length=100)

    # Подпись названия линзы
    plotter.add_point_labels([lens.center_x, 8, 0], [item["label"]], font_size=14, text_color="cyan")

    # Трассировка пучка лучей для текущей линзы
    system = lens.get_surfaces()
    y_range = np.linspace(5, -5, 100)  # 7 параллельных лучей

    for y in y_range:
        # 1. Создаем луч
        start_pos = [lens.center_x - 12, y, 0]
        ray = Ray(origin=start_pos, direction=[1, 0, 0])
        trajectory = run_simulation(ray, system, max_bounces=5)

        # 2. Отрисовка источника (начальной точки)
        # Создаем маленькую белую сферу в начале траектории
        source_point = pv.Sphere(radius=0.15, center=trajectory[0])
        plotter.add_mesh(source_point, color="white", emissive=True, label="Источник" if y == y_range[0] else "")

        # 3. Отрисовка самой линии луча
        path = pv.PolyData(trajectory)
        lines = np.hstack(([len(trajectory)], range(len(trajectory))))
        path.lines = lines

        # Можно сделать плавное затухание цвета от источника
        plotter.add_mesh(path, color="yellow", line_width=2, opacity=0.1)

        # 4. Точки столкновения (как и раньше)
        if len(trajectory) > 2:
            hits = pv.PolyData(trajectory[1:-1])
            plotter.add_mesh(hits, color="red", point_size=10, render_points_as_spheres=True)

# Настройка камеры и запуск
plotter.add_axes()
plotter.view_xy()  # По умолчанию смотрим в плоскости X-Y
plotter.show()