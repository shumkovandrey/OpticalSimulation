# # 1. Настройка плоттера
# plotter = pv.Plotter()
# plotter.set_background("black")
#
#
# # Функция для создания пучка лучей вдоль оптической оси линзы
# def trace_lens_bundle(lens, system, plotter):
#     # Генерируем 5 лучей в локальных координатах линзы
#     for y_off in np.linspace(-2, 2, 5):
#         # Локальный старт и направление
#         local_start = np.array([-15, y_off, 0])
#         local_dir = np.array([1, 0, 0])
#
#         # Переводим в мировые координаты с помощью матрицы вращения линзы
#         world_start = lens.origin + lens.rotation @ local_start
#         world_dir = lens.rotation @ local_dir
#
#         ray = Ray(origin=world_start, direction=world_dir)
#         traj = run_simulation(ray, system, max_bounces=5)
#
#         # Визуализация
#         path = pv.PolyData(traj)
#         lines = np.hstack(([len(traj)], range(len(traj))))
#         path.lines = lines
#         plotter.add_mesh(path, color="yellow", line_width=1.5, opacity=0.6)
#
#
# # 2. Создание линз с разной ориентацией
# # Все линзы направлены в сторону точки (0, 0, 0)
#
# # Линза 1: Стоит на оси X, смотрит в центр
# l1 = UniversalLens(origin=[-25, 0, 0], axis_dir=[1, 0, 0],
#                    R1=15, R2=15, thickness=5, edge_radius=8)
#
# # Линза 2: Стоит под углом 45 градусов в плоскости XY
# l2 = UniversalLens(origin=[20, 20, 0], axis_dir=[-1, -1, 0],
#                    R1=15, R2=15, thickness=5, edge_radius=8)
#
# # Линза 3: Стоит сверху и смотрит вниз (ось Z)
# l3 = UniversalLens(origin=[0, 0, 25], axis_dir=[0, 0, -1],
#                    R1=15, R2=15, thickness=5, edge_radius=8)
#
# lenses = [l1, l2, l3]
#
# # 3. Симуляция и отрисовка
# for l in lenses:
#     # Важно: система для каждого пучка должна включать только поверхности ЭТОЙ линзы
#     # (или все поверхности сразу, если они не перекрывают друг друга)
#     current_system = l.get_surfaces()
#
#     # Рисуем меш линзы
#     plotter.add_mesh(l.get_mesh(), color="cyan", opacity=0.3)
#
#     # Пускаем лучи
#     trace_lens_bundle(l, current_system, plotter)


# # 1. Настройка плоттера
# plotter = pv.Plotter()
# plotter.set_background("black")
#
# # 2. Оптические элементы
# # Линза в центре (X=0)
# lens = UniversalLens(center_x=0, R1=15, R2=15, thickness=5.5, edge_radius=8, n=1.5)
#
# # Зеркало 1: стоит после линзы, разворачивает луч на 90 градусов вбок (в плоскости XY)
# mirror1 = PlaneSurface(point=[15, 0, 0], normal=[-1, 1, 0], is_mirror=True)
#
# # Зеркало 2: принимает отраженный луч и уводит его ВВЕРХ (вдоль оси Z)
# mirror2 = PlaneSurface(point=[15, 15, 0], normal=[0, -1, 1], is_mirror=True)
#
# # Экран: ловит луч наверху
# screen = Screen(point=[15, 15, 15], normal=[0, 0, -1], size=10)
#
# system = [*lens.get_surfaces(), mirror1, mirror2, screen]
#
# # 3. Источник света (небольшой пучок)
# for y_off in np.linspace(-1, 1, 5):
#     ray = Ray(origin=[-20, y_off, 0], direction=[1, 0, 0])
#     trajectory = run_simulation(ray, system, max_bounces=10)
#
#     # Визуализация луча
#     path = pv.PolyData(trajectory)
#     lines = np.hstack(([len(trajectory)], range(len(trajectory))))
#     path.lines = lines
#     plotter.add_mesh(path, color="yellow", line_width=2, render_lines_as_tubes=True)
#
#     # Точки удара
#     if len(trajectory) > 2:
#         hits = pv.PolyData(trajectory[1:-1])
#         plotter.add_mesh(hits, color="red", point_size=10)
#
# # 4. Визуализация моделей
# # Линза
# plotter.add_mesh(lens.get_mesh(), color="cyan", opacity=0.3)
# # Зеркало 1
# m1_mesh = pv.Plane(center=mirror1.point, direction=mirror1.normal, i_size=10, j_size=10)
# plotter.add_mesh(m1_mesh, color="silver", opacity=0.8)
# # Зеркало 2
# m2_mesh = pv.Plane(center=mirror2.point, direction=mirror2.normal, i_size=10, j_size=10)
# plotter.add_mesh(m2_mesh, color="silver", opacity=0.8)
# # Экран
# s_mesh = pv.Plane(center=screen.point, direction=screen.normal, i_size=10, j_size=10)
# plotter.add_mesh(s_mesh, color="white", opacity=0.5)
#
# # Настройка камеры для 3D вида
# plotter.camera_position = [(-50, -50, 50), (10, 10, 0), (0, 0, 1)]
# plotter.add_axes()


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
