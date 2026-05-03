"""
Есть система из двух линз с одинаковым фокусным расстоянием. Расстояние между ними 2 фокуса.
Точечный источник света находится слева от левой линзы на расстоянии 0.5Ф и на некоторой высоте от главной оптической оси.
Как построить изображение этой точки
"""
import numpy as np

from main import *

# Параметры системы
f = 4.0                          # фокусное расстояние каждой линзы
d = 2 * f                        # расстояние между линзами
source_x = -0.5 * f              # расстояние от источника до левой линзы
source_y = 2.0                   # высота источника над осью (ось Y)

# Позиции линз
lens1_center = np.array([0.0, 0.0, 0.0])
lens2_center = np.array([d, 0.0, 0.0])

# Создаём линзы (собирающие, тонкие)
lens1 = ThinLens(center=lens1_center, focal_length=f, edge_radius=3.0,
                 axis_dir=np.array([1.0, 0.0, 0.0]))
lens2 = ThinLens(center=lens2_center, focal_length=f, edge_radius=3.0,
                 axis_dir=np.array([1.0, 0.0, 0.0]))

# Источник – точечный (просто для визуализации)
source_pos = np.array([source_x, source_y, 0.0])

# Набор лучей из источника под разными углами к оси X
# Углы от -25° до +25°, чтобы покрыть апертуру линз
y = np.linspace(1, -1, 11)  # 11 лучей
rays = []
for dy in y:
    # direction = np.array([np.cos(np.radians(angle)),
    #                       np.sin(np.radians(angle)),
    #                       0.0])
    rays.append(Ray(origin=np.array([source_x, source_y, 0]), direction=np.array([1, 0, 0]),
                    energy=1.0, current_n=1.0,
                    color="yellow", wavelength=550,
                    energy_color_type=2))

# Сцена

# Визуализация линз
plotter.add_mesh(lens1.get_mesh(), color="cyan", opacity=0.3, show_edges=True)
plotter.add_mesh(lens2.get_mesh(), color="cyan", opacity=0.3, show_edges=True)

# Визуализация источника (красная точка)
plotter.add_points(source_pos.reshape(1,3), color="red", point_size=15, render_points_as_spheres=True)

# Трассировка всех лучей (режим 'tree' с глубиной 3 – чтобы прошли обе линзы)
segments, colors, types = [], [], []
for ray in rays:
    tracer = RayTracer(plotter, mode='tree', max_depth=3, min_energy=0.01, offset_distance=0.1)
    tracer.add_elements(lens1)
    tracer.add_elements(lens2)
    tracer.add_ray(ray)
    segs, cols, typs = tracer.trace_all()
    segments.extend(segs)
    colors.extend(cols)
    types.extend(typs)

# Облако лучей (один актор)
cloud = RayCloud(plotter, energy_color_type=2, default_color="yellow")
cloud.update_from_segments(segments, base_colors=colors, energy_types=types)

# Подписи
plotter.add_point_labels([source_pos, lens1_center, lens2_center],
                         ["Источник", "Линза 1 (f)", f"Линза 2 (f)\nрасстояние 2f"],
                         font_size=10, text_color="white", shape=None, show_points=False)

# Настройка камеры
plotter.camera.position = (d+5, source_y*0.5, 15)
plotter.camera.focal_point = (d/2, source_y*0.5, 0)
plotter.reset_camera()
plotter.show()
