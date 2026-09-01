import pyvista as pv
import numpy as np
from main import *

# 1. Инициализация окна
plotter = pv.Plotter()
plotter.set_background("black")
plotter.view_isometric()
plotter.enable_parallel_projection()
plotter.enable_terrain_style(mouse_wheel_zooms=True)
plotter.add_axes(color="white")

# 2. Создание двояковыпуклой линзы
lens = UniversalLens(
    origin=(0.0, 0.0, 0.0),          # центр линзы
    rotation_degrees=(0, 0, 0),      # оптическая ось вдоль X
    R1=10.0,                         # радиус передней поверхности (выпуклая)
    R2=10.0,                        # радиус задней поверхности (выпуклая)
    thickness=2.0,                   # толщина по оси
    edge_radius=3.0,                 # радиус апертуры
    n=1.5,                           # показатель преломления
    reflection_range=(0, np.inf),           # отражение отключено
    refraction_range=(0, np.inf),    # преломление во всём диапазоне
    absorption_range=None
)

# Добавляем линзу на сцену как полупрозрачный объект
plotter.add_mesh(
    lens.get_mesh(),
    color="cyan",
    opacity=0.6,
    smooth_shading=True
)

# 3. Создание эмиттера пучка параллельных лучей
emitter = BeamEmitter(
    origin=(-20.0, 0.0, 0.0),        # стартовая точка эмиттера
    direction=(1.0, 0.0, 0.0),       # направление распространения (вдоль X)
    rotation_degrees=(0, 0, 0),      # без дополнительного поворота
    num_rays=20,                      # количество лучей (нечётное, чтобы был центральный)
    min_offset=-1.0,                 # минимальное смещение от оси в плоскости YZ
    max_offset=1.0,                  # максимальное смещение
    color="green",                  # цвет лучей
    wavelength=550.0,                # длина волны (нм)
    energy_color_type=2,             # прозрачность зависит от энергии (гамма)
    energy=0.1,                      # начальная энергия
    current_n=1.0                    # показатель преломления среды
)

# 4. Создание трассировщика
tracer = RayTracer(
    plotter=plotter,
    mode='tree',                   # последовательная трассировка без ветвления
    pool=None,                       # без пула лучей
    default_color="yellow",
    line_width=2.0,
    min_alpha=0.05,
    gamma=0.3
)

# 5. Добавление эмиттера и линзы в трассировщик
tracer.add_emitter(emitter)
tracer.add_elements(lens)            # линза как единый объект сцены

# 6. Трассировка всех лучей и обновление облака отрезков на сцене
tracer.render()

# 7. Показ сцены
plotter.show()