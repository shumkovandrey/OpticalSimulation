import numpy as np
import pyvista as pv
from main import *       # ваш основной модуль (с исправленными классами)

# ---------- СОЗДАЁМ ЭЛЕМЕНТЫ ----------
# 1. Двояковыпуклая линза (фокусное расстояние ~ положительное)
lens_biconvex = UniversalLens(
    origin=[0, 10, 0],
    rotation_degrees=(0, 0, 0),
    R1=10, R2=-10, thickness=0.5, edge_radius=3.0, n=1.5,
    refraction_range=(0, 10000)
)

# 2. Плоско-выпуклая линза (R1=бесконечность, R2=-10)
lens_planoconvex = UniversalLens(
    origin=[0, 30, 0],
    rotation_degrees=(0, 0, 0),
    R1=None, R2=-10, thickness=0.5, edge_radius=3.0, n=1.5,
    refraction_range=(0, 10000)
)

# 3. Выпукло-вогнутая линза (мениск)
lens_meniscus = UniversalLens(
    origin=[0, 50, 0],
    rotation_degrees=(0, 0, 120),
    R1=15, R2=10, thickness=1, edge_radius=3.0, n=1.5,
    refraction_range=(0, 10000)
)

# 4. Вогнутое сферическое зеркало
sphere_mirror = SphereSurface(
    radius=-15,
    rotation_degrees=(0, 0, 0),
    edge_radius=4.0, thickness=10.0,
    n_inside=1.0, reflection_range=(0, 10000),
    lens_origin=[10, 70, 0]        # вершина, лучи идут слева направо
)

# ---------- ВИЗУАЛИЗАЦИЯ ----------

# Добавляем меши
# front_mesh = lens_biconvex.front.get_mesh()
# back_mesh = lens_biconvex.back.get_mesh()
# plotter.add_mesh(front_mesh, color="red", opacity=0.7, name="front_surface 3")
# plotter.add_mesh(back_mesh, color="blue", opacity=0.7, name="back_surface 3")
plotter.add_mesh(lens_biconvex.get_mesh(), color="cyan", opacity=0.5, name="biconvex")
# front_mesh = lens_planoconvex.front.get_mesh()
# back_mesh = lens_planoconvex.back.get_mesh()
# plotter.add_mesh(front_mesh, color="red", opacity=0.7, name="front_surface")
# plotter.add_mesh(back_mesh, color="blue", opacity=0.7, name="back_surface")
plotter.add_mesh(lens_planoconvex.get_mesh(), color="blue", opacity=0.5, name="planoconvex")
# front_mesh_2 = lens_meniscus.front.get_mesh()
# back_mesh_2 = lens_meniscus.back.get_mesh()
# plotter.add_mesh(front_mesh_2, color="red", opacity=0.7, name="front_surface 2")
# plotter.add_mesh(back_mesh_2, color="blue", opacity=0.7, name="back_surface 2")
plotter.add_mesh(lens_meniscus.get_mesh(), color="magenta", opacity=0.5, name="meniscus")
plotter.add_mesh(sphere_mirror.get_mesh(), color="white", name="mirror")

# Трассировщик
tracer = RayTracer(plotter, mode='tree', max_depth=5, min_energy=0.00001,
                   offset_distance=0.01, energy_color_type=0, default_color="yellow")

# Добавляем элементы в трейсер
tracer.elements.clear()
tracer.add_elements(lens_biconvex, lens_planoconvex, lens_meniscus, sphere_mirror)

# Лучи для каждой линзы и зеркала
# Для двояковыпуклой (y=10)
for y_off in np.linspace(-2.5, 2.5, 7):
    tracer.add_ray(Ray([-15, 10 + y_off, 0], [1, 0, 0], color="red", wavelength=650, energy_color_type=1))
# Для плоско-выпуклой (y=30)
for y_off in np.linspace(-2.5, 2.5, 7):
    tracer.add_ray(Ray([-15, 30 + y_off, 0], [1, 0, 0], color="green", wavelength=550, energy_color_type=1))
# Для мениска (y=50)
for y_off in np.linspace(-2.5, 2.5, 7):
    tracer.add_ray(Ray([-15, 50 + y_off, 0], [1, 0, 0], color="orange", wavelength=500, energy_color_type=1))
# Для зеркала (y=70)
for y_off in np.linspace(-3.5, 3.5, 7):
    tracer.add_ray(Ray([-15, 70 + y_off, 0], [1, 0, 0], color="yellow", wavelength=580, energy_color_type=1))

# Трассируем и показываем результат
tracer.render()
plotter.reset_camera()
plotter.show()