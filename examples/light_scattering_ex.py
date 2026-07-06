import random

import numpy as np
import pyvista as pv
from main import *          # ваш модуль со всеми классами и функциями
import trimesh

# ---------- Параметры ----------
n_rays = 3000                # количество лучей
beam_radius = 1.0           # радиус входного пучка
n_scatterers = 200           # число рассеивающих тел
scatter_region = (30.0, 0.5, 0.5)   # размеры области, где располагаются рассеиватели (X,Y,Z)

# ---------- Сцена ----------

# 1. Экран (регистрация)
screen = RectangularScreen(center=[15, 0, 10], rotation_degrees=(0, 90, 0), width=10, height=50)
plotter.add_mesh(screen.get_mesh(), color="white", opacity=1, name="screen")

# 2. Рассеивающие объекты (маленькие икосаэдры)
scatterers = []
for _ in range(n_scatterers):
    # Случайное положение в заданном объёме
    pos = [random.uniform(0, 10), random.uniform(-0.7, 0.7), random.uniform(-0.7, 0.7)]
    ico = trimesh.creation.icosahedron()
    ico.apply_scale(0.01)                     # маленький размер
    # ico.apply_translation(pos)               # перемещаем в нужную точку
    surf = MeshSurface(ico, n_inside=1.0, reflection_range=(0, 10000), rotation_degrees=(random.randint(0, 360), random.randint(0, 360), random.randint(0, 360)))
    surf.translate(pos)
    scatterers.append(surf)
    plotter.add_mesh(surf.get_mesh(), color="gray", opacity=0.7, pbr=True, metallic=0.9)

# 3. Генерация лучей (параллельный пучок, круглое сечение)
direction = np.array([1.0, 0.0, 0.0])
angles = np.random.uniform(0, 2*np.pi, n_rays)
radii = np.sqrt(np.random.uniform(0, beam_radius**2, n_rays))  # sqrt для равномерного распределения по площади
y_offsets = radii * np.cos(angles)
z_offsets = radii * np.sin(angles)
origins = np.column_stack([np.full(n_rays, -5.0), y_offsets, z_offsets])

# 4. Трассировка
tracer = RayTracer(plotter, mode='tree', max_depth=5, min_energy=0.005,
                   offset_distance=0.1, energy_color_type=2, default_color="yellow")
for elem in scatterers + [screen]:
    tracer.add_elements(elem)

for i in range(n_rays):
    tracer.add_ray(Ray(origin=origins[i], direction=direction,
                       energy=0.5, current_n=1.0, color="lime",
                       wavelength=550, energy_color_type=2))

cloud = tracer.render()
plotter.reset_camera()
plotter.show()