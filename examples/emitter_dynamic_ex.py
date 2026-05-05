import time
import numpy as np
import pyvista as pv
from main import *

# -------------------------------
# Создание сцены и объектов
# -------------------------------

pool = RayPool(initial_size=500)

# Призма и экран для взаимодействия
tri = trimesh.load("../Models/Prism.stl")
tri.apply_scale(3)
prism = MeshSurface(tri, n_inside=1.5, refraction_range=(0, 10000), reflection_range=(0, 10000),
                       translation=[0, 60, 0], rotation_degrees=(0, 0, 0),)

# Визуализация статических объектов
plotter.add_mesh(prism.get_mesh(), color="cyan", opacity=0.2, name="prism_mesh", show_edges=True)

# Излучатели
# 1. Изменяет min_offset и max_offset (колеблется ширина пучка)
emitter1 = BeamEmitter(
    origin=[-15, 4, 0],
    rotation_degrees=(0, 0, 76),
    num_rays=5, min_offset=-2, max_offset=2,
    color="red", wavelength=650, energy_color_type=2,
    pool=pool
)

# 2. Изменяет num_rays (меняется количество лучей)
emitter2 = BeamEmitter(
    origin=[-15, -4, 0],
    rotation_degrees=(0, 0, 78),
    num_rays=3, min_offset=-1.5, max_offset=1.5,
    color="blue", wavelength=450, energy_color_type=2,
    pool=pool
)

# 3. Вращается и перемещается (динамический)
emitter3 = BeamEmitter(
    origin=[-15, 10, 0],
    rotation_degrees=(0, 0, 45),
    num_rays=6, min_offset=-2, max_offset=2,
    color="green", wavelength=550, energy_color_type=2,
    pool=pool
)

# Акторы визуализации излучателей (стрелки)
emitter1_actor = plotter.add_mesh(emitter1.get_mesh(), color="red", name="emitter1")
emitter2_actor = plotter.add_mesh(emitter2.get_mesh(), color="blue", name="emitter2")
emitter3_actor = plotter.add_mesh(emitter3.get_mesh(), color="green", name="emitter3")

# Трассировщик (один на всё время)
tracer = RayTracer(plotter, mode='tree', max_depth=6, min_energy=0.01,
                   offset_distance=0.3, energy_color_type=2, default_color="yellow", pool=pool)
tracer.add_elements(prism)

# Параметры анимации
state = {'paused': False}
plotter.add_key_event("space", lambda: state.update(paused=not state['paused']))
last_time = time.time()
angle = 0.0  # для вращения emitter3

plotter.reset_camera()
plotter.show(interactive_update=True)

while plotter.render:
    if state['paused']:
        plotter.update()
        last_time = time.time()
        continue

    dt = time.time() - last_time
    last_time = time.time()
    if dt > 0.1: dt = 0.1

    # ---------- Анимация параметров ----------
    # Эмиттер 1: колебания ширины пучка
    t = time.time()
    amplitude = 2.0 + np.sin(t * 2) * 1.5   # от 0.5 до 3.5
    emitter1.min_offset = -amplitude
    emitter1.max_offset = amplitude
    # Визуализация: стрелка не меняется, только параметры

    # Эмиттер 2: меняем число лучей от 2 до 10
    new_count = 2 + int((np.sin(t * 3) * 0.5 + 0.5) * 8)  # 2..10
    emitter2.num_rays = max(2, new_count)

    # Эмиттер 3: вращаем и перемещаем
    angle += 10 * dt
    emitter3.rotate((0, 0, 10 * dt))       # вращение вокруг своей оси
    emitter3.translate(np.array([0.0, 0.05 * np.cos(angle * np.pi / 180), 0.0]))  # колебания по Y

    # Обновляем меши стрелок
    emitter1_actor.mapper.dataset.copy_from(emitter1.get_mesh())
    emitter2_actor.mapper.dataset.copy_from(emitter2.get_mesh())
    emitter3_actor.mapper.dataset.copy_from(emitter3.get_mesh())

    # ---------- Трассировка ----------
    tracer.elements.clear()
    tracer.add_elements(prism)

    # Добавляем лучи от эмиттеров
    for emitter in [emitter1, emitter2, emitter3]:
        for ray in emitter.emit():
            tracer.add_ray(ray)

    tracer.render()
    plotter.update()