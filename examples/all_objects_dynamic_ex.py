import time
import numpy as np
import pyvista as pv
from main import *


# Создаём объекты
plane_mirror = PlaneSurface(
    point=[0, -20, 0], rotation_degrees=(0, 0, 45),
    half_sizes=(2.0, 1.5), n_inside=1.0,
    reflection_range=(0, 10000)
)

sphere_mirror = SphereSurface(
    radius=-15.0, rotation_degrees=(0, 0, 0),
    edge_radius=3.0, thickness=10.0,
    lens_origin=[10, 8, 0], lens_axis=[-1, 0, 0],
    n_inside=1.0, reflection_range=(0, 10000)
)

lens = UniversalLens(
    origin=[0, 20, 0], rotation_degrees=(0, 0, 0),
    R1=10, R2=-20, thickness=0.2, edge_radius=3, n=1.5,
    refraction_range=(0, 10000), reflection_range=(0, 10000)
)

tri = trimesh.load("../Models/Prism.stl")
tri.apply_scale(3)
mesh_obj = MeshSurface(tri, n_inside=1.5, refraction_range=(0, 10000), reflection_range=(0, 10000),
                       translation=[0, 60, 0], rotation_degrees=(0, 0, 0),)


# Акторы для визуализации геометрии
plane_actor  = plotter.add_mesh(plane_mirror.get_mesh(), color="white", show_edges=True, name="plane")
sphere_actor = plotter.add_mesh(sphere_mirror.get_mesh(), color="white", metallic=0.9, name="sphere")
lens_actor   = plotter.add_mesh(lens.get_mesh(), color="cyan", opacity=0.5, name="lens")
mesh_actor   = plotter.add_mesh(mesh_obj.get_mesh(), color="gold", opacity=0.5, name="mesh")

# Создаём трассировщик с единственным облаком лучей
tracer = RayTracer(
    plotter,
    mode='tree',
    max_depth=6,
    min_energy=0.001,
    offset_distance=0.01,
    energy_color_type=2,
    default_color="yellow"
)

# ---------- Параметры анимации ----------
state = {'paused': False}
plotter.add_key_event("space", lambda: state.update(paused=not state['paused']))

plane_angle = 0.0
sphere_angle = 0.0
mesh_angle = 0.0
lens_x = 0.0
lens_dir = 1

last_time = time.time()
plotter.reset_camera()
plotter.show(interactive_update=True)

# ---------- Главный цикл ----------
while plotter.render:
    if state['paused']:
        plotter.update()
        last_time = time.time()
        continue

    # Дельта времени
    dt = time.time() - last_time
    last_time = time.time()
    if dt > 0.1:
        dt = 0.1

    plane_angle += 30 * dt
    plane_target_x = 2.0 * np.sin(np.radians(plane_angle))  # начальное смещение от 0
    # plane_mirror.rotate((0, 0, 6 * dt))  # вращение вокруг своего центра
    # plane_mirror.translate(np.array([plane_target_x - plane_mirror.point[0], 0.0, 0.0]))  # перемещение вдоль X

    # ---------- Обновление позиций ----------
    plane_angle += 30 * dt
    sphere_angle += 20 * dt
    mesh_angle += 25 * dt
    lens_x += 3.0 * dt * lens_dir
    if lens_x > 10:
        lens_x = 10.0
        lens_dir = -1
    elif lens_x < -10:
        lens_x = -10.0
        lens_dir = 1

    sphere_mirror.rotate((0, 0, 4 * dt))
    mesh_obj.rotate((0, 0, 20 * dt))
    lens.translate(np.array([lens_x - lens.origin[0], 0.0, 0.0]))
    lens.rotate((0, 0, 8 * dt))

    # ---------- Визуализация мешей ----------
    plane_actor.mapper.dataset.copy_from(plane_mirror.get_mesh())
    sphere_actor.mapper.dataset.copy_from(sphere_mirror.get_mesh())
    lens_actor.mapper.dataset.copy_from(lens.get_mesh())
    mesh_actor.mapper.dataset.copy_from(mesh_obj.get_mesh())

    # ---------- Трассировка ----------
    tracer.elements.clear()
    tracer.add_elements(plane_mirror, sphere_mirror, *lens.get_surfaces(), mesh_obj)

    # Добавляем лучи
    for y in np.linspace(-19, -21, 4):
        tracer.add_ray(Ray([-15, y, 0], [1, 0, 0], color="red", wavelength=650, energy_color_type=1))
    for y in np.linspace(8, 10, 40):
        tracer.add_ray(Ray([5, y, 0], [1, 0, 0], color="blue", wavelength=450, energy_color_type=1))
    for y in np.linspace(19, 21, 5):
        tracer.add_ray(Ray([-10, y, 0], [1, 0, 0], color="green", wavelength=550, energy_color_type=1))
    for y in np.linspace(60, 61, 5):
        tracer.add_ray(Ray([-15, y, 0], [1, 0, 0], color="magenta", wavelength=500, energy_color_type=1))

    # Автоматически обновляем RayCloud (правильный метод выбирается по self.mode)
    tracer.render()

    plotter.update()