import time
import numpy as np
import pyvista as pv
import os
from main import *        # ваш основной модуль
from fast_math import *

# ---------------------- ГЛОБАЛЬНЫЕ НАСТРОЙКИ ----------------------
global_interaction = True      # True – лучи видят все объекты, False – только "свой"
state = {'paused': False}

# Флаги клавиш (True – зажата)
keys = {
    'w': False, 'a': False, 's': False, 'd': False,   # перемещение объектов
    'q': False, 'e': False,                            # поворот объектов
    't': False, 'f': False, 'g': False, 'h': False,    # перемещение лучей
    'r': False, 'y': False                             # поворот лучей
}

# ---------------------- ОБЪЕКТЫ СЦЕНЫ ----------------------
# 1. Плоское зеркало
plane_mirror = PlaneSurface(
    point=[0, -20, 0], rotation_degrees=(0, 0, 45),
    half_sizes=(2.0, 2.0), n_inside=1.0,
    reflection_range=(0, 10000)
)

# 2. Сферическое вогнутое зеркало (исправлены lens_origin и lens_axis)
sphere_mirror = SphereSurface(radius=15.0,edge_radius=4.0, thickness=10.0,lens_origin=[10, -5, 0],rotation_degrees=[0, 0, 180],n_inside=1.0, reflection_range=(0, 10000))

# 3. Толстая линза 1
lens1 = UniversalLens(
    origin=[0, 10, 0], rotation_degrees=(0, 0, 0),
    R1=10.0, R2=-20.0, thickness=0.5, edge_radius=3.5, n=1.5,
    refraction_range=(0, 10000), reflection_range=(0, 10000),
)

# 4. Толстая линза 2
lens2 = UniversalLens(
    origin=[0, 25, 0], rotation_degrees=(0, 0, 0),
    R1=10.0, R2=-20.0, thickness=0.2, edge_radius=3.0, n=1.5,
    refraction_range=(0, 10000), reflection_range=(0, 10000),
)

# 5. Параболическое зеркало
parabolic_mirror = AsphericSurface(
    center=[0, 40, 0],
    radius=-5.0, conic_constant=-1.0,
    edge_radius=4.5, thickness=10.0,
    reflection_range=(0, 10000), n_inside=1.0
)

# 6. Гиперболическая линза
hyperbolic_lens = AsphericSurface(
    center=[0, 55, 0],
    radius=5.0, conic_constant=-5.0,
    edge_radius=4.0, thickness=10.0,
    refraction_range=(0, 10000),reflection_range=(0, 10000), n_inside=1.5
)

# 7. Произвольная модель (MeshSurface) – путь замените на свой
mesh_path = "../Models/Prism.stl"
if os.path.exists(mesh_path):
    tri_mesh = trimesh.load(mesh_path)
    tri_mesh.apply_scale(2.0)
    mesh_obj = MeshSurface(
        tri_mesh, n_inside=1.5,
        refraction_range=(0, 10000), reflection_range=(0, 10000),
        translation=[0, 70, 0], rotation_degrees=(0, 0, 0)
    )
else:
    mesh_obj = None
    print("Файл модели не найден, MeshSurface пропущен.")

# Упаковка объектов в список (объект, имя, цвет лучей по умолчанию)
objects = [
    (plane_mirror, "plane", "red"),
    (sphere_mirror, "sphere", "green"),
    (lens1, "lens1", "blue"),
    (lens2, "lens2", "cyan"),
    (parabolic_mirror, "parabolic", "orange"),
    (hyperbolic_lens, "hyperbolic", "magenta"),
]
if mesh_obj:
    objects.append((mesh_obj, "mesh", "yellow"))

# ---------------------- ЛУЧИ ----------------------
rays_dict = {}   # ключ – имя объекта
for obj, name, color in objects:
    if obj is None:
        continue
    # Определяем Y‑центр объекта для создания лучей
    if isinstance(obj, PlaneSurface):
        y0 = obj.point[1]
        dy = 2.0
    elif isinstance(obj, SphereSurface):
        y0 = obj.lens_origin[1]
        dy = 3.0
    elif isinstance(obj, UniversalLens):
        y0 = obj.origin[1]
        dy = 2.5
    elif isinstance(obj, AsphericSurface):
        y0 = obj.center[1]
        dy = 3.0
    elif isinstance(obj, MeshSurface):
        y0 = 70.0
        dy = 3.0
    else:
        y0, dy = 0.0, 2.0

    ray_list = []
    for y_shift in np.linspace(-dy, dy, 20):
        ray = Ray(
            origin=[-15, y0 + y_shift, 0],
            direction=[1, 0, 0],
            color=color,
            wavelength=550 if color != "red" else 650,
            energy_color_type=1
        )
        ray_list.append(ray)
    rays_dict[name] = ray_list

# Акторы объектов (без PBR, чтобы избежать проблем с обновлением)
actors = {}
for obj, name, _ in objects:
    if obj is None:
        continue
    actor = plotter.add_mesh(obj.get_mesh(), color="white", opacity=0.8)
    actors[name] = actor

# Трассировщик
tracer = RayTracer(
    plotter,
    mode='tree',
    max_depth=10,
    min_energy=0.000001,
    offset_distance=0.01,
    energy_color_type=2,
    default_color="white"
)

# ---------------------- БЛОКИРОВКА СТАНДАРТНЫХ КЛАВИШ ----------------------
# Список клавиш, которые мы забираем себе (строчные и заглавные)
blocked_keys = {
    'w','a','s','d','q','e','r','t','f','g','h','y','m','space',
    'W','A','S','D','Q','E','R','T','F','G','H','Y','M','Space'
}

def block_vtk_keys(interactor, event):
    key = interactor.GetKeySym()
    if key in blocked_keys:
        # Подменяем код клавиши на неиспользуемый, чтобы VTK ничего не делал
        interactor.SetKeyCode("\x00")
        interactor.SetKeySym("")
        return

# Наблюдатель с высоким приоритетом, вызывается перед стандартной обработкой
plotter.iren.interactor.AddObserver("CharEvent", block_vtk_keys, 10.0)

# ---------------------- НАШИ ОБРАБОТЧИКИ ----------------------
def on_key_press(interactor, event):
    key = interactor.GetKeySym().lower()
    if key in keys:                     # wasd, qe, tfgh, ry
        keys[key] = True
    elif key == 'space':
        state['paused'] = not state['paused']
    elif key == 'm':
        global global_interaction
        global_interaction = not global_interaction

def on_key_release(interactor, event):
    key = interactor.GetKeySym().lower()
    if key in keys:
        keys[key] = False

plotter.iren.add_observer('KeyPressEvent', on_key_press)
plotter.iren.add_observer('KeyReleaseEvent', on_key_release)


# ---------------------- ГЛАВНЫЙ ЦИКЛ ----------------------
last_time = time.time()
plotter.reset_camera()
plotter.show(interactive_update=True)
while plotter.render:
    if state['paused']:
        plotter.update()
        last_time = time.time()
        continue

    dt = time.time() - last_time
    last_time = time.time()
    if dt > 0.1:
        dt = 0.1

    move_speed = 5.0      # ед./с
    rot_speed = 30.0      # град/с

    # --- Перемещение объектов (WASD) ---
    delta_obj = np.array([0.0, 0.0, 0.0])
    if keys['w']: delta_obj[1] += move_speed * dt
    if keys['s']: delta_obj[1] -= move_speed * dt
    if keys['a']: delta_obj[0] -= move_speed * dt
    if keys['d']: delta_obj[0] += move_speed * dt
    if np.any(delta_obj != 0):
        for obj, _, _ in objects:
            obj.translate(delta_obj)

    # --- Поворот объектов вокруг Z (Q/E) ---
    if keys['q']:
        for obj, _, _ in objects:
            obj.rotate((0, 0, rot_speed * dt))
    if keys['e']:
        for obj, _, _ in objects:
            obj.rotate((0, 0, -rot_speed * dt))

    # --- Перемещение начальных точек лучей (TFGH) ---
    delta_rays = np.array([0.0, 0.0, 0.0])
    if keys['t']: delta_rays[1] += move_speed * dt
    if keys['g']: delta_rays[1] -= move_speed * dt
    if keys['f']: delta_rays[0] -= move_speed * dt
    if keys['h']: delta_rays[0] += move_speed * dt

    # --- Поворот направления лучей вокруг Z (R/Y) ---
    angle_ray = 0.0
    if keys['r']:
        angle_ray = rot_speed * dt
    elif keys['y']:
        angle_ray = -rot_speed * dt

    if angle_ray != 0.0:
        cos_a, sin_a = np.cos(np.radians(angle_ray)), np.sin(np.radians(angle_ray))
        rot_matrix = np.array([[cos_a, -sin_a, 0],
                               [sin_a,  cos_a, 0],
                               [0,      0,     1]])

    for name, ray_list in rays_dict.items():
        for ray in ray_list:
            if np.any(delta_rays != 0):
                ray.origin += delta_rays
            if angle_ray != 0.0:
                ray.direction = rot_matrix @ ray.direction
                ray.direction /= np.linalg.norm(ray.direction)

    # --- Обновление мешей объектов ---
    for name, actor in actors.items():
        obj = next(o for o, n, _ in objects if n == name)
        actor.mapper.dataset.copy_from(obj.get_mesh())

    # --- Трассировка лучей ---
    all_segments, all_colors, all_types = [], [], []
    if global_interaction:
        # Все объекты и все лучи
        tracer.elements.clear()
        for obj, _, _ in objects:
            if obj: tracer.elements.append(obj)
        tracer.rays.clear()
        for ray_list in rays_dict.values():
            tracer.rays.extend(ray_list)
        result, colors, types = tracer.trace_all()
        all_segments.extend(result)
        all_colors.extend(colors)
        all_types.extend(types)
    else:
        # Каждый объект со своими лучами
        for obj, name, _ in objects:
            if obj is None or name not in rays_dict:
                continue
            tracer.elements.clear()
            tracer.elements.append(obj)
            tracer.rays.clear()
            tracer.rays.extend(rays_dict[name])
            result, colors, types = tracer.trace_all()
            all_segments.extend(result)
            all_colors.extend(colors)
            all_types.extend(types)

    if all_segments:
        tracer.cloud.update_from_segments(
            all_segments, base_colors=all_colors, energy_types=all_types
        )

    plotter.update()