import time
import numpy as np
import pyvista as pv
import os
from main import *
from fast_math import *


def trajectories_to_segments(trajectories, colors):
    """Преобразует траектории (списки точек) в отрезки для единого облака."""
    segments = []
    for traj, color in zip(trajectories, colors):
        for i in range(len(traj) - 1):
            p1 = traj[i]
            p2 = traj[i+1]
            segments.append((p1, p2, 1.0, color))   # энергия 1.0, цвет луча
    return segments


# ---------------------- ГЛОБАЛЬНЫЕ НАСТРОЙКИ ----------------------
global_interaction = True
state = {'paused': False}

# --- Настройки лучей (изменяемые стрелками) ---
ray_count = 5
ray_spacing = 1.0
ray_count_min, ray_count_max = 1, 100
spacing_min, spacing_max = 0.1, 5.0

# Сохраняем базовые параметры для каждого объекта (для перегенерации)
obj_ray_defaults = {}

# Флаги зажатия клавиш
keys = {
    'w': False, 'a': False, 's': False, 'd': False,
    'q': False, 'e': False,
    't': False, 'f': False, 'g': False, 'h': False,
    'r': False, 'y': False
}

need_reset = False

# Переключение видимости (1..7) – защита от автоповтора
digit_pressed = {str(i): False for i in range(1, 8)}

# ---------------------- ФУНКЦИЯ СОЗДАНИЯ ОБЪЕКТОВ И ЛУЧЕЙ ----------------------
def create_scene():
    """Возвращает (objects, rays_dict, actors, mesh_obj)"""
    plane_mirror = PlaneSurface(
        point=[0, -20, 0], rotation_degrees=(0, 0, 45),
        half_sizes=(2.0, 2.0), n_inside=1.0,
        reflection_range=(0, 10000)
    )
    sphere_mirror = SphereSurface(
        radius=-15.0,
        rotation_degrees=(0, 0, 0),
        edge_radius=4.0, thickness=10.0,
        n_inside=1.0, reflection_range=(0, 10000),
        lens_origin=[10, -5, 0]
    )
    lens1 = UniversalLens(
        origin=[0, 10, 0], rotation_degrees=(0, 0, 0),
        R1=10.0, R2=10.0, thickness=3, edge_radius=3.5, n=1.5,
        refraction_range=(0, 10000), reflection_range=(0, 10000)
    )
    lens2 = UniversalLens(
        origin=[0, 25, 0], rotation_degrees=(0, 0, 0),
        R1=15.0, R2=-30.0, thickness=0.8, edge_radius=4.0, n=1.6,
        refraction_range=(0, 10000), reflection_range=(0, 10000)
    )
    parabolic_mirror = AsphericSurface(
        center=[0, 40, 0], rotation_degrees=(0, 0, 180),
        radius=5.0, conic_constant=-1.0,
        edge_radius=4.5, thickness=10.0,
        reflection_range=(0, 10000), n_inside=1.0
    )
    hyperbolic_lens = AsphericSurface(
        center=[0, 55, 0],
        radius=5.0, conic_constant=-5.0,
        edge_radius=4.0, thickness=10.0,
        refraction_range=(0, 10000), reflection_range=(0, 10000),
        n_inside=1.5
    )

    mesh_path = "../Models/Untitled.stl"
    if os.path.exists(mesh_path):
        tri_mesh = trimesh.load(mesh_path)
        tri_mesh.apply_scale(2.0)
        mesh_obj = MeshSurface(
            tri_mesh, n_inside=1.5,
            # refraction_range=(0, 10000),
            reflection_range=(0, 10000),
            translation=[0, 70, 0], rotation_degrees=(0, 90, 0)
        )
    else:
        mesh_obj = None
        print("Файл модели не найден, MeshSurface пропущен.")

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

    # Лучи
    rays_dict = {}
    for obj, name, color in objects:
        if obj is None:
            continue
        if isinstance(obj, PlaneSurface):
            y0, dy = obj.point[1], 2.0
        elif isinstance(obj, SphereSurface):
            y0, dy = obj.lens_origin[1], 3.0
        elif isinstance(obj, UniversalLens):
            y0, dy = obj.origin[1], 2.5
        elif isinstance(obj, AsphericSurface):
            y0, dy = obj.center[1], 3.0
        elif isinstance(obj, MeshSurface):
            y0, dy = 70.0, 3.0
        else:
            y0, dy = 0.0, 2.0

        ray_list = []
        for y_shift in np.linspace(-dy, dy, 4):
            ray = Ray(
                origin=[-15, y0 + y_shift, 0],
                direction=[1, 0, 0],
                color=color,
                wavelength=550 if color != "red" else 650,
                energy_color_type=2
            )
            ray_list.append(ray)
        rays_dict[name] = ray_list

    return objects, rays_dict, mesh_obj


def regenerate_rays():
    """Пересоздаёт все лучи с текущими ray_count и ray_spacing."""
    global rays_dict
    rays_dict.clear()
    for name, defaults in obj_ray_defaults.items():
        y0 = defaults['y0']
        base_dy = defaults['dy']
        half_range = base_dy * ray_spacing
        if ray_count == 1:
            offsets = [0.0]
        else:
            offsets = np.linspace(-half_range, half_range, ray_count)
        ray_list = []
        for y_shift in offsets:
            ray = Ray(
                origin=[-15, y0 + y_shift, 0],
                direction=[1, 0, 0],
                color=defaults['color'],
                wavelength=defaults['wavelength'],
                energy_color_type=defaults['energy_color_type']
            )
            ray_list.append(ray)
        rays_dict[name] = ray_list
    # Очищаем облако, чтобы новые лучи корректно отобразились
    tracer.cloud.update([])


# ---------------------- ПЕРВОНАЧАЛЬНОЕ СОЗДАНИЕ ----------------------
objects, rays_dict, mesh_obj = create_scene()
# Исходные копии для сброса
initial_objects, initial_rays_dict, _ = create_scene()

# Заполняем параметры для управления лучами
for obj, name, color in objects:
    if obj is None: continue
    if isinstance(obj, PlaneSurface):        y0, dy = obj.point[1], 2.0
    elif isinstance(obj, SphereSurface):     y0, dy = obj.lens_origin[1], 3.0
    elif isinstance(obj, UniversalLens):     y0, dy = obj.origin[1], 2.5
    elif isinstance(obj, AsphericSurface):   y0, dy = obj.center[1], 3.0
    elif isinstance(obj, MeshSurface):       y0, dy = 70.0, 3.0
    else:                                    y0, dy = 0.0, 2.0

    obj_ray_defaults[name] = {
        'y0': y0,
        'dy': dy,
        'color': color,
        'wavelength': 550 if color != "red" else 650,
        'energy_color_type': 1
    }

# Сохраняем начальные параметры объектов
initial_obj_states = []
for obj, name, _ in objects:
    obj_state = {}
    if isinstance(obj, PlaneSurface):
        obj_state['point'] = np.copy(obj.point)
        obj_state['normal'] = np.copy(obj.normal)
        obj_state['lens_origin'] = np.copy(obj.lens_origin)
        obj_state['lens_axis'] = np.copy(obj.lens_axis)
    elif isinstance(obj, SphereSurface):
        obj_state['lens_origin'] = np.copy(obj.lens_origin)
        obj_state['lens_axis'] = np.copy(obj.lens_axis)
        obj_state['center'] = np.copy(obj.center)
        obj_state['radius'] = obj.radius
    elif isinstance(obj, UniversalLens):
        obj_state['origin'] = np.copy(obj.origin)
        obj_state['axis_dir'] = np.copy(obj.axis_dir)
        obj_state['rotation'] = np.copy(obj.rotation)
        obj_state['rotation_degrees'] = obj.rotation_degrees
        obj_state['R1'] = obj.R1
        obj_state['R2'] = obj.R2
        obj_state['thickness'] = obj.thickness
        obj_state['edge_radius'] = obj.edge_radius
        obj_state['n'] = obj.n
    elif isinstance(obj, AsphericSurface):
        obj_state['center'] = np.copy(obj.center)
        obj_state['lens_origin'] = np.copy(obj.lens_origin)
        obj_state['lens_axis'] = np.copy(obj.lens_axis)
        obj_state['radius'] = obj.radius
    elif isinstance(obj, MeshSurface):
        # Для MeshSurface сохраняем только текущую матрицу трансформации?
        # Проще сохранить trimesh-объект и повторно применить translation/rotation
        obj_state['mesh'] = obj.mesh.copy()   # trimesh.Trimesh
    initial_obj_states.append((name, obj_state))

# Начальные параметры лучей
initial_ray_params = {}
for name, ray_list in rays_dict.items():
    params = []
    for ray in ray_list:
        params.append({
            'origin': np.copy(ray.origin),
            'direction': np.copy(ray.direction),
            'energy': 1.0,
            'color': ray.color,
            'wavelength': ray.wavelength,
            'energy_color_type': ray.energy_color_type
        })
    initial_ray_params[name] = params

# Сохраняем начальные параметры лучей (origin, direction, energy, color, wavelength, energy_color_type)
initial_ray_params = {}
for name, ray_list in rays_dict.items():
    params = []
    for ray in ray_list:
        params.append({
            'origin': np.copy(ray.origin),
            'direction': np.copy(ray.direction),
            'energy': 1.0,
            'color': ray.color,
            'wavelength': ray.wavelength,
            'energy_color_type': ray.energy_color_type
        })
    initial_ray_params[name] = params


# Видимость объектов (по умолчанию все видны)
object_visible = {name: True for _, name, _ in objects}

# ---------------------- ВИЗУАЛИЗАЦИЯ ----------------------
# Строим акторы
actors = {}
for obj, name, _ in objects:
    if obj is None:
        continue
    actor = plotter.add_mesh(obj.get_mesh(), color="white", opacity=0.8, name=name)
    actors[name] = actor


tracer = RayTracer(
    plotter,
    mode=TreeMode(max_depth=10, min_energy=0.001, offset_distance=0.1, energy_color_type=0),
    default_color="white"
)

# ---------------------- ОБРАБОТЧИКИ КЛАВИШ ----------------------
# Стрелки, которые мы перехватываем
arrow_keys = ["Left", "Right", "Up", "Down"]
arrow_pressed = {k: False for k in arrow_keys}

def on_key_press_early(interactor, event):
    """Перехват стрелок с высоким приоритетом, чтобы заблокировать стандартное поведение."""
    key = interactor.GetKeySym()
    if key not in arrow_keys:
        return  # не стрелка — сразу выходим, не мешаем другим обработчикам

    if key in arrow_keys:
        # Защита от автоповтора
        if arrow_pressed[key]:
            interactor.SetKeyCode("\x00")
            interactor.SetKeySym("")
            return
        arrow_pressed[key] = True

        global ray_count, ray_spacing
        need_regenerate = False
        if key == "Up":
            ray_spacing = min(ray_spacing + 0.1, spacing_max)
            need_regenerate = True
        elif key == "Down":
            ray_spacing = max(ray_spacing - 0.1, spacing_min)
            need_regenerate = True
        elif key == "Right":
            ray_count = min(ray_count + 1, ray_count_max)
            need_regenerate = True
        elif key == "Left":
            ray_count = max(ray_count - 1, ray_count_min)
            need_regenerate = True

        if need_regenerate:
            regenerate_rays()

        # Блокируем дальнейшую обработку стрелки VTK
        interactor.SetKeyCode("\x00")
        interactor.SetKeySym("")
        return


def on_key_press(interactor, event):
    key = interactor.GetKeySym().lower()
    # Непрерывные действия
    if key in keys:
        keys[key] = True
    elif key == 'space':
        state['paused'] = not state['paused']
    elif key == 'm':
        global global_interaction

        global_interaction = not global_interaction
    elif key == 'n':
        if isinstance(tracer.mode, SimpleMode):
            tracer.mode = TreeMode(max_depth=10, min_energy=0.001, offset_distance=0.1, energy_color_type=0)
        else:
            tracer.mode = SimpleMode(max_bounces=10, offset_distance=0.1, energy_color_type=0)
        tracer.cloud.update([], energy_color_type=tracer.mode.energy_color_type)  # очистка
    elif key == 'x':
        global need_reset
        need_reset = True
    elif key.isdigit() and 1 <= int(key) <= len(objects):
        # Защита от автоповтора – переключаем только при первом нажатии
        if not digit_pressed[key]:
            digit_pressed[key] = True
            idx = int(key) - 1
            obj, name, _ = objects[idx]
            object_visible[name] = not object_visible[name]
            actors[name].SetVisibility(object_visible[name])

def on_key_release(interactor, event):
    key = interactor.GetKeySym()
    # Сначала обрабатываем стрелки
    if key in arrow_keys:
        arrow_pressed[key] = False
        return
    key_lower = key.lower()
    if key_lower in keys:
        keys[key_lower] = False
    elif key_lower.isdigit() and 1 <= int(key_lower) <= len(objects):
        digit_pressed[key_lower] = False


# Блокировка нежелательных клавиш PyVista
problematic_keys = {'w', 'W', 's', 'S', 'r', 'R', 'q', 'Q', 'e', 'E', '3', '7', 'x', 'X'}

def block_vtk_keys(interactor, event):
    key = interactor.GetKeySym()
    if key in problematic_keys:
        interactor.SetKeyCode("\x00")
        interactor.SetKeySym("")
        return

plotter.iren.interactor.AddObserver("CharEvent", block_vtk_keys, 10.0)

# Перехват стрелок (высокий приоритет)
plotter.iren.interactor.AddObserver('KeyPressEvent', on_key_press_early, 10.0)
# Основные нажатия (буквы, цифры, пробел и т.д.)
plotter.iren.interactor.AddObserver('KeyPressEvent', on_key_press)
# Отпускание всех клавиш
plotter.iren.interactor.AddObserver('KeyReleaseEvent', on_key_release)

# ---------------------- СБРОС СЦЕНЫ ----------------------
def reset_scene():
    global ray_count, ray_spacing, objects, rays_dict, actors, object_visible, need_reset
    # 1. Очищаем облако лучей (старые линии исчезнут)
    tracer.cloud.update([])

    # 2. Восстанавливаем объекты из начальных состояний
    for (name, state) in initial_obj_states:
        # Находим текущий объект по имени (он тот же, что и был)
        obj = next(o for o, n, _ in objects if n == name)
        if isinstance(obj, PlaneSurface):
            obj.point = np.copy(state['point'])
            obj.normal = np.copy(state['normal'])
            obj.lens_origin = np.copy(state['lens_origin'])
            obj.lens_axis = np.copy(state['lens_axis'])
            if obj.face_tangents is not None:
                # Пересчитать тангенты по новой нормали
                obj.face_tangents = get_tangents(obj.normal)
        elif isinstance(obj, SphereSurface):
            obj.lens_origin = np.copy(state['lens_origin'])
            obj.lens_axis = np.copy(state['lens_axis'])
            obj.center = np.copy(state['center'])
            # edge_radius, thickness не менялись
        elif isinstance(obj, UniversalLens):
            obj.origin = np.copy(state['origin'])
            obj.axis_dir = np.copy(state['axis_dir'])
            obj.rotation = np.copy(state['rotation'])
            obj.rotation_degrees = state['rotation_degrees']
            # Пересоздаём поверхности с исходными параметрами
            obj.R1 = state['R1']
            obj.R2 = state['R2']
            obj.thickness = state['thickness']
            obj.edge_radius = state['edge_radius']
            obj.n = state['n']
            obj._create_surfaces()
        elif isinstance(obj, AsphericSurface):
            obj.center = np.copy(state['center'])
            obj.lens_origin = np.copy(state['lens_origin'])
            obj.lens_axis = np.copy(state['lens_axis'])
            obj.radius = state['radius']
            # Восстанавливаем правильные тангенциальные векторы
            obj._t1, obj._t2 = get_tangents(obj.lens_axis)
            # Обновляем матрицы перехода
            obj._rot_local_to_world = np.column_stack([obj.lens_axis, obj._t1, obj._t2])
            obj._rot_world_to_local = obj._rot_local_to_world.T
        elif isinstance(obj, MeshSurface):
            # Восстанавливаем исходный trimesh (без накопленных трансформаций)
            obj.mesh = state['mesh'].copy()
            obj.intersector = trimesh.ray.ray_triangle.RayMeshIntersector(obj.mesh)

    # Сбрасываем настройки лучей к исходным
    ray_count = 5
    ray_spacing = 1.0
    regenerate_rays()

    # 3. Сбрасываем лучи в начальное положение
    for name, ray_list in rays_dict.items():
        if name in initial_ray_params:
            for ray, init in zip(ray_list, initial_ray_params[name]):
                ray.origin = np.copy(init['origin'])
                ray.direction = np.copy(init['direction'])
                ray.energy = init['energy']
                # цвет, длина волны не менялись

    # 4. Обновляем меши акторов (акторы те же, геометрия новая)
    for name, actor in actors.items():
        obj = next(o for o, n, _ in objects if n == name)
        actor.mapper.dataset.copy_from(obj.get_mesh())

    # 5. Сбрасываем видимость и защиту клавиш
    for k in object_visible:
        object_visible[k] = True
    for k in digit_pressed:
        digit_pressed[k] = False

    # 6. Принудительно обновим сцену, чтобы изменения отобразились сразу
    plotter.render()


# ---------------------- ГЛАВНЫЙ ЦИКЛ ----------------------
last_time = time.time()
plotter.reset_camera()
plotter.show(interactive_update=True)

print("""
WASD - перемещение объектов
QE - вращение объектов
TFGH - перемещение лучей
RY - изменение направления лучей

------------------------------------------

1-7 - включить/выключить объекты и его лучи
вверх/вниз - увеличить/уменьшить расстояние между лучами
вправо/влево - увеличить/уменьшить кол-во лучей
X - сброс сцены

------------------------------------------

Пробел - заморозить/разморозить сцену
M - режим взаимодействия лучей со своим или со всеми объектами
N - режим взаимодействия лучей: simple/tree
""")

while plotter.render:
    if state['paused']:
        plotter.update()
        last_time = time.time()
        continue

    if need_reset:
        reset_scene()
        need_reset = False

    dt = time.time() - last_time
    last_time = time.time()
    if dt > 0.1:
        dt = 0.1

    move_speed = 5.0
    rot_speed = 30.0

    # Перемещение объектов (WASD)
    delta_obj = np.array([0.0, 0.0, 0.0])
    if keys['w']: delta_obj[1] += move_speed * dt
    if keys['s']: delta_obj[1] -= move_speed * dt
    if keys['a']: delta_obj[0] -= move_speed * dt
    if keys['d']: delta_obj[0] += move_speed * dt
    if np.any(delta_obj != 0):
        for obj, name, _ in objects:
            if object_visible[name]:
                obj.translate(delta_obj)

    # Поворот объектов (Q/E)
    if keys['q']:
        for obj, name, _ in objects:
            if object_visible[name]:
                obj.rotate((0, 0, rot_speed * dt))
    if keys['e']:
        for obj, name, _ in objects:
            if object_visible[name]:
                obj.rotate((0, 0, -rot_speed * dt))

    # Перемещение лучей (TFGH)
    delta_rays = np.array([0.0, 0.0, 0.0])
    if keys['t']: delta_rays[1] += move_speed * dt
    if keys['g']: delta_rays[1] -= move_speed * dt
    if keys['f']: delta_rays[0] -= move_speed * dt
    if keys['h']: delta_rays[0] += move_speed * dt

    # Поворот направления лучей (R/Y)
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
        if not object_visible[name]:
            continue
        for ray in ray_list:
            if np.any(delta_rays != 0):
                ray.origin += delta_rays
            if angle_ray != 0.0:
                ray.direction = rot_matrix @ ray.direction
                ray.direction /= np.linalg.norm(ray.direction)

    # Обновление мешей (только видимых)
    for name, actor in actors.items():
        if object_visible[name]:
            obj = next(o for o, n, _ in objects if n == name)
            actor.mapper.dataset.copy_from(obj.get_mesh())

    # --- Трассировка ---
    if global_interaction:
        tracer.elements.clear()
        tracer.rays.clear()
        for obj, name, _ in objects:
            if obj and object_visible[name]:
                tracer.elements.append(obj)
        for name, ray_list in rays_dict.items():
            if object_visible[name]:
                tracer.rays.extend(ray_list)

        segments = tracer.trace_all()
        tracer.cloud.update(segments)
    else:
        all_segments = []
        for obj, name, _ in objects:
            if obj is None or not object_visible[name]:
                continue
            tracer.elements.clear()
            tracer.elements.append(obj)
            tracer.rays.clear()
            tracer.rays.extend(rays_dict[name])
            segments = tracer.trace_all()
            all_segments.extend(segments)
        tracer.cloud.update(all_segments)

    plotter.update()
