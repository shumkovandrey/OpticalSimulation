import time

from main import *

# --------------------------------
# Сцена с вращающейся призмой
# --------------------------------
# plotter = pv.Plotter()
# plotter.set_background("black")
# plotter.show_grid(color="white")
# plotter.view_isometric()
# plotter.enable_parallel_projection()
# plotter.enable_terrain_style(mouse_wheel_zooms=True)

# Призма
origin = np.array([0.0, 0.0, 0.0])
size = [10.0, 20.0, 15.0]  # X, Y, Z
prism = BoxPrism(origin=origin, size_x=size[0], size_y=size[1], size_z=size[2], n=1.5)

# Экран
screen = Screen(point=[20, 0, 0], normal=[-1, 0, 0], size=30)

# Источник лучей: разноцветные лучи с разными показателями преломления
colors = ["yellow"]
n_values = np.linspace(1.50, 1.65, len(colors))
source_pos = np.array([-25.0, 5.0, 0.0])
target_point = np.array([-5.0, 0.0, 0.0])
direction = (target_point - source_pos) / np.linalg.norm(target_point - source_pos)

# Акторы лучей (обновляются каждый кадр)
ray_actors = []

# Статические меши
plotter.add_mesh(prism.get_mesh(), color="cyan", opacity=0.2,
                 name="prism_mesh", show_edges=True)
screen_mesh = pv.Plane(center=screen.point, direction=screen.normal,
                       i_size=screen.size, j_size=screen.size)
plotter.add_mesh(screen_mesh, color="white", opacity=1, name="screen_mesh", show_edges=True)

# Параметры вращения
rotation_speed_z = 30.0   # градусов в секунду
state = {'paused': False}

def toggle_pause():
    state['paused'] = not state['paused']
    print("Пауза" if state['paused'] else "Воспроизведение")

plotter.add_key_event("space", toggle_pause)
plotter.reset_camera()
plotter.show(interactive_update=True)
last_time = time.time()

MAX_SEGMENTS = 100
ray_actors_pool = []  # заранее созданные акторы для линий
# В начале создаём пул пустых акторов
for _ in range(MAX_SEGMENTS):
    actor = plotter.add_mesh(pv.PolyData(), color="yellow", line_width=2,
                             render_lines_as_tubes=False)
    ray_actors_pool.append(actor)

while plotter.render:
    if state["paused"]:
        plotter.update()
        last_time = time.time()
        continue
        # prism.rotate((0, 0, 1))
        # state["paused"] = False

    now = time.time()
    delta = now - last_time
    last_time = now
    if delta > 0.1:
        delta = 0.1

    # Поворачиваем призму (инкрементально)
    prism.rotate((0, 0, rotation_speed_z * delta))

    # Обновляем визуализацию призмы
    plotter.remove_actor("prism_mesh")
    plotter.add_mesh(prism.get_mesh(), color="cyan", opacity=0.2,
                     name="prism_mesh", show_edges=True)

    # Удаляем старые лучи
    for actor in ray_actors_pool:
        actor.mapper.dataset.copy_from(pv.PolyData())  # очищаем

    # Трассировка для каждого цвета
    surfaces = prism.get_surfaces() + [screen]
    # Задаём показатели преломления для поверхностей (временно)
    # Поскольку n у всех граней призмы одинаковый, достаточно установить его один раз
    for surf in prism.get_surfaces():
        surf.n = prism.n
    screen.n = 1.0

    for i, n_val in enumerate(n_values):
        # На самом деле цветной луч должен иметь свой n для материала призмы?
        # Обычно дисперсия в призме: n зависит от цвета. У нас n_val меняется.
        # Поэтому надо задать разный n для граней призмы для каждого цвета.
        # Но так как мы трассируем все лучи параллельно, можно временно модифицировать
        # поверхности. Это упрощение: будем считать, что каждый луч видит свой n.
        for surf in prism.get_surfaces():
            surf.n = n_val

        ray = Ray(origin=source_pos, direction=direction, energy=1.0, current_n=1.0)
        segments = trace_ray_tree(ray, surfaces, max_depth=8, min_energy=0.005)

        for i, (p1, p2, energy) in enumerate(segments):
            if i >= MAX_SEGMENTS:
                break
            line = pv.Line(p1, p2)
            # opacity по логарифмической шкале
            # opacity = max(0.05, energy ** 0.3)
            opacity = energy
            # Обновляем актор
            ray_actors_pool[i].mapper.dataset.copy_from(line)
            # Цвет и прозрачность (через свойство actor.prop.opacity)
            ray_actors_pool[i].prop.opacity = opacity
            # Цвет можно задать через actor.prop.color

    plotter.update()