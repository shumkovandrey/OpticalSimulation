import random

from main import *

import time


# Объекты сцены
origin = np.array([0.0, 0.0, 0.0])
prism = BoxPrism(origin=origin, size_x=10.0, size_y=20.0, size_z=15.0, n=1.5)
screen = RectangularScreen(center=np.array([20.0, 0.0, 0.0]),
                           normal=np.array([-1.0, 0.0, 0.0]),
                           width=80.0, height=80.0)

# Постоянные актёры (создаём один раз)
prism_actor = plotter.add_mesh(prism.get_mesh(), color="cyan", opacity=0.2,
                               name="prism_mesh", show_edges=True)
screen_actor = plotter.add_mesh(screen.get_mesh(), color="white", opacity=0.9,
                                name="screen_mesh", show_edges=True)

# Пучок лучей
start_origins = [np.array([-15.0, -1.0 + dy, 0.0]) for dy in np.linspace(-3, 3, 50)]
direction = np.array([1.0, 0, 0.0]) / np.linalg.norm([1,1,0])

# Облако лучей (единый актор, однократно)
cloud = RayCloud(plotter, energy_color_type=2, default_color="yellow", line_width=2)

# Вращение
rotation_speed_z = 20.0
state = {'paused': False}

def toggle_pause():
    state['paused'] = not state['paused']
    print("Пауза" if state['paused'] else "Воспроизведение")

plotter.add_key_event("space", toggle_pause)
plotter.reset_camera()
plotter.show(interactive_update=True)

last_time = time.time()

while plotter.render:
    if state['paused']:
        plotter.update()
        last_time = time.time()
        continue

    now = time.time()
    delta = now - last_time
    last_time = now
    if delta > 0.1:
        delta = 0.1

    # Поворачиваем призму
    prism.rotate((0, 0, rotation_speed_z * delta))
    # Обновляем геометрию актёра призмы (не пересоздавая)
    prism_actor.mapper.dataset.copy_from(prism.get_mesh())

    # Трассировщик на этот кадр
    tracer = RayTracer(mode="trace_ray_tree", max_depth=10, min_energy=0.01)
    tracer.add_elements(screen)
    for s in prism.get_surfaces():
        s.n = 1.5
        tracer.add_elements(s)

    for start in start_origins:
        ray = Ray(origin=start, direction=direction,
                  energy=1.0, current_n=1.0,
                  color=(1, 0.5, 0), energy_color_type=1)
        tracer.add_ray(ray)

    # Получаем сегменты и обновляем облако (без удаления актора)
    segments, colors, types = tracer.trace_all()
    cloud.update_from_segments(segments, base_colors=colors, energy_types=types)

    plotter.update()