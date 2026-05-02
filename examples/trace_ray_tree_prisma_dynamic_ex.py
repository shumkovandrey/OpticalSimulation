from main import *
import time


# Призма и прямоугольный экран
origin = np.array([0.0, 0.0, 0.0])
prism = BoxPrism(origin=origin, size_x=10.0, size_y=20.0, size_z=15.0, n=1.5)
screen = RectangularScreen(center=np.array([20.0, 0.0, 0.0]),
                           normal=np.array([-1.0, 0.0, 0.0]),
                           width=8.0, height=8.0)

# Статические меши
plotter.add_mesh(prism.get_mesh(), color="cyan", opacity=0.2,
                 name="prism_mesh", show_edges=True)
plotter.add_mesh(screen.get_mesh(), color="white", opacity=0.9,
                 name="screen_mesh", show_edges=True)

# Падающие лучи (параллельный пучок)
base_dir = np.array([1.0, 1.0, 0.0]) / np.linalg.norm([1, 1, 0])
start_origins = [np.array([-15.0, -1.0 + dy, 0.0]) for dy in np.linspace(-3, 3, 5)]

# Облако лучей с энергией: цветовая карта "plasma",
# прозрачность от 0.1 до 1.0, слабые лучи видны благодаря низкому min_energy_visible
ray_cloud = RayCloud(plotter, energy_color_type=2, default_color=(0.5, 1, 1, 1))

# Вращение призмы
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

    prism.rotate((0, 0, rotation_speed_z * delta))
    plotter.remove_actor("prism_mesh")
    plotter.add_mesh(prism.get_mesh(), color="cyan", opacity=0.2,
                     name="prism_mesh", show_edges=True)

    surfaces = prism.get_surfaces() + [screen]
    # Для упрощения все поверхности призмы имеют одинаковый n
    for s in prism.get_surfaces():
        s.n = 1.5

    all_segments = []
    for start_pt in start_origins:
        ray = Ray(origin=start_pt, direction=base_dir,
                  energy=1.0, current_n=1.0)
        segments = trace_ray_tree(ray, surfaces, max_depth=8, min_energy=0.005)
        all_segments.extend(segments)

    # Передаём все отрезки в облако. Цвет и прозрачность рассчитываются
    # автоматически на основе энергии каждого отрезка.
    ray_cloud.update_from_segments(all_segments)

    plotter.update()