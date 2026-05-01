from main import *
import time


# 1. Настройка сцены
plotter = pv.Plotter()
plotter.set_background("black")
plotter.show_grid(color="white")
plotter.view_xy()
plotter.enable_terrain_style(mouse_wheel_zooms=True)

# Параметры призмы
origin = np.array([0.0, 0.0, 0.0])
size = [10.0, 20.0, 15.0]

# Настройка спектра лучей
colors = ["red", "orange", "yellow", "green", "cyan", "blue", "violet"]
n_values = np.linspace(1.50, 1.65, len(colors))
source_pos = np.array([-25.0, 5.0, 0.0])
target_point = np.array([-5.0, 0.0, 0.0])
direction = (target_point - source_pos) / np.linalg.norm(target_point - source_pos)

# Создаём акторов для лучей (один раз)
ray_actors = []
for color in colors:
    actor = plotter.add_mesh(pv.PolyData(), color=color, opacity=0.9, line_width=3)
    ray_actors.append(actor)

# Создаём призму один раз (начальный поворот – нули)
prism = BoxPrism(origin=origin, size_x=size[0], size_y=size[1], size_z=size[2], n=2)

# Создаём экран один раз
sc = Screen(point=[20, 0, 0], normal=[-1, 0, 0], size=30)

# Добавляем начальные меши призмы и экрана
prism_mesh = prism.get_mesh()
plotter.add_mesh(prism_mesh, color="cyan", opacity=0.2, name="prism_mesh", show_edges=True)

screen_mesh = pv.Plane(center=sc.point, direction=sc.normal, i_size=sc.size, j_size=sc.size)
plotter.add_mesh(screen_mesh, color="white", opacity=1, name="screen_mesh", show_edges=True)

# Углы поворота (текущие значения)
current_angle_z = 0.0
current_angle_x = 0.0
current_angle_y = 0.0

# Скорости вращения в градусах в секунду
speed_z = 30.0   # было angle_step_z = 0.5 за кадр → теперь 30°/с
speed_x = -15.0  # было -0.25
speed_y = 60.0   # было 1.0

state = {'paused': False}

def toggle_pause():
    state['paused'] = not state['paused']
    print("Пауза" if state['paused'] else "Воспроизведение")

plotter.add_key_event("space", toggle_pause)

# Запускаем интерактивное окно
plotter.show(interactive_update=True)

# Засекаем время перед циклом
last_time = time.time()

while plotter.render:
    # Вычисляем реальный интервал времени
    now = time.time()
    delta_time = now - last_time
    last_time = now

    # Защита от слишком большого скачка (например, после паузы или сворачивания окна)
    if delta_time > 0.1:
        delta_time = 0.1

    if state['paused']:
        plotter.update()
        continue

    # Обновляем углы, используя скорости и прошедшее время
    current_angle_z += speed_z * delta_time
    current_angle_x += speed_x * delta_time
    current_angle_y += speed_y * delta_time

    # Поворачиваем математическую призму (накапливаем абсолютный поворот)
    prism.rotate((speed_x * delta_time, speed_y * delta_time, speed_z * delta_time))
    # Обратите внимание: теперь rotate получает именно добавочный поворот за кадр,
    # а не абсолютный угол, как раньше. Метод rotate умножает матрицу на новый поворот,
    # поэтому он уже накапливает преобразование. Мы больше не храним абсолютные углы
    # для поворота, а просто передаём дельту.

    # Обновляем визуальный меш призмы
    plotter.remove_actor("prism_mesh")
    prism_mesh = prism.get_mesh()
    plotter.add_mesh(prism_mesh, color="cyan", opacity=0.2, name="prism_mesh", show_edges=True)

    # Список поверхностей для трассировки
    surfaces = prism.get_surfaces() + [sc]

    # Пересчитываем лучи
    for i, n_val in enumerate(n_values):
        for surf in surfaces:
            surf.n = n_val
        ray = Ray(origin=source_pos, direction=direction)
        trajectory = run_simulation(ray, surfaces, max_bounces=4)
        new_path = pv.PolyData(trajectory)
        new_path.lines = np.hstack(([len(trajectory)], range(len(trajectory))))
        ray_actors[i].mapper.dataset.copy_from(new_path)

    plotter.update()
