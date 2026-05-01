from main import *


import time
import numpy as np
import pyvista as pv

# Все ваши классы и функции (Ray, PlaneSurface, RectangularMirror, get_tangents, run_simulation и т.д.) должны быть определены выше.

if __name__ == "__main__":

    # Параметры зеркала
    mirror_width = 4.0
    mirror_height = 2.0
    mirror_center = np.array([0.0, 0.0, 0.0])

    # Начальный угол (нормаль вдоль оси X)
    initial_angle = 0.0  # нормаль = (1, 0, 0)

    # Скорость вращения (градусы в секунду)
    rotation_speed = 45.0

    # Создаём первое зеркало (будем обновлять его в цикле)
    current_angle = initial_angle
    normal = np.array([np.cos(np.radians(current_angle)), np.sin(np.radians(current_angle)), 0.0])
    mirror = RectangularMirror(
        center=mirror_center,
        normal=normal,
        width=mirror_width,
        height=mirror_height
    )

    # Добавляем начальную визуализацию зеркала
    plotter.add_mesh(mirror.get_mesh(), color="silver", opacity=0.7, show_edges=True,
                     pbr=True, metallic=0.8, roughness=0.2, name="mirror_mesh")
    # Линия нормали (вспомогательная)
    normal_line = pv.Line(mirror_center, mirror_center + normal * 2.0)
    plotter.add_mesh(normal_line, color="cyan", line_width=3, name="normal_line")

    # Лучи: параллельный пучок, идущий слева направо
    ray_direction = np.array([1.0, 0.0, 0.0])
    # Стартовые точки лучей: варьируем Y и Z, чтобы покрыть область вокруг зеркала
    ray_origins = []
    y_offsets = np.linspace(-3.0, 3.0, 13)  # часть попадает в зеркало, часть мимо
    z_offsets = np.linspace(-1.5, 1.5, 5)
    start_x = -8.0
    for y in y_offsets:
        for z in z_offsets:
            ray_origins.append(np.array([start_x, y, z]))

    # Создаём постоянные объекты Ray (источники)
    rays = []
    for org in ray_origins:
        rays.append(Ray(origin=org, direction=ray_direction))

    # Актёры для лучей (обновляются каждый кадр)
    ray_actors = []
    for _ in rays:
        actor = plotter.add_mesh(pv.PolyData(), color="yellow", line_width=2, render_lines_as_tubes=True)
        ray_actors.append(actor)

    # Точки пересечения (можно отдельными актёрами, но проще обновлять в основном акторе)
    # Будем обновлять актор луча и в нём же отображать точки (уже есть в траектории).

    # Состояние паузы
    state = {'paused': False}

    def toggle_pause():
        state['paused'] = not state['paused']
        print("Пауза" if state['paused'] else "Воспроизведение")

    plotter.add_key_event("space", toggle_pause)

    # Запускаем интерактивный режим
    plotter.reset_camera()
    plotter.show(interactive_update=True)

    # Время для deltaTime
    last_time = time.time()

    while plotter.render:
        # Обработка паузы
        if state['paused']:
            plotter.update()
            # Чтобы после паузы не было скачка времени, сбрасываем last_time
            last_time = time.time()
            continue

        # Вычисляем прошедшее время
        now = time.time()
        delta_time = now - last_time
        last_time = now
        if delta_time > 0.1:  # защита от выбросов
            delta_time = 0.1

        # Обновляем угол
        current_angle += rotation_speed * delta_time
        # Нормализуем угол, чтобы не накапливать огромные значения
        current_angle %= 360.0

        # Новая нормаль
        normal = np.array([np.cos(np.radians(current_angle)), np.sin(np.radians(current_angle)), 0.0])

        # Создаём новое зеркало с обновлённой ориентацией
        mirror = RectangularMirror(
            center=mirror_center,
            normal=normal,
            width=mirror_width,
            height=mirror_height
        )

        # Обновляем визуализацию зеркала
        plotter.remove_actor("mirror_mesh")
        plotter.add_mesh(mirror.get_mesh(), color="silver", opacity=0.7, show_edges=True,
                         pbr=True, metallic=0.8, roughness=0.2, name="mirror_mesh")

        # Обновляем линию нормали
        plotter.remove_actor("normal_line")
        normal_line = pv.Line(mirror_center, mirror_center + normal * 2.0)
        plotter.add_mesh(normal_line, color="cyan", line_width=3, name="normal_line")

        # Трассировка лучей через текущее зеркало (только оно)
        surfaces = [mirror]
        for i, ray in enumerate(rays):
            trajectory = run_simulation(ray, surfaces, max_bounces=2)
            # Создаём PolyData из траектории
            if len(trajectory) >= 2:
                path = pv.PolyData(np.array(trajectory))
                path.lines = np.hstack(([len(trajectory)], range(len(trajectory))))
                # Обновляем актор
                ray_actors[i].mapper.dataset.copy_from(path)
            else:
                # Если луч вообще не встретил поверхностей, показываем только начальную точку
                # (можно оставить пустым или показать точку)
                path = pv.PolyData(np.array([ray.origin]))
                ray_actors[i].mapper.dataset.copy_from(path)

        plotter.update()