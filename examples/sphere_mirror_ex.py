from main import *


if __name__ == "__main__":
    plotter = pv.Plotter()
    plotter.set_background("black")
    plotter.show_grid(color="white")
    plotter.view_isometric()
    plotter.enable_parallel_projection()
    plotter.enable_terrain_style(mouse_wheel_zooms=True)

    # ------------------------------------------------------------
    # Параметры зеркала (можно менять для тестов)
    # ------------------------------------------------------------
    mirror_type = "concave"  # "concave" или "convex"

    if mirror_type == "concave":
        # Вогнутое зеркало: ось направлена от вершины к центру
        vertex = np.array([5.0, 0.0, 0.0])
        axis = normalize([-10.0, -10, 3])
        radius = 12.0
        edge_radius = 4.0
    else:
        # Выпуклое зеркало: ось от вершины к центру, радиус отрицательный
        vertex = np.array([-5.0, 0.0, 0.0])
        axis = normalize([10.0, 10, 3])
        radius = -12.0
        edge_radius = 4.0

    mirror = SphereSurface(
        center=vertex + axis * radius,
        radius=radius,
        n_inside=1.0,           # не важно для зеркала
        lens_origin=vertex,
        lens_axis=axis,
        edge_radius=edge_radius,
        thickness=20.0,         # произвольно большое значение
        is_mirror=True
    )

    # Экран для наблюдения отражённых лучей (необязательно)
    screen = Screen(point=np.array([-15.0, 0.0, 0.0]), normal=np.array([1.0, 0.0, 0.0]), size=10.0)

    # Поверхности для трассировки
    surfaces = [mirror]

    # ------------------------------------------------------------
    # Лучи: параллельный пучок, попадающий в апертуру зеркала
    # ------------------------------------------------------------
    trajectories = []
    # Направление падающих лучей: выберем так, чтобы они шли на зеркало
    if mirror_type == "concave":
        start_x = 15.0
        direction = np.array([-1.0, 0.0, 0.0])  # справа налево
    else:
        start_x = -15.0
        direction = np.array([1.0, 0.0, 0.0])   # слева направо

    y_offsets = np.linspace(-2.5, 2.5, 6)  # 6 лучей по вертикали
    for dy in y_offsets:
        start = np.array([start_x, vertex[1] + dy, 0.0])
        ray = Ray(origin=start, direction=direction)
        traj = run_simulation(ray, surfaces, max_bounces=2)
        trajectories.append(traj)

    # ------------------------------------------------------------
    # Визуализация
    # ------------------------------------------------------------
    # Зеркало
    mirror_mesh = mirror.get_mesh()
    plotter.add_mesh(mirror_mesh, color="silver", opacity=0.8, smooth_shading=True,
                     pbr=True, metallic=0, roughness=0.2)

    # Ось зеркала
    axis_line = pv.Line(vertex, vertex + axis * radius)
    plotter.add_mesh(axis_line, color="red", line_width=3)

    # Экран
    screen_mesh = pv.Plane(center=screen.point, direction=screen.normal,
                           i_size=screen.size, j_size=screen.size)
    plotter.add_mesh(screen_mesh, color="white", opacity=0.3, show_edges=True)

    # Лучи
    for i, traj in enumerate(trajectories):
        path = pv.PolyData(traj)
        path.lines = np.hstack(([len(traj)], range(len(traj))))
        # Цвет зависит от номера луча
        color = "yellow"
        plotter.add_mesh(path, color=color, line_width=2, render_lines_as_tubes=True)
        pts = pv.PolyData(traj)
        plotter.add_mesh(pts, color=color, point_size=8, render_points_as_spheres=True)

    # Подписи
    plotter.add_point_labels([vertex + np.array([0, 5, 0])],
                             [f"{mirror_type} mirror\nR={radius}"],
                             font_size=12, text_color="white", shape=None, show_points=False)

    plotter.show()

