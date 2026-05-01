from main import *


# ---------- 1. Объектив ----------
lens_origin = np.array([0.0, 0.0, 0.0])
optical_axis = np.array([1.0, 0.0, 0.0])
lens = UniversalLens(
    origin=lens_origin,
    axis_dir=optical_axis,
    R1=10.0, R2=10.0,      # двояковыпуклая линза
    thickness=2.0,
    edge_radius=3.0,
    n=1.5
)

# ---------- 2. Диафрагма ----------
# Расположена сразу за линзой
aperture_pos = np.array([2.5, 0.0, 0.0])
aperture = Aperture(
    point=aperture_pos,
    normal=optical_axis,
    aperture_radius=1.5,   # отверстие
    outer_radius=3.0       # непрозрачная часть
)

# ---------- 3. Экран (матрица) ----------
# Приблизительно в фокальной плоскости (подбираем расстояние)
screen_distance = 11.0
screen_pos = np.array([screen_distance, 0.0, 0.0])
screen = Screen(point=screen_pos, normal=np.array([-1.0, 0.0, 0.0]), size=6.0)

# Все поверхности для трассировки
all_surfaces = lens.get_surfaces() + [aperture, screen]

# ---------- 4. Лучи от удалённого объекта ----------
trajectories = []
# Три точки объекта: верхняя (синяя), центральная (зелёная), нижняя (красная)
# Для удалённого объекта лучи от каждой точки параллельны
center_dir = np.array([1.0, 0.0, 0.0])
angle = np.radians(2.0)  # угол для крайних точек
top_dir = np.array([np.cos(angle), -np.sin(angle), 0.0])      # верхняя точка объекта -> лучи идут вниз
bottom_dir = np.array([np.cos(angle), np.sin(angle), 0.0])    # нижняя точка -> лучи идут вверх

directions = [top_dir, center_dir, bottom_dir]
colors = ['royalblue', 'lime', 'crimson']

# Параллельные пучки лучей, покрывающие апертуру объектива
y_offsets = np.linspace(-2.5, 2.5, 10)
z_offsets = np.linspace(-2.5, 2.5, 10)
start_x = -10.0

for dir_vec, col in zip(directions, colors):
    for y in y_offsets:
        for z in z_offsets:
            start = np.array([start_x, y, z])
            ray = Ray(origin=start, direction=dir_vec)
            traj = run_simulation(ray, all_surfaces, max_bounces=10)
            trajectories.append((traj, col))

# ---------- 5. Визуализация ----------
# Линза
plotter.add_mesh(lens.get_mesh(), color="cyan", opacity=0.5, smooth_shading=True)
# Диафрагма – серый диск с отверстием
plotter.add_mesh(aperture.get_mesh(), color="gray", opacity=0.8, show_edges=True)
# Экран (матрица)
plotter.add_mesh(pv.Plane(center=screen_pos, direction=screen.normal,
                          i_size=screen.size, j_size=screen.size),
                 color="white", opacity=0.9, show_edges=True)

# Лучи
for traj, col in trajectories:
    if len(traj) < 2:
        continue
    path = pv.PolyData(traj)
    path.lines = np.hstack(([len(traj)], range(len(traj))))
    plotter.add_mesh(path, color=col, line_width=2, render_lines_as_tubes=True)
    pts = pv.PolyData(traj)
    plotter.add_mesh(pts, color=col, point_size=4, render_points_as_spheres=True)

# Подписи
plotter.add_point_labels(
    [lens_origin, aperture_pos, screen_pos],
    ["Lens", "Aperture", "Sensor"],
    font_size=12, text_color="white", shape=None, show_points=False
)

plotter.reset_camera()
plotter.show()