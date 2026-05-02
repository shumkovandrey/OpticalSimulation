from main import *


USE_TRACE_TREE = True   # переключатель

# Линзы
lenses = [
    UniversalLens(origin=[0, 80, 0], axis_dir=[1,0,0], R1=10, R2=10, thickness=2, edge_radius=3, n=1.5),
    UniversalLens(origin=[0, 40, 0], axis_dir=[1,0,0], R1=-10, R2=-10, thickness=2, edge_radius=3, n=1.5),
    UniversalLens(origin=[0, 0, 0], axis_dir=[1,0,0], R1=None, R2=10, thickness=2, edge_radius=3, n=1.5),
    UniversalLens(origin=[0, -40, 0], axis_dir=[1,0,0], R1=10, R2=None, thickness=2, edge_radius=3, n=1.5),
    UniversalLens(origin=[0, -80, 0], axis_dir=[1,0,0], R1=10, R2=5, thickness=2, edge_radius=3, n=1.5)
]

all_surfaces = []
for lens in lenses:
    all_surfaces.extend(lens.get_surfaces())

# Цвета для линз в RGB
lens_colors_rgb = [
    (0.9, 0.2, 0.2),    # красноватый
    (0.2, 0.6, 1.0),    # голубой
    (0.2, 0.9, 0.3),    # зелёный
    (1.0, 0.7, 0.1),    # оранжевый
    (0.8, 0.3, 1.0)     # фиолетовый
]

direction = np.array([1.0, 0.0, 0.0])
y_offsets = np.linspace(-2.5, 2.5, 10)

all_segments = []
base_colors_list = []

if USE_TRACE_TREE:
    for lens_idx, lens in enumerate(lenses):
        for dy in y_offsets:
            start = np.array([-15.0, lens.origin[1] + dy, 0.0])
            ray = Ray(origin=start, direction=direction, energy=1.0, current_n=1.0)
            segs = trace_ray_tree(ray, all_surfaces, max_depth=6, min_energy=0.01)
            all_segments.extend(segs)
            base_colors_list.extend([lens_colors_rgb[lens_idx]] * len(segs))
else:
    trajectories = []
    colors = []
    for lens_idx, lens in enumerate(lenses):
        for dy in y_offsets:
            start = np.array([-15.0, lens.origin[1] + dy, 0.0])
            ray = Ray(origin=start, direction=direction)
            traj = run_simulation(ray, all_surfaces, max_bounces=5)
            trajectories.append(traj)
            colors.append(lens_colors_rgb[lens_idx])

# Визуализация линз
for lens in lenses:
    plotter.add_mesh(lens.get_mesh(), color="cyan", opacity=0.5, smooth_shading=True)

# Создаём облако лучей
if USE_TRACE_TREE:
    # energy_color_type=2: плавное затухание
    cloud = RayCloud(plotter, energy_color_type=2, default_color="white",
                     min_alpha=0.05, gamma=0.3)
    cloud.update_from_segments(all_segments, base_colors=base_colors_list)
else:
    cloud = RayCloud(plotter, energy_color_type=0)   # 0 – без энергии
    cloud.update_from_trajectories(trajectories, colors=colors)

# Подписи
labels = ["Biconvex", "Biconcave", "Plano-convex", "Convex-plano", "Meniscus"]
label_pts = [np.array([0, 80, 4]), np.array([0, 40, 4]), np.array([0, 0, 4]),
             np.array([0, -40, 4]), np.array([0, -80, 4])]
plotter.add_point_labels(label_pts, labels, font_size=10, text_color="white",
                         shape=None, show_points=False)

plotter.reset_camera()
plotter.show()