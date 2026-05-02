from main import *
import time


# Призма и экран
origin = np.array([0.0, 0.0, 0.0])
prism = BoxPrism(origin=origin, size_x=10.0, size_y=20.0, size_z=15.0, n=1.5)
screen = RectangularScreen(center=np.array([20.0, 0.0, 0.0]),
                           normal=np.array([-1.0, 0.0, 0.0]),
                           width=8.0, height=8.0)

plotter.add_mesh(prism.get_mesh(), color="cyan", opacity=0.2, show_edges=True)
plotter.add_mesh(screen.get_mesh(), color="white", opacity=0.9, show_edges=True)

# Пучок лучей (без разделения)
surfaces = prism.get_surfaces() + [screen]
source_pos = np.array([-15.0, -10.0, 0.0])
base_dir = normalize([1.0, 1.0, 0.0])
trajectories = []
for dy in np.linspace(2, 8, 6):
    for dz in np.linspace(-3, 3, 6):
        start = source_pos + np.array([0, dy, dz])
        ray = Ray(origin=start, direction=base_dir)
        traj = run_simulation(ray, surfaces, max_bounces=4)
        trajectories.append(traj)

# Единый актор, фиксированный цвет
cloud = RayCloud(plotter, energy_color_type=False, default_color=(0.5, 1, 1, 0.5))
cloud.update_from_trajectories(trajectories)

plotter.reset_camera()
plotter.show()
