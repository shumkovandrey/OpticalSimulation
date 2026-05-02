from main import *


# lenses = [
#     ,
#     UniversalLens(origin=[0, 40, 0], axis_dir=[1,0,0], R1=-10, R2=-10, thickness=2, edge_radius=3, n=1.5),
#     UniversalLens(origin=[0, 0, 0], axis_dir=[1,0,0], R1=None, R2=10, thickness=2, edge_radius=3, n=1.5),
#     UniversalLens(origin=[0, -40, 0], axis_dir=[1,0,0], R1=10, R2=None, thickness=2, edge_radius=3, n=1.5),
#     UniversalLens(origin=[0, -80, 0], axis_dir=[1,0,0], R1=10, R2=5, thickness=2, edge_radius=3, n=1.5)
# ]


# Создаём трассировщик
tracer = RayTracer(mode="trace_ray_tree", max_depth=6, min_energy=0.005)

lens = UniversalLens(origin=[0, 0, 0], axis_dir=[1,0,0], R1=10, R2=10, thickness=2, edge_radius=3, n=1.5)

# Добавляем объекты сцены
tracer.add_element(*lens.get_surfaces())          # BoxPrism или любая поверхность

# Генерируем лучи
for y in np.linspace(-2, 2, 5):
    origin = np.array([-15.0, y, 0.0])
    ray = Ray(origin=origin, direction=[1, 0, 0], energy=1.0, current_n=1.0, color=["red", "blue", "green", "yellow", "orange"][round(y)+2], energy_color_type=[1, 2, 0, 1, 0][round(y)+2])
    tracer.add_ray(ray, color=(1, 1, 1))

# Трассируем всё сразу
# segments, colors, types = tracer.trace_all()

# Визуализируем через RayCloud
cloud = tracer.render(plotter, line_width=2)

plotter.add_mesh(lens.get_mesh(), opacity=0.5, color="silver", pbr=True, metallic=0.9)

plotter.reset_camera()
plotter.show()
