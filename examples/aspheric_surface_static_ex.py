from main import *


plotter.show_grid(color="white")

# 1. Парабола (отражение)
parabolic = AsphericSurface(
    center=[0, 10, 0], radius=20.0, conic_constant=-1.0,
    edge_radius=5.0, thickness=20.0,
    reflection_range=(0, 10000), n_inside=1.0
)
# 2. Гипербола (преломление)
hyperbolic = AsphericSurface(
    center=[0, -10, 0], radius=15.0, conic_constant=-2.0,
    edge_radius=4.0, thickness=20.0,
    refraction_range=(0, 10000), n_inside=1.5
)
# 3. Эллипс (отражение)
elliptical = AsphericSurface(
    center=[0, 0, 15], radius=12.0, conic_constant=-0.5,
    edge_radius=4.0, thickness=20.0,
    reflection_range=(0, 10000), n_inside=1.0
)

# Визуализация
plotter.add_mesh(parabolic.get_mesh(), color="silver", pbr=True, metallic=0.9, name="parabolic")
plotter.add_mesh(hyperbolic.get_mesh(), color="lightblue", opacity=0.5, name="hyperbolic")
plotter.add_mesh(elliptical.get_mesh(), color="gold", pbr=True, metallic=0.7, name="elliptical")

# Трассировка одного луча для каждой поверхности
tracer = RayTracer(plotter, mode='tree', max_depth=3, min_energy=0.01, offset_distance=0.3,
                   energy_color_type=2, default_color="yellow")
for elem in [parabolic, hyperbolic, elliptical]:
    tracer.add_elements(elem)

# Лучи
tracer.add_ray(Ray([-15, 10, 0], [1, 0, 0], color="red", wavelength=650))
tracer.add_ray(Ray([-15, -10, 0], [1, 0, 0], color="blue", wavelength=450))
tracer.add_ray(Ray([-15, 0, 15], [1, 0, 0], color="green", wavelength=550))

tracer.render()
plotter.reset_camera()
plotter.show()