from main import *


# Параболическое зеркало радиусом 3 (сильная кривизна)
parabolic = AsphericSurface(
    center=[0,0,0], radius=3.0, conic_constant=-1.0,
    edge_radius=2.5, thickness=10.0,
    reflection_range=(0,10000), n_inside=1.0
)
# Повернём его на 45° вокруг Z, чтобы лучи падали под углом
parabolic.rotate((0,0,-210))
plotter.add_mesh(parabolic.get_mesh(), color="silver", pbr=True, name="para")

# Гиперболическая линза (преломление) радиусом 4
hyperbolic = AsphericSurface(
    center=[0,10,0], radius=4.0, conic_constant=-2.0,
    edge_radius=3.0, thickness=10.0,
    refraction_range=(0,10000), n_inside=1.5
)
hyperbolic.rotate((0,0,20))
plotter.add_mesh(hyperbolic.get_mesh(), color="cyan", opacity=0.5, name="hyp")

# Трассировщик
tracer = RayTracer(plotter, mode='tree', max_depth=5, min_energy=0.01,
                   offset_distance=0.3, energy_color_type=2, default_color="yellow")
tracer.add_elements(parabolic)
tracer.add_elements(hyperbolic)

# Параллельные лучи на параболу (под углом к оси)
for y in np.linspace(-2, 2, 5):
    tracer.add_ray(Ray([-5, y, 0], [1,0,0], color="red", wavelength=650, energy_color_type=2))
# Лучи на гиперболу
for y in np.linspace(7, 13, 5):
    tracer.add_ray(Ray([-5, y, 0], [1,0,0], color="blue", wavelength=450, energy_color_type=2))

tracer.render()
plotter.reset_camera()
plotter.show()