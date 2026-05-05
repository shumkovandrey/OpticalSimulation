from main import *
import time


# ---------- Объекты ----------
parabolic_mirror = AsphericSurface(
    center=[0, 10, 0],
    radius=5.0, conic_constant=-1,
    edge_radius=5.0, thickness=20.0,
    reflection_range=(0, 10000), n_inside=1.0
)
hyperbolic_lens = AsphericSurface(
    center=[0, -10, 0],
    radius=5.0, conic_constant=-5.0,
    edge_radius=4.0, thickness=20.0,
    refraction_range=(0, 10000), n_inside=1.5
)
elliptical_mirror = AsphericSurface(
    center=[0, 0, 15],
    radius=5.0, conic_constant=-0.5,
    edge_radius=4.0, thickness=20.0,
    reflection_range=(0, 10000), n_inside=1.0
)

# Акторы (прямые ссылки)
parabolic_actor = plotter.add_mesh(parabolic_mirror.get_mesh(), color="silver", pbr=True, metallic=0.9)
hyperbolic_actor = plotter.add_mesh(hyperbolic_lens.get_mesh(), color="lightblue", opacity=0.5)
# elliptical_actor = plotter.add_mesh(elliptical_mirror.get_mesh(), color="gold", pbr=True, metallic=0.7)

# Трассировщик
tracer = RayTracer(plotter, mode='tree', max_depth=5, min_energy=0.01,
                   offset_distance=0.3, energy_color_type=2, default_color="yellow")

# Анимация
state = {'paused': False}
plotter.add_key_event("space", lambda: state.update(paused=not state['paused']))

angles = np.zeros(3)
speeds = [15, 20, 25]
last_time = time.time()
plotter.show_grid(color='white')
plotter.reset_camera()
plotter.show(interactive_update=True)

while plotter.render:
    if state['paused']:
        plotter.update()
        last_time = time.time()
        continue

    dt = time.time() - last_time
    last_time = time.time()
    if dt > 0.1: dt = 0.1

    # Вращение объектов
    for i, obj in enumerate([parabolic_mirror, hyperbolic_lens, elliptical_mirror]):
        angles[i] += speeds[i] * dt
        obj.rotate((0, 0, speeds[i] * dt))

    # Обновление мешей (используем прямые ссылки на акторы)
    parabolic_actor.mapper.dataset.copy_from(parabolic_mirror.get_mesh())
    hyperbolic_actor.mapper.dataset.copy_from(hyperbolic_lens.get_mesh())
    # elliptical_actor.mapper.dataset.copy_from(elliptical_mirror.get_mesh())

    # Трассировка
    tracer.elements.clear()
    tracer.add_elements(parabolic_mirror, hyperbolic_lens)

    # Лучи
    for y_shift in np.linspace(-3, 3, 500):
        tracer.add_ray(Ray([-15, 10 + y_shift, 0], [1, 0, 0], color="red", wavelength=650, energy_color_type=2))
    for y_shift in np.linspace(-2.5, 2.5, 50):
        tracer.add_ray(Ray([-15, -10 + y_shift, 0], [1, 0, 0], color="blue", wavelength=450, energy_color_type=2))
    for z_shift in np.linspace(-3, 3, 50):
        tracer.add_ray(Ray([-15, 0, 15 + z_shift], [1, 0, 0], color="green", wavelength=550, energy_color_type=2))

    tracer.render()
    plotter.update()