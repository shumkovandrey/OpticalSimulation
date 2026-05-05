from PIL.ImageChops import offset

from main import *

# Стеклянная пластина (преломляющая, n=1.5)
glass_surface = PlaneSurface(
    point=[0, 0, 0],
    rotation_degrees=(0, 0, 0),
    n_inside=1.5,
    edge_radius=3.0,
    refraction_range=(0, 10000)
)



pv_plane = pv.Plane(center=[0,0,0], direction=[1,0,0], i_size=6, j_size=6)
plotter.add_mesh(pv_plane, color="white", opacity=0.3)

angles = [10, 20, 30, 40, 50, 56.3, 60, 70, 80]
colors_default = ["cyan", "yellow", "magenta", "red", "green", "blue", "pink", "purple", "brown"]
for angle in angles:
    tracer = RayTracer(plotter, mode='tree', max_depth=3, min_energy=0.001,
                   offset_distance=0.0, energy_color_type=0,
                   use_polarization_color=True)   # ← включаем окрашивание по поляризации

for angle, def_color in zip(angles, colors_default):
    rad = np.radians(angle)
    direction = np.array([np.cos(rad), -np.sin(rad), 0.0])
    origin = np.array([-5.0, 5.0*np.tan(rad), 0.0])

    # Линейная поляризация под 45° в плоскости XY:
    # E = (Es, Ep) в глобальной системе: Ep (Y) = cos(45°), Es (Z) = cos(45°)
    pol = np.array([0.0, 1.0/np.sqrt(2), 1.0/np.sqrt(2)], dtype=complex)

    ray = Ray(origin=origin, direction=direction, energy=1.0, current_n=1.0,
              color=def_color, wavelength=550, energy_color_type=0,
              polarization=pol)
    tracer.add_ray(ray)

tracer.add_elements(glass_surface)
cloud = tracer.render()
plotter.reset_camera()
plotter.show()