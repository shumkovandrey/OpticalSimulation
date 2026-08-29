import numpy as np
import pyvista as pv
from main import RayTracer, RayPool, UniversalLens, Ray, SimpleMode, TreeMode

# Создаём плоттер
plotter = pv.Plotter()
plotter.set_background("#1a1a2e")
plotter.add_axes(color="white")
plotter.enable_parallel_projection()
plotter.view_isometric()

# 1. Сначала добавляем линзы
lens1 = UniversalLens(origin=(-2,0,0), R1=5, R2=-5, thickness=0.5,
                      edge_radius=1, n=1.5, refraction_range=(0, np.inf))
lens2 = UniversalLens(origin=(2,0,0), R1=5, R2=2, thickness=0.5,
                      edge_radius=1, n=1.5, rotation_degrees=(0,0,32),
                      refraction_range=(0, np.inf))

plotter.add_mesh(lens1.get_mesh(), color="cyan", opacity=0.5, smooth_shading=True)
plotter.add_mesh(lens2.get_mesh(), color="cyan", opacity=0.5, smooth_shading=True)

# 2. Затем создаём RayTracer (облако лучей добавится поверх)
pool = RayPool(initial_size=0)
rt = RayTracer(plotter, mode=TreeMode(energy_color_type=1),
               pool=pool, line_width=2.0, min_alpha=0.05, gamma=1.0)

# 3. Добавляем поверхности для трассировки
for lens in (lens1, lens2):
    for surf in lens.get_surfaces():
        rt.add_elements(surf)

# 4. Добавляем параллельные лучи
for y in np.linspace(-1, 1, 50):
    ray = Ray(origin=(-5.0, y, 0.0), direction=(1,0,0),
              energy=1.0, color="yellow", wavelength=550)
    rt.add_ray(ray)

# 5. Трассируем и обновляем облако
segments = rt.trace_all()
print(f"Сегментов: {len(segments)}")
rt.cloud.update(segments, energy_color_type=1)
plotter.reset_camera()
# 6. Рендерим и показываем
plotter.render()
plotter.show()