import numpy as np
import pyvista as pv
from main import *

# ---------- Параметры ----------
f_obj = 5.0          # фокусное расстояние объектива
f_oc = 3.0           # фокусное расстояние окуляра
aperture_radius = 2.0

# Позиции компонентов
obj_lens_pos = np.array([0.0, 0.0, 0.0])
prism_center = np.array([f_obj + 2.0, 0.0, 0.0])   # после объектива
oc_lens_pos = np.array([prism_center[0] + 3.0, 0.0, 0.0])
screen_pos = np.array([oc_lens_pos[0] + f_oc + 1.0, 0.0, 0.0])

# ---------- Объектив (двояковыпуклая линза) ----------
lens_obj = UniversalLens(
    origin=obj_lens_pos,
    rotation_degrees=(0, 0, 0),
    R1=10.0, R2=-10.0,       # двояковыпуклая
    thickness=1.0, edge_radius=aperture_radius, n=1.5,
    refraction_range=(0, 10000)
)
plotter.add_mesh(lens_obj.get_mesh(), color="cyan", opacity=0.5)

# ---------- Призма Порро (три плоскости) ----------
# Размеры граней призмы
prism_size = 1.5
# Входная грань (преломление) – расположена слева от central point, нормаль направлена вправо
entry_plane = PlaneSurface(
    point=prism_center + np.array([-0.5, 0.0, 0.0]),
    normal=np.array([1.0, 0.0, 0.0]),
    n_inside=1.5,
    half_sizes=(prism_size, prism_size),
    refraction_range=(0, 10000)
)
# Отражающая грань (гипотенуза) – нормаль под 45° к оси X, reflection_range
refl_plane = PlaneSurface(
    point=prism_center,
    normal=np.array([-np.cos(np.radians(45)), -np.sin(np.radians(45)), 0.0]),
    n_inside=1.0,
    half_sizes=(prism_size*1.2, prism_size),
    reflection_range=(0, 10000)
)
# Выходная грань (преломление) – после отражения, лучи выходят через неё
exit_plane = PlaneSurface(
    point=prism_center + np.array([0.5, -0.5, 0.0]),
    normal=np.array([0.0, 1.0, 0.0]),
    n_inside=1.5,
    half_sizes=(prism_size, prism_size),
    refraction_range=(0, 10000)
)

# Визуализация призмы (три прямоугольника)
plotter.add_mesh(entry_plane.get_mesh(), color="lightblue", opacity=0.8, show_edges=True)
plotter.add_mesh(refl_plane.get_mesh(), color="silver", opacity=0.8, show_edges=True)
plotter.add_mesh(exit_plane.get_mesh(), color="lightblue", opacity=0.8, show_edges=True)

# ---------- Окуляр (линза) ----------
lens_oc = UniversalLens(
    origin=oc_lens_pos,
    rotation_degrees=(0, 0, 0),
    R1=8.0, R2=-8.0,
    thickness=0.8, edge_radius=1.5, n=1.5,
    refraction_range=(0, 10000)
)
plotter.add_mesh(lens_oc.get_mesh(), color="cyan", opacity=0.5)

# ---------- Экран ----------

# plotter.add_mesh(screen.get_mesh(), color="white", opacity=0.9)

# ---------- Лучи от удалённого объекта ----------
# Параллельный пучок под небольшим углом (чтобы попасть в линзу и затем в призму)
direction = np.array([1.0, 0.02, 0.0])  # почти параллельно оси, но с наклоном
direction /= np.linalg.norm(direction)

# Запускаем лучи с x = -10, распределённые по высоте в пределах апертуры объектива
tracer = RayTracer(plotter, mode='tree', max_depth=10, min_energy=0.005, offset_distance=0.1,
                   energy_color_type=2, default_color="yellow")
tracer.add_elements(*lens_obj.get_surfaces())
# tracer.add_elements(entry_plane)
# tracer.add_elements(refl_plane)
# tracer.add_elements(exit_plane)
# tracer.add_elements(lens_oc)
# tracer.add_elements(screen)

for y in np.linspace(-1.5, 1.5, 7):
    origin = np.array([-10.0, y, 0.0])
    tracer.add_ray(Ray(origin=origin, direction=direction,
                        energy=1.0, current_n=1.0,
                        color="yellow", wavelength=550,
                        energy_color_type=2))

cloud = tracer.render()
plotter.reset_camera()
plotter.show()