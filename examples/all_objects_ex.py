from main import *


# Плоское зеркало
plane_mirror = PlaneSurface(
    point=np.array([0, -20, 0]),
    normal=normalize([1, 0.5, 0]),
    n_inside=1.5,
    lens_origin=np.array([0, -20, 0]),   # ← исправлено
    lens_axis=normalize([1, 0.5, 0]),
    edge_radius=0.0,
    half_sizes=(2.0, 1.5),
    face_tangents=(np.array([0, 1, 0]), np.array([0, 0, 1])),
    reflection_range=(0, 10000),         # только отражение
    refraction_range=(0, 10000),
)

# Сферическое зеркало – отражение
sphere_mirror = SphereSurface(
    center=np.array([25, 8, 0]),         # вычислен правильно для отражения
    radius=-15.0,
    n_inside=1.5,
    lens_origin=np.array([10, 8, 0]),
    lens_axis=np.array([-1, 0, 0]),
    edge_radius=3.0,
    thickness=10.0,
    reflection_range=(0, 10000),          # ← отражение, а не поглощение
    refraction_range=(0, 10000),
)

# ---------- 3. Двояковыпуклая линза ----------
lens = UniversalLens(
    origin=np.array([0, 20, 0]),
    axis_dir=np.array([1, 0, 0]),
    R1=10.0, R2=10.0,
    thickness=2.0, edge_radius=3.0, n=1.5
)
# Задаём диапазон преломления (по умолчанию мог быть None)
lens.front.refraction_range = (0, 10000)
lens.back.refraction_range = (0, 10000)

# ---------- 4. Стеклянная призма ----------
prism = BoxPrism(origin=np.array([0, 40, 0]), size_x=10, size_y=6, size_z=6, n=1.5)
prism.rotate((0, 0, 60))
for surf in prism.get_surfaces():
    surf.refraction_range = (0, 10000)
    surf.reflection_range = (0, 10000)
    surf.absorption_range = None

# ---------- 5. Импортированная модель (икосаэдр) ----------
import trimesh
tri = trimesh.load("../Models/Lens.stl")
tri.apply_scale(3)
tri.apply_translation([0, 60, 0])
mesh_obj = MeshSurface(
    tri,
    n_inside=1.5,
    reflection_range=(0, 10000),   # отражающая модель
    refraction_range=(0, 10000),
    # absorption_range=(0, 10000),
)

# ---------- Добавление объектов на сцену ----------
# Плоское зеркало
pv_plane = pv.Plane(center=plane_mirror.point, direction=plane_mirror.normal,
                    i_size=4, j_size=3)
plotter.add_mesh(pv_plane, color="white", opacity=1, show_edges=True)

# Сферическое зеркало
plotter.add_mesh(sphere_mirror.get_mesh(), color="white", opacity=1,
                 pbr=True, metallic=0.9)

# Линза
plotter.add_mesh(lens.get_mesh(), color="cyan", opacity=0.5, smooth_shading=True)

# Призма
plotter.add_mesh(prism.get_mesh(), color="lightblue", opacity=0.3, show_edges=True)

# Модель
plotter.add_mesh(mesh_obj.get_mesh(), color="gold", opacity=0.9,
                 pbr=True, metallic=0.7)

# ---------- Трассировка лучей ----------
all_segments = []
all_colors = []
all_types = []
elements = [plane_mirror, sphere_mirror, *lens.get_surfaces(), *prism.get_surfaces(), mesh_obj]

def add_ray(origin, direction, color, wavelength=550, **trace_kwargs):
    ray = Ray(origin=origin, direction=direction, color=color, wavelength=wavelength, energy_color_type=2)
    segs = trace_ray(ray, elements, mode='simple', **trace_kwargs)
    all_segments.extend(segs)
    all_colors.extend([color] * len(segs))
    all_types.extend([2] * len(segs))

# Лучи к плоскому зеркалу (слева направо)
for y in np.linspace(-19, -21, 4):
    add_ray(np.array([-15, y, 0]), np.array([1, 0, 0]), "red", 650)

# Лучи к сферическому зеркалу (справа налево)
for y in np.linspace(10,8, 4):
    add_ray(np.array([-5, y, 0]), np.array([1, 0, 0]), "blue", 450)

# Лучи через линзу (слева направо)
for y in np.linspace(19, 21, 5):
    add_ray(np.array([-10, y, 0]), np.array([1, 0, 0]), "green", 550)

# Лучи через призму (слева направо, смещение по высоте)
for y in np.linspace(36, 45, 10):
    add_ray(np.array([-15, y, 0]), np.array([1, 0, 0]), "orange", 600, max_depth=12, min_energy=0.001, offset_distance=0.5)

# Лучи на модель (снизу вверх)
for y in np.linspace(59, 61, 5):
    add_ray(np.array([-15, y, 0]), np.array([1, 0, 0]), "magenta", 500)

# ---------- Визуализация облака лучей ----------
cloud = RayCloud(plotter, energy_color_type=2, default_color="yellow")
cloud.update_from_segments(all_segments, base_colors=all_colors,
                           energy_types=all_types)

plotter.reset_camera()
plotter.show()