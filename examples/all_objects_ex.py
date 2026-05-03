from main import *


# ---------- Создание объектов ----------
plane_mirror = PlaneSurface(
    point=np.array([0, -20, 0]),
    normal=normalize([1, 0.5, 0]),
    n_inside=1.5,
    lens_origin=np.array([0, -20, 0]),
    lens_axis=normalize([1, 0.5, 0]),
    edge_radius=0.0,
    half_sizes=(2.0, 1.5),
    face_tangents=(np.array([0, 1, 0]), np.array([0, 0, 1])),
    reflection_range=(0, 10000)
)

sphere_mirror = SphereSurface(
    center=np.array([25, 8, 0]),
    radius=-15.0,
    n_inside=1.5,
    lens_origin=np.array([10, 8, 0]),
    lens_axis=np.array([-1, 0, 0]),
    edge_radius=3.0,
    thickness=10.0,
    reflection_range=(0, 10000)
)

lens = UniversalLens(
    origin=np.array([0, 20, 0]),
    axis_dir=np.array([1, 0, 0]),
    R1=10.0, R2=10.0,
    thickness=2.0, edge_radius=3.0, n=1.5
)

prism = BoxPrism(origin=np.array([0, 40, 0]), size_x=10, size_y=6, size_z=6, n=1.5)
prism.rotate((0, 0, 30))
for surf in prism.get_surfaces():
    surf.refraction_range = (0, 10000)
    surf.reflection_range = None
    surf.absorption_range = None

# Загружаем модель
import trimesh
tri = trimesh.load("../Models/Prism.stl")
tri.apply_scale(3)
tri.apply_translation([0, 60, 0])
mesh_obj = MeshSurface(tri, n_inside=1.5, reflection_range=(0, 10000), refraction_range=(0, 10000))

# ---------- Визуализация объектов ----------
plotter.add_mesh(pv.Plane(center=plane_mirror.point, direction=plane_mirror.normal,
                          i_size=4, j_size=3), color="white", opacity=1, show_edges=True)
plotter.add_mesh(sphere_mirror.get_mesh(), color="white", opacity=1, pbr=True, metallic=0.9)
plotter.add_mesh(lens.get_mesh(), color="cyan", opacity=0.5, smooth_shading=True)
plotter.add_mesh(prism.get_mesh(), color="lightblue", opacity=0.3, show_edges=True)
plotter.add_mesh(mesh_obj.get_mesh(), color="gold", opacity=0.9, pbr=True, metallic=0.7)

# ---------- Создаём один трейсер ----------
# Используем mode='tree', чтобы видеть энергию/прозрачность и отражения.
tracer = RayTracer(mode='simple', max_depth=8, min_energy=0.005, offset_distance=0.1)

# Добавляем все элементы
for elem in [plane_mirror, sphere_mirror, *lens.get_surfaces(), *prism.get_surfaces(), mesh_obj]:
    tracer.add_element(elem)

# Добавляем лучи
def add_ray(origin, direction, color, wavelength=550):
    tracer.add_ray(Ray(origin=origin, direction=direction,
                       color=color, wavelength=wavelength,
                       energy=1, energy_color_type=2))

# Красные лучи к плоскому зеркалу
for y in np.linspace(-19, -21, 4):
    add_ray(np.array([-15, y, 0]), np.array([1, 0, 0]), "red", 650)

# Синие лучи к сферическому зеркалу
for y in np.linspace(10, 8, 4):
    add_ray(np.array([-5, y, 0]), np.array([1, 0, 0]), "blue", 450)

# Зелёные лучи через линзу
for y in np.linspace(19, 21, 5):
    add_ray(np.array([-10, y, 0]), np.array([1, 0, 0]), "green", 550)

# Оранжевые лучи через призму
for y in np.linspace(36, 45, 10):
    add_ray(np.array([-15, y, 0]), np.array([1, 0, 0]), "orange", 600)

# Фиолетовые лучи на модель
for y in np.linspace(59, 61, 5):
    add_ray(np.array([-15, y, 0]), np.array([1, 0, 0]), "magenta", 500)

# ---------- Автоматическая трассировка и визуализация ----------
cloud = tracer.render(plotter, line_width=2)

plotter.reset_camera()
plotter.show()