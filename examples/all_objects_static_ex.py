from main import *


# ---------- 1. Плоское прямоугольное зеркало ----------
# Плоское зеркало, нормаль которого получается из поворота (0°, 0°, 0°) → [1,0,0]
plane_mirror = PlaneSurface(
    point=[0, -20, 0],
    rotation_degrees=(0, 0, 45),   # нормаль вдоль X
    half_sizes=(2.0, 1.5),
    # reflection_range=(0, 10000),
    # refraction_range=(0, 10000),
    absorption_range=(0, 10000),
    n_inside=1.5
)
# Визуализация: используем get_mesh
plotter.add_mesh(plane_mirror.get_mesh(), color="white", opacity=1, show_edges=True)

# Сферическое зеркало, ось направлена по X
sphere_mirror = SphereSurface(
    center=[25, 8, 0],
    radius=-15.0,
    rotation_degrees=(0, 0, 0),
    edge_radius=3.0,
    thickness=10.0,
    lens_origin=[10, 8, 0],      # вершина сферы
    lens_axis=[-1, 0, 0],        # направление от вершины к центру
    reflection_range=(0, 10000),
    refraction_range=(0, 10000),
    n_inside=1.5
)
plotter.add_mesh(sphere_mirror.get_mesh(), color="white", opacity=1, pbr=True, metallic=0.9)

# Линза, повёрнутая на 30° вокруг Z
lens = UniversalLens(
    origin=[0, 20, 0],
    rotation_degrees=(0, 0, 30),
    R1=10.0, R2=10.0,
    thickness=2.0, edge_radius=3.0,
    n=1.5,
    refraction_range=(0, 10000),
    reflection_range=(0, 10000),
)
plotter.add_mesh(lens.get_mesh(), color="cyan", opacity=0.5, smooth_shading=True)


# ---------- 5. Импортированная модель ----------
tri = trimesh.load("../Models/Prism.stl")
tri.apply_scale(3)
mesh_obj = MeshSurface(tri, n_inside=1.5, refraction_range=(0, 10000),
                       translation=[0, 60, 0], rotation_degrees=(50, 30, 90))  # ← новинка: можно сразу двигать
plotter.add_mesh(mesh_obj.get_mesh(), color="gold", opacity=0.9)

print("Mesh bounds:", mesh_obj.mesh.bounds)

# Проверка для одного луча
test_ray = Ray(origin=np.array([-15, 60, 0]), direction=np.array([1, 0, 0]))
t = mesh_obj.intersect(test_ray)
print("Intersection test:", t)
if t is not None:
    hp = test_ray.origin + test_ray.direction * t
    print("Hit point:", hp)

# ---------- Трассировка ----------
tracer = RayTracer(mode='simple', max_depth=8, min_energy=0.005, offset_distance=0.1)
for elem in [plane_mirror, sphere_mirror, *lens.get_surfaces(), mesh_obj]:
    tracer.add_element(elem)  # lens автоматически развернётся в две поверхности

def add_ray(origin, direction, color, wavelength=550):
    tracer.add_ray(Ray(origin=origin, direction=direction,
                       color=color, wavelength=wavelength,
                       energy=1.0, energy_color_type=2))

# Лучи
for y in np.linspace(-19, -21, 4):
    add_ray(np.array([-15, y, 0]), np.array([1, 0, 0]), "red", 650)
for y in np.linspace(10, 8, 4):
    add_ray(np.array([5, y, 0]), np.array([1, 0, 0]), "blue", 450)
for y in np.linspace(19, 21, 5):
    add_ray(np.array([-10, y, 0]), np.array([1, 0, 0]), "green", 550)
for y in np.linspace(59, 61, 5):
    add_ray(np.array([-15, y, 0]), np.array([1, 0, 0]), "magenta", 500)

cloud = tracer.render(plotter, line_width=2)
plotter.reset_camera()
plotter.show()