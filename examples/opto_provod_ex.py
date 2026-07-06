from main import *


# ---------- 5. Импортированная модель ----------
tri = trimesh.load("../Models/provod.stl")
tri.apply_scale(3)
mesh_obj = MeshSurface(tri, n_inside=1.5, reflection_range=(0, 10000),
                       translation=[0, 0, 0], rotation_degrees=(0, 0, 0))
plotter.add_mesh(mesh_obj.get_mesh(), color="gold", opacity=0.1)

screen = PlaneSurface(
    point=[0.749*3, -4.0405*3, 31.0929*3], rotation_degrees=(0, 0, 0),
    half_sizes=(5, 5), n_inside=1.0,
    reflection_range=(0, 10000)
)
screen.rotate((0, 90, 0))
screen.rotate((-15, 0, 0))
plotter.add_mesh(screen.get_mesh(), color="white", opacity=1)

print("Mesh bounds:", mesh_obj.mesh.bounds)

# Проверка для одного луча
test_ray = Ray(origin=np.array([-15, 60, 0]), direction=np.array([1, 0, 0]))
t = mesh_obj.intersect(test_ray)
print("Intersection test:", t)
if t is not None:
    hp = test_ray.origin + test_ray.direction * t
    print("Hit point:", hp)

# ---------- Трассировка ----------
tracer = RayTracer(plotter=plotter, mode='simple', max_depth=240, min_energy=0.005, offset_distance=0.1)
for elem in [mesh_obj, screen]:
    tracer.add_elements(elem)  # lens автоматически развернётся в две поверхности

def add_ray(origin, direction, color, wavelength=550):
    tracer.add_ray(Ray(origin=origin, direction=direction,
                       color=color, wavelength=wavelength,
                       energy=1.0, energy_color_type=2))

# Лучи
for y in np.linspace(-0.5, 0.5, 10):
    add_ray(np.array([0, y, 0]), np.array([0, 0, 1]), ["red", "blue", "green", "purple", "orange"][int((y + 0.5) * 4)], 650)

cloud = tracer.render()
plotter.reset_camera()
plotter.show()