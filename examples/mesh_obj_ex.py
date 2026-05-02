from main import *


# mesh = trimesh.load("../Models/Untitled.stl")
mesh = trimesh.creation.icosphere()
mesh.apply_scale(5)

surface = MeshSurface(mesh, is_mirror=True)

# Лучи идут вдоль оси X, y от -40 до 40 (попадают в сферу)
origins = []
for y in np.linspace(-5, 5, 50):
    origins.append(np.array([-30, y, 0]))
direction = normalize([1.0, 0, 0.0])

plotter.add_mesh(surface.get_mesh(), opacity=0.5, color="silver", pbr=True, metallic=0.9)

all_segments = []
for orig in origins:
    ray = Ray(origin=orig, direction=direction, energy=1.0, current_n=1.0)
    segs = trace_ray_tree(ray, [surface], 5)
    all_segments.extend(segs)

cloud = RayCloud(plotter, energy_color_type=2, default_color="yellow", gamma=0.3)
print("Total segments collected:", len(all_segments))
if all_segments:
    print("First segment:", all_segments[0])
cloud.update_from_segments(all_segments)

plotter.show_grid(color="white")
plotter.reset_camera()
plotter.show()