from main import *


origin = np.array([0.0, 0.0, 0.0])
prism = BoxPrism(origin=origin, size_x=10.0, size_y=20.0, size_z=15.0, n=1.5)

source_pos = np.array([-15.0, -1.0, 0.0])
direction = normalize([1.0, 1.0, 0.5])   # диагональный луч
direction = direction / np.linalg.norm(direction)

ray = Ray(origin=source_pos, direction=direction, energy=1.0, current_n=1.0)
surfaces = prism.get_surfaces()

segments = trace_ray_tree(ray, surfaces, max_depth=100, min_energy=1e-10)

print(f"Получено {len(segments)} отрезков")
for idx, (p1, p2, energy) in enumerate(segments):
    print(f"Segment {idx}: {p1} -> {p2}, energy={energy:.3f}")

# Визуализация
plotter.add_mesh(prism.get_mesh(), color="cyan", opacity=0.2, show_edges=True)
plotter.add_mesh(pv.PolyData(source_pos), color='purple', point_size=10.0, render_points_as_spheres=True)
print(segments)
for p1, p2, energy in segments:
    line = pv.Line(p1, p2)
    opacity = max(0.05, energy ** 0.3)
    # opacity = energy
    # Цвет: жёлтый для преломлённых, голубой для отражённых (можно автоматически по depth, но в segments пока depth не передаётся)
    plotter.add_mesh(line, color="yellow", opacity=opacity, line_width=2, render_lines_as_tubes=True)

plotter.reset_camera()
plotter.show()