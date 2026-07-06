import numpy as np
import pyvista as pv
from main import *
from interaction import SceneEditor

plotter = pv.Plotter()
plotter.set_background("black")
plotter.show_grid(color="white")
plotter.view_isometric()
plotter.enable_parallel_projection()

# Создаём объекты
plane_mirror = PlaneSurface(
    point=[0, -3, 0],
    normal=[1, 0.5, 0],
    half_sizes=(2.0, 1.5),
    reflection_range=(0, 10000),
    n_inside=1.0
)
sphere_mirror = SphereSurface(
    center=[0, 3, 0],
    radius=-5.0,
    edge_radius=2.0,
    thickness=5.0,
    reflection_range=(0, 10000),
    n_inside=1.0
)
lens = UniversalLens(
    origin=[-4, 0, 0],
    rotation_degrees=(0, 0, 30),
    R1=10.0, R2=10.0,
    thickness=1.5, edge_radius=2.0, n=1.5,
    refraction_range=(0, 10000)
)

# Акторы
plane_actor = plotter.add_mesh(plane_mirror.get_mesh(), color="white", show_edges=True)
sphere_actor = plotter.add_mesh(sphere_mirror.get_mesh(), color="white", pbr=True, metallic=0.9)
lens_actor = plotter.add_mesh(lens.get_mesh(), color="cyan", opacity=0.5)

# Редактор
editor = SceneEditor(plotter)
editor.register(plane_mirror, plane_actor)
editor.register(sphere_mirror, sphere_actor)
editor.register(lens, lens_actor)

print("Инструкция:")
print(" - ЛКМ по объекту для выделения (станет красным)")
print(" - G: начать перемещение, затем X/Y/Z для оси, движение мышью")
print(" - R: начать вращение, затем X/Y/Z для оси, движение мышью")
print(" - Enter: применить, Escape: отменить")

plotter.show()