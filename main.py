"""Точка входа для ручного запуска демо-сцены.

Публичный API трассировщика живёт в отдельных модулях:
    core.rays        — Ray, DispersiveRay, WhiteRay, RayPool
    core.segments    — Segment, HitInfo
    core.dispersion  — get_dispersion_n
    surfaces.*       — PlaneSurface, SphereSurface, CylinderSurface,
                       MeshSurface, AsphericSurface, ThinLens,
                       UniversalLens, HyperbolicLens
    tracing.*        — TraceMode, SimpleMode, TreeMode, find_best_hit, split_ray
    emitters         — BeamEmitter
    visualization    — RayCloud, RayTracer
    utils.geometry   — normalize, get_tangents, calculate_rotation_matrix, calculate_radius
    utils.color      — wavelength_to_rgb, energy_to_alpha, to_rgb
    constants        — RAY_INFINITY_DISTANCE, DEFAULT_OFFSET_DISTANCE и др.

Пример использования — см. build_default_scene() ниже.
"""

import pyvista as pv


def build_default_scene():
    """Собирает демо-сцену: двояковыпуклая линза + пучок параллельных лучей.

    Возвращает (plotter, tracer) для дальнейших манипуляций.
    Запускается из 'if __name__ == "__main__"'.
    """
    from core.rays import RayPool
    from surfaces.universal_lens import UniversalLens
    from emitters import BeamEmitter
    from visualization import RayTracer

    plotter = pv.Plotter()
    plotter.set_background("black")
    plotter.view_isometric()
    plotter.enable_parallel_projection()
    plotter.enable_terrain_style(mouse_wheel_zooms=True)
    plotter.view_vector((0, 0, 1), viewup=(0, 1, 0))
    plotter.add_axes(color="white")

    pool = RayPool(initial_size=100)
    tracer = RayTracer(plotter, mode="simple", pool=pool)

    lens = UniversalLens(
        origin=(0, 0, 0), R1=5.0, R2=5.0,
        thickness=0.5, edge_radius=1.5, n=1.5,
    )
    plotter.add_mesh(lens.get_mesh(), color="cyan", opacity=0.4, name="lens")
    for surf in lens.get_surfaces():
        tracer.add_elements(surf)

    emitter = BeamEmitter(
        origin=(-5.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0),
        num_rays=7, min_offset=-1.0, max_offset=1.0,
    )
    tracer.add_emitter(emitter)

    tracer.render()
    return plotter, tracer


if __name__ == "__main__":
    plotter, tracer = build_default_scene()
    plotter.show()
