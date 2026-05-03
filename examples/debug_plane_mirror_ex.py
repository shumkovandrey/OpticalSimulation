from main import *   # ваш модуль

if __name__ == "__main__":
    # Зеркало – только reflection_range
    mirror = PlaneSurface(
        point=np.array([0.0, 0.0, 0.0]),
        normal=normalize([1.0, 1.0, 0.0]),
        n_inside=1.0,
        lens_origin=np.array([0.0, 0.0, 0.0]),
        lens_axis=normalize([1.0, 1.0, 0.0]),
        edge_radius=0.0,
        half_sizes=(3.0, 3.0),
        face_tangents=(np.array([0,1,0]), np.array([0,0,1])),
        reflection_range=(0, 10000),   # только ОТРАЖЕНИЕ
        refraction_range=None,
        absorption_range=None
    )

    # Визуализация
    plotter.add_mesh(pv.Plane(center=(0,0,0), direction=normalize([1,1,0]), i_size=6, j_size=6),
                     color="gray", opacity=0.5)

    # Трассировщик в простом режиме
    tracer = RayTracer(mode='simple', max_depth=3, offset_distance=0.5)
    tracer.add_element(mirror)
    tracer.add_ray(Ray(origin=[-5, 0.5, 0], direction=[1, 0, 0],
                       color="red", wavelength=650, energy_color_type=0))

    # Рендер
    cloud = tracer.render(plotter, line_width=2)
    plotter.reset_camera()
    plotter.show()