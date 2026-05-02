from main import *   # ваш модуль со всеми классами

if __name__ == "__main__":

    # Плоское зеркало в начале координат, нормаль по X, размер 2x1.5
    plane_mirror = PlaneSurface(
        point=np.array([0.0, 0.0, 0.0]),
        normal=normalize([1.0, 1.0, 0.0]),
        n_inside=1.5,
        lens_origin=np.array([0.0, 0.0, 0.0]),   # совпадает с point!
        lens_axis=np.array([1.0, 0.0, 0.0]),
        edge_radius=0.0,
        half_sizes=(2.0, 1.5),                  # ширина 4 (Y), высота 3 (Z)
        face_tangents=(np.array([0,1,0]), np.array([0,0,1])),
        reflection_range=(0, 10000),             # только отражение
        refraction_range=(0, np.inf)
    )

    # Визуализация зеркала
    pv_plane = pv.Plane(center=plane_mirror.point, direction=plane_mirror.normal,
                        i_size=4, j_size=3)
    plotter.add_mesh(pv_plane, color="blue", opacity=1, show_edges=True)

    # Трассируем один луч
    ray = Ray(origin=np.array([-5.0, 0.0, 0.0]),
              direction=np.array([1.0, 0.0, 0.0]),
              color="red", wavelength=650, energy_color_type=2)

    # Включаем отладку (временно модифицируем _trace_recursive, см. ниже)
    segments = trace_ray(ray, [plane_mirror], mode='tree', max_depth=3, min_energy=0.01, offset_distance=0.1)

    # Облако
    cloud = RayCloud(plotter, energy_color_type=2, default_color="yellow")
    if segments:
        cloud.update_from_segments(segments, base_colors=["red"]*len(segments), energy_types=[2]*len(segments))
    else:
        print("Луч не пересёкся с зеркалом!")

    plotter.show_grid(color="white")
    plotter.reset_camera()
    plotter.show()