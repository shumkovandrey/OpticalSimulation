from main import *  # ваш модуль со всеми классами

if __name__ == "__main__":
    # ---------- 1. Поверхность с разными спектральными диапазонами ----------
    # Отражает длины волн 380–500 нм (синий), преломляет 500–700 нм (зелёный-красный)
    # Поглощает 700–780 нм (дальний красный) – добавим для полноты.
    special_surface = PlaneSurface(
        point=np.array([0.0, 0.0, 0.0]),
        normal=np.array([1.0, 0.0, 0.0]),
        n_inside=1.5,                     # материал для преломления
        lens_origin=np.array([0.0, 0.0, 0.0]),
        lens_axis=np.array([1.0, 0.0, 0.0]),
        edge_radius=3.0,                  # круглое пятно радиусом 3
        reflection_range=(380, 700),      # отражает фиолетовый/синий
        refraction_range=(500, 700),      # преломляет зелёный-красный
        absorption_range=(700, 780)       # поглощает дальний красный
    )

    # Визуализация поверхности (круг)
    disc = pv.Disc(center=(0,0,0), normal=(1,0,0), inner=0, outer=3, c_res=64)
    plotter.add_mesh(disc, color="gray", opacity=0.5, show_edges=True)

    # ---------- 2. Лучи с разными длинами волн ----------
    elements = [special_surface]
    all_segments = []
    all_colors = []
    all_types = []

    # Три луча: 450 нм (синий), 550 нм (зелёный), 650 нм (красный), 750 нм (дальний красный)
    wavelengths = [450, 550, 650, 750]
    colors = ["blue", "green", "red", "darkred"]
    labels = ["450 nm", "550 nm", "650 nm", "750 nm"]

    # Все лучи стартуют слева, проходят через поверхность или отражаются
    i = -5
    for wl, col, label in zip(wavelengths, colors, labels):
        i+=0.5
        ray = Ray(origin=np.array([-8.0, i, 0.0]),
                  direction=normalize([1.0, 0.5, 0.0]),
                  color=col, wavelength=wl, energy_color_type=0)
        segs = trace_ray(ray, elements, mode='tree', max_depth=6, min_energy=0.01)
        all_segments.extend(segs)
        all_colors.extend([col] * len(segs))
        all_types.extend([2] * len(segs))

        # Подпись длины волны рядом с источником
        plotter.add_point_labels(
            [ray.origin + np.array([0, 0.3, 0])],
            [label],
            font_size=10, text_color="white", shape=None, show_points=False
        )

    # ---------- 3. Облако лучей ----------
    cloud = RayCloud(plotter, energy_color_type=2, default_color="yellow")
    cloud.update_from_segments(all_segments, base_colors=all_colors, energy_types=all_types)

    # Подпишем саму поверхность
    # plotter.add_point_labels(
    #     [np.array([0, 0, 0.5])],
    #     ["Special surface\nrefl 380-500\nrefr 500-700\nabs 700-780"],
    #     font_size=12, text_color="white", shape=None, show_points=False
    # )

    plotter.reset_camera()
    plotter.show()