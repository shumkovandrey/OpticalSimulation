import pyvista as pv

plotter = pv.Plotter()

# Путь к шрифту с поддержкой кириллицы (для Windows)
font_path = '/System/Library/Fonts/Supplemental/Arial.ttf'

plotter.add_text(
    "Привет, мир!",
    font_file=font_path,
    font_size=18,
    position='upper_left'
)

plotter.add_mesh(pv.Sphere())
plotter.show()

