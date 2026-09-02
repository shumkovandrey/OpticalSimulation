from main import RayTracer, UniversalLens, plotter, DispersiveRay

tracer = RayTracer(plotter, mode='simple')

# Добавляем линзу с базовым n=1.5
lens = UniversalLens(origin=[10, 0, 0], R1=15, R2=15, thickness=4, edge_radius=5, n=1.5)
tracer.add_elements(*lens.get_surfaces())

# Генерируем 3 луча разных спектров из одной точки
wavelengths = [400.0, 550.0, 700.0]  # Фиолетовый, Зеленый, Красный

for wl in wavelengths:
    ray = DispersiveRay(
        origin=[0, 1.5, 0],            # Чуть выше оптической оси для демонстрации преломления
        direction=[1.0, 0.0, 0.0],
        wavelength=wl
    )
    tracer.add_ray(ray)

# Запуск
tracer.render()
plotter.show()

