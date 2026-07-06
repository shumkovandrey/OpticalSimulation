from main import *


# Создаем элементы сцены
lens1 = UniversalLens(origin=[-2, 0, -5], R1=1.5, R2=None, thickness=4, edge_radius=3, n=1.5)
lens2 = UniversalLens(origin=[2, 0, -5], R1=-1.5, R2=None, thickness=4, edge_radius=3, n=1.5)

# Создаем параллельные лучи
rays = []
for i in range(5):
    x = -1 + i * (2 / 4)
    direction = np.array([0, 0, 1])
    ray = Ray(origin=[x, 0, 1], direction=direction, energy=1.0, current_n=1.0, color="yellow")
    rays.append(ray)

# Добавляем элементы сцены и источники пучка лучей
elements = [*lens1.get_surfaces(), *lens2.get_surfaces()]
emitters = [BeamEmitter(origin=[-5, 0, 0], direction=[1, 0, 0], num_rays=5)]

# Инициализируем трассировщик
tracer = RayTracer(plotter, mode='tree', max_depth=20)
tracer.add_emitter(*emitters)
tracer.add_elements(*elements)

# Запускаем симуляцию и визуализацию
result = tracer.render()

# Отрисовываем траектории лучей
trajectory_list = result.get_trajectories()
visualize_scene(plotter, trajectory_list, elements, lenses=[lens1, lens2])

# Запуск интерактивного режима PyVista
plotter.reset_camera()
plotter.show()