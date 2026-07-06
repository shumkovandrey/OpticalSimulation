import pyvista as pv
import numpy as np


class InteractiveObject:
    def __init__(self, plotter, mesh):
        self.plotter = plotter
        self.actor = plotter.add_mesh(mesh, color='orange')
        self.is_moving = False
        self.axis = np.array([1, 1, 1])  # По умолчанию по всем осям

        # Регистрация событий
        self.plotter.iren.add_observer("MouseMoveEvent", self.on_mouse_move)
        self.plotter.add_key_event("g", self.start_grab)
        self.plotter.add_key_event("x", lambda: self.set_axis([1, 0, 0]))
        self.plotter.add_key_event("y", lambda: self.set_axis([0, 1, 0]))
        self.plotter.add_key_event("z", lambda: self.set_axis([0, 0, 1]))
        # Левый клик для подтверждения
        self.plotter.iren.add_observer("LeftButtonPressEvent", self.stop_grab)

    def set_axis(self, axis):
        self.axis = np.array(axis)
        print(f"Ось: {axis}")

    def start_grab(self):
        self.is_moving = True
        # Запоминаем, где был объект в момент начала
        self.start_matrix = np.copy(self.actor.user_matrix)
        # Запоминаем точку в 3D, где находится центр объекта
        self.start_obj_pos = np.array(self.actor.center)
        # Точка на экране в момент нажатия G
        self.start_mouse_pos = self.plotter.iren.get_event_position()
        print("Захват объекта...")

    def on_mouse_move(self, vtk_obj, event):
        if not self.is_moving:
            return

        # 1. Получаем текущую позицию мыши
        curr_mouse_pos = self.plotter.iren.get_event_position()

        # 2. Вычисляем смещение в мировых координатах
        # Мы используем pick_mouse_position для получения 3D точки под курсором
        # Или более надежный метод для перемещения — плоскость:
        new_pos = self.plotter.get_pick_position()

        if new_pos is not None:
            # Вычисляем вектор смещения от начальной точки
            diff = (np.array(new_pos)[:3] - self.start_obj_pos) * self.axis

            # 3. Создаем матрицу трансформации
            mat = np.eye(4)
            mat[:3, 3] = diff

            # Применяем к актору (обновление в реальном времени)
            self.actor.user_matrix = mat @ self.start_matrix

            # 4. Перерисовываем сцену немедленно
            self.plotter.render()

    def stop_grab(self, vtk_obj, event):
        if self.is_moving:
            self.is_moving = False
            print("Позиция зафиксирована")


pl = pv.Plotter()
obj = InteractiveObject(pl, pv.Sphere())
pl.add_axes()
pl.show()
