import numpy as np
import pyvista as pv
import vtk


# 1. СЛОЙ ДАННЫХ (Data Model)
class OpticElement:
    """Абстрактное описание объекта в коде."""

    def __init__(self, name="Lens", initial_pos=(0.0, 0.0, 0.0)):
        self.name = name
        self.position = np.array(initial_pos, dtype=float)
        self.rotation_euler = np.array([0.0, 0.0, 0.0], dtype=float)

    def update_from_matrix(self, vtk_matrix):
        """ШАГ 1: Изменение параметров абстрактного объекта."""
        if vtk_matrix is None:
            return

        # Извлекаем позицию из последней колонки
        self.position[0] = vtk_matrix.GetElement(0, 3)
        self.position[1] = vtk_matrix.GetElement(1, 3)
        self.position[2] = vtk_matrix.GetElement(2, 3)

        # Извлекаем углы Эйлера
        r11 = vtk_matrix.GetElement(0, 0)
        r21 = vtk_matrix.GetElement(1, 0)
        r31 = vtk_matrix.GetElement(2, 0)
        r32 = vtk_matrix.GetElement(2, 1)
        r33 = vtk_matrix.GetElement(2, 2)

        pitch = np.arctan2(-r31, np.sqrt(r11**2 + r21**2))
        if np.abs(np.cos(pitch)) > 1e-6:
            roll = np.arctan2(r32, r33)
            yaw = np.arctan2(r21, r11)
        else:
            roll = 0.0
            yaw = np.arctan2(
                -vtk_matrix.GetElement(0, 1), vtk_matrix.GetElement(1, 1)
            )

        self.rotation_euler = np.degrees([roll, pitch, yaw])
        print(
            f"[{self.name} Model] -> Pos: {self.position.round(2)} | Rot: {self.rotation_euler.round(2)}"
        )


# 2. СЛОЙ ВИЗУАЛИЗАЦИИ И УПРАВЛЕНИЯ (View & Controller)
class UnityStyleBoxScene:

    def __init__(self):
        self.plotter = pv.Plotter()
        self.plotter.add_axes()

        self.actor_to_model = {}
        self.selected_actor = None

        # Инициализируем ОДИН независимый vtkBoxWidget на всю сцену
        self.box_widget = vtk.vtkBoxWidget()
        self.box_widget.SetInteractor(self.plotter.iren.interactor)
        self.box_widget.SetPlaceFactor(1.15)  # Размер коробки гизмо вокруг сферы

        # Настраиваем ограничения гизмо (выключаем скалирование)
        self.box_widget.ScalingEnabledOff()
        self.box_widget.RotationEnabledOn()
        self.box_widget.TranslationEnabledOn()

        # Вешаем событие изменения гизмо
        self.box_widget.AddObserver("InteractionEvent", self.on_gizmo_interact)

        # Создаем две абстрактные модели данных
        self.model_left = OpticElement("Lens_Left", initial_pos=(-3.0, 0.0, 0.0))
        self.model_right = OpticElement("Lens_Right", initial_pos=(3.0, 0.0, 0.0))

        # Базовая геометрия сферы (линзы) центрированная в (0,0,0)
        self.lens_mesh = pv.Sphere(
            radius=1.0, phi_resolution=30, theta_resolution=30
        )

        # Рендерим левый объект
        self.actor_left = self.plotter.add_mesh(
            self.lens_mesh, color="cyan", opacity=0.6, show_edges=True
        )
        self.actor_to_model[self.actor_left] = self.model_left

        # Рендерим правый объект
        self.actor_right = self.plotter.add_mesh(
            self.lens_mesh, color="lightgreen", opacity=0.6, show_edges=True
        )
        self.actor_to_model[self.actor_right] = self.model_right

        # Применяем стартовые матрицы трансформации к акторам на основе моделей
        self.sync_actor_with_model(self.actor_left, self.model_left)
        self.sync_actor_with_model(self.actor_right, self.model_right)

        # Настройка Raycasting кликов через Picker VTK (Приоритет 0.0)
        self.picker = vtk.vtkCellPicker()
        self.plotter.iren.add_observer(
            vtk.vtkCommand.LeftButtonPressEvent,
            self.on_left_button_press,
            0.0,
        )

    def sync_actor_with_model(self, actor, model):
        """Строит 4x4 матрицу на основе позиции/поворота модели и применяет к

        Актору.
        """
        transform = vtk.vtkTransform()
        transform.PostMultiply()
        transform.RotateX(model.rotation_euler[0])
        transform.RotateY(model.rotation_euler[1])
        transform.RotateZ(model.rotation_euler[2])
        transform.Translate(model.position)
        actor.SetUserTransform(transform)

    def on_left_button_press(self, interactor, event):
        """Вызывается при клике ЛКМ по сцене."""
        click_x, click_y = interactor.GetEventPosition()

        renderer = self.plotter.renderer
        self.picker.Pick(click_x, click_y, 0, renderer)
        picked_actor = self.picker.GetActor()

        if picked_actor in self.actor_to_model:
            self.select_object(picked_actor)
        else:
            self.deselect_object()

        self.plotter.render()

    def select_object(self, actor):
        """Выделяет объект и позиционирует НЕЗАВИСИМЫЙ vtkBoxWidget вокруг

        него.
        """
        if self.selected_actor == actor:
            return

        self.deselect_object()
        self.selected_actor = actor

        # Подсветка выделения
        self.selected_actor.GetProperty().SetEdgeColor(0.8, 0.1, 0.8)
        self.selected_actor.GetProperty().SetLineWidth(4)

        model = self.actor_to_model[actor]
        print(f"\n>>> ВЫБРАН ОБЪЕКТ: {model.name} <<<")

        # ВАЖНО: Размещаем коробку гизмо в пространстве по исходной геометрии меша (в 0,0,0)
        self.box_widget.PlaceWidget(self.lens_mesh.bounds)

        # А затем принудительно трансформируем саму коробку гизмо в мировые координаты объекта
        if actor.GetUserTransform():
            self.box_widget.SetTransform(actor.GetUserTransform())

        # Включаем отображение гизмо (SetProp3D НЕ ИСПОЛЬЗУЕМ, чтобы избежать багов VTK)
        self.box_widget.On()

    def deselect_object(self):
        """Полностью отключает гизмо и очищает графические маркеры."""
        if self.selected_actor:
            self.selected_actor.GetProperty().SetEdgeColor(0, 0, 0)
            self.selected_actor.GetProperty().SetLineWidth(1)
            self.selected_actor = None

        if self.box_widget.GetEnabled():
            self.box_widget.Off()
            self.plotter.render()

    def on_gizmo_interact(self, widget, event):
        """Callback, вызываемый при перетаскивании гизмо."""
        if self.selected_actor:
            # 1. Получаем сгенерированную матрицу трансформации из виджета гизмо
            transform = vtk.vtkTransform()
            widget.GetTransform(transform)
            vtk_matrix = transform.GetMatrix()

            # 2. ШАГ ИЗМЕНЕНИЯ МОДЕЛИ ДАННЫХ: Обновляем абстрактный объект в коде
            model = self.actor_to_model[self.selected_actor]
            model.update_from_matrix(vtk_matrix)

            # 3. ШАГ ОБНОВЛЕНИЯ СЦЕНЫ: Двигаем визуальную сферу вслед за матрицей
            self.selected_actor.SetUserTransform(transform)


    def run(self):
        self.plotter.view_isometric()
        self.plotter.show()


if __name__ == "__main__":
    scene = UnityStyleBoxScene()
    scene.run()
