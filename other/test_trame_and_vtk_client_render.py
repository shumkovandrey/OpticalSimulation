import numpy as np
import pyvista as pv
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3 as vuetify
from trame.widgets import vtk as trame_vtk


# =============================================================================
# 1. СЛОЙ ДАННЫХ (Data Model)
# =============================================================================

class OpticElement:
    def __init__(self, name="Lens", position=0.0, tilt=0.0, radius=1.0):
        self.name = name
        self.position = float(position)
        self.tilt = float(tilt)
        self.radius = float(radius)


# =============================================================================
# 2. КОНТРОЛЛЕР И СЦЕНА (Controller & View)
# =============================================================================

class OpticsWebController:
    def __init__(self, server):
        self.server = server
        self.state = server.state
        self.ctrl = server.controller

        # Создаем плоттер PyVista (настройки из рабочего минимального теста)
        self.plotter = pv.Plotter(off_screen=True)
        self.plotter.add_axes()
        self.plotter.set_background("#2A2E35")

        # Абстрактные модели линз
        self.models = {
            "Lens_1": OpticElement("Входная линза", position=-3.0, tilt=0.0, radius=1.2),
            "Lens_2": OpticElement("Фокусирующая линза", position=2.0, tilt=15.0, radius=1.0)
        }
        self.actors = {}

        # Инициализация реактивного состояния Trame
        self.state.selected_id = "Lens_1"
        self.state.prop_name = self.models["Lens_1"].name
        self.state.prop_pos = self.models["Lens_1"].position
        self.state.prop_tilt = self.models["Lens_1"].tilt

        # Привязываем слайдеры к функциям Python
        self.state.change("prop_pos")(self.on_position_slider_changed)
        self.state.change("prop_tilt")(self.on_tilt_slider_changed)

    def rebuild_scene(self):
        """Перерисовка 3D объектов и луча."""
        for actor in list(self.actors.values()):
            self.plotter.remove_actor(actor)
        self.actors.clear()

        # 1. Отрисовка линз
        for key, model in self.models.items():
            lens_mesh = pv.Sphere(radius=model.radius, phi_resolution=30, theta_resolution=30)
            lens_mesh.points[:, 0] *= 0.25
            lens_mesh.rotate_y(model.tilt, inplace=True)
            lens_mesh.translate((model.position, 0, 0), inplace=True)

            is_selected = (key == self.state.selected_id)
            color = "#FFD700" if is_selected else ("#00FFFF" if key == "Lens_1" else "#85BB65")

            actor = self.plotter.add_mesh(lens_mesh, color=color, opacity=0.5, show_edges=True)
            self.actors[key] = actor

        # 2. Расчет хода луча
        ray_points = [[-8.0, 0.0, 0.0]]
        x1 = self.models["Lens_1"].position
        t1 = np.radians(self.models["Lens_1"].tilt)
        ray_points.append([x1, 0.0, 0.0])

        x2 = self.models["Lens_2"].position
        t2 = np.radians(self.models["Lens_2"].tilt)
        y2 = (x2 - x1) * np.sin(t1)
        ray_points.append([x2, y2, 0.0])

        x3 = 8.0
        y3 = y2 + (x3 - x2) * np.sin(t1 + t2)
        ray_points.append([x3, y3, 0.0])

        ray_pts_array = np.array(ray_points, dtype=float)
        ray_line = pv.MultipleLines(points=ray_pts_array)

        ray_actor = self.plotter.add_mesh(ray_line, color="#FF3333", line_width=4)
        self.actors["Ray"] = ray_actor

        # Обновляем сцену через контроллер, как в рабочем тесте
        self.ctrl.view_update()

    def select_element(self, element_id):
        self.state.selected_id = element_id
        model = self.models[element_id]
        self.state.prop_name = model.name
        self.state.prop_pos = model.position
        self.state.prop_tilt = model.tilt
        self.rebuild_scene()

    def on_position_slider_changed(self, prop_pos, **kwargs):
        model = self.models[self.state.selected_id]
        if model.position != float(prop_pos):
            model.position = float(prop_pos)
            self.rebuild_scene()

    def on_tilt_slider_changed(self, prop_tilt, **kwargs):
        model = self.models[self.state.selected_id]
        if model.tilt != float(prop_tilt):
            model.tilt = float(prop_tilt)
            self.rebuild_scene()


# =============================================================================
# 3. ВЕБ-ИНТЕРФЕЙС (Надежная сетка VRow/VCol)
# =============================================================================

server = get_server()
server.client_type = "vue3"
app_controller = OpticsWebController(server)

# Используем стандартный SinglePageLayout, в котором окно точно появлялось
with SinglePageLayout(server) as layout:
    layout.title.set_text("Оптический 3D Симулятор")

    with layout.content:
        # Создаем контейнер на всю доступную высоту экрана
        with vuetify.VContainer(fluid=True, class_="pa-0 fill-height", style="height: 100vh; overflow: hidden;"):
            # Делим рабочую область на две колонки через VRow
            with vuetify.VRow(no_gutters=True, class_="fill-height", style="height: 100%;"):
                # КОЛОНКА 1 (Слева): Панель управления (занимает 3 части из 12)
                with vuetify.VCol(cols=3, class_="pa-4 bg-grey-darken-4",
                                  style="border-right: 1px solid #444; overflow-y: auto; color: white;"):
                    vuetify.VCardTitle("Оптические Элементы", class_="text-h6 px-0")

                    with vuetify.VRow(class_="py-2 px-0 justify-space-around", no_gutters=True):
                        vuetify.VBtn("Линза 1", color="cyan", click=lambda: app_controller.select_element("Lens_1"))
                        vuetify.VBtn("Линза 2", color="green", click=lambda: app_controller.select_element("Lens_2"))

                    vuetify.VDivider(class_="my-4")
                    vuetify.VCardTitle("Свойства: {{ prop_name }}", class_="text-subtitle-1 px-0 text-cyan-lighten-2")

                    # Слайдеры управления
                    vuetify.VSlider(
                        v_model=("prop_pos",),
                        min=-6.0, max=6.0, step=0.1,
                        label="Позиция X",
                        color="cyan", class_="mt-4"
                    )
                    vuetify.VSlider(
                        v_model=("prop_tilt",),
                        min=-45.0, max=45.0, step=1.0,
                        label="Наклон Y",
                        color="amber", class_="mt-2"
                    )

                # КОЛОНКА 2 (Справа): 3D Окно (занимает 9 частей из 12)
                with vuetify.VCol(cols=9, class_="fill-height", style="height: 100%; position: relative;"):
                    html_view = trame_vtk.VtkLocalView(app_controller.plotter.render_window)

                    # Жестко связываем методы обновления, как в рабочем тесте
                    app_controller.ctrl.view_update = html_view.update
                    app_controller.ctrl.view_reset_camera = html_view.reset_camera

# Первичное построение
app_controller.rebuild_scene()
app_controller.plotter.view_isometric()

if __name__ == "__main__":
    server.start(host="127.0.0.1", port=8085)
