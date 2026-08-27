import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")

import numpy as np
import pyvista as pv
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3 as vuetify
from pyvista.trame.ui import plotter_ui   # серверный рендеринг
from scipy.spatial.transform import Rotation as R

from main import (
    RayTracer, RayPool, UniversalLens, BeamEmitter, RAY_INFINITY_DISTANCE, Ray
)

class OpticsAppController:
    def __init__(self, server):
        self.server = server
        self.state = server.state
        self.ctrl = server.controller
        self.manual_rays = []

        self.plotter = pv.Plotter(off_screen=True)
        self.plotter.set_background("#1a1a2e")
        self.plotter.add_axes(color="white")
        self.plotter.enable_parallel_projection()
        self.plotter.view_isometric()

        self.pool = RayPool(initial_size=0)
        self.scene_objects = []
        self.object_counter = 0
        self.initializing = True
        self._updating = False

        self.ray_tracer = None

        # Состояние
        self.state.selected_object_id = None
        self.state.selected_object_type = None
        self.state.pick_coords = None
        self.state.trace_mode = "tree"
        self.state.energy_color_type = 2

        # Параметры выбранного объекта
        self.state.param_pos_x = 0.0
        self.state.param_pos_y = 0.0
        self.state.param_pos_z = 0.0
        self.state.param_rot_x = 0.0
        self.state.param_rot_y = 0.0
        self.state.param_rot_z = 0.0
        self.state.param_n = 1.5
        self.state.param_R1 = 5.0
        self.state.param_R2 = -5.0
        self.state.param_thickness = 0.5
        self.state.param_edge_radius = 1.0
        self.state.param_num_rays = 5
        self.state.param_min_offset = -0.5
        self.state.param_max_offset = 0.5
        self.state.param_wavelength = 550.0

        # Обработчики
        self.state.change("selected_object_id")(self.on_object_selected)
        self.state.change("pick_coords")(self.on_pick_coords)
        self.state.change("trace_mode")(self.on_trace_mode_changed)

        self.create_initial_objects()
        self.initializing = False



    # ---------------------------------------------------------------
    # Создание / удаление
    # ---------------------------------------------------------------
    def create_initial_objects(self):
        self.add_object("lens", "Линза 1", {
            "origin": (-2.0, 0.0, 0.0),
            "rotation": (0, 0, 0),
            "R1": 2, "R2": 3, "thickness": 0.5,
            "edge_radius": 1.0, "n": 1.5,
            "reflection_range": None,
            "refraction_range": (0, np.inf),
            "absorption_range": None
        })
        self.add_object("lens", "Линза 2", {
            "origin": (2.0, 0.0, 0.0),
            "rotation": (0, 0, 0),
            "R1": -3.0, "R2": 2.0, "thickness": 0.5,
            "edge_radius": 1.0, "n": 1.5,
            "reflection_range": None,
            "refraction_range": (0, np.inf),
            "absorption_range": None
        })
        # self.add_object("emitter", "Источник", {
        #     "origin": (-5.0, 0.0, 0.0),
        #     "rotation": (0, 0, 0),
        #     "num_rays": 5, "min_offset": -0.5, "max_offset": 0.5,
        #     "wavelength": 550.0, "color": "yellow",
        #     "energy": 1.0, "current_n": 1.0
        # })
        for y in np.linspace(-0.5, 0.5, 5):
            ray = Ray(origin=(-5.0, y, 0.0), direction=(1, 0, 0),
                      energy=1.0, color="yellow", wavelength=550)
            self.manual_rays.append(ray)
        if self.scene_objects:
            self.state.selected_object_id = self.scene_objects[0]["id"]

    def add_object(self, obj_type, name, params):
        self.object_counter += 1
        obj_id = f"obj_{self.object_counter}"
        instance = self._create_instance(obj_type, params)
        self.scene_objects.append({
            "id": obj_id,
            "type": obj_type,
            "name": name,
            "params": params,
            "instance": instance
        })
        if not self.initializing:
            self.state.selected_object_id = obj_id
            self.update_scene()
        return obj_id

    def add_lens_click(self):
        self.add_object("lens", f"Линза {len(self.scene_objects)+1}", {
            "origin": (0, 0, 0),
            "rotation": (0, 0, 0),
            "R1": 5.0, "R2": 5.0, "thickness": 0.5,
            "edge_radius": 1.0, "n": 1.5,
            "reflection_range": None,
            "refraction_range": (0, np.inf),
            "absorption_range": None
        })

    def add_emitter_click(self):
        self.add_object("emitter", f"Источник {len(self.scene_objects)+1}", {
            "origin": (0, 0, 0),
            "rotation": (0, 0, 0),
            "num_rays": 5, "min_offset": -0.5, "max_offset": 0.5,
            "wavelength": 550.0, "color": "yellow",
            "energy": 1.0, "current_n": 1.0
        })

    def remove_object(self, obj_id):
        self.scene_objects = [o for o in self.scene_objects if o["id"] != obj_id]
        if self.state.selected_object_id == obj_id:
            self.state.selected_object_id = self.scene_objects[0]["id"] if self.scene_objects else None
        self.update_scene()

    def _create_instance(self, obj_type, params):
        if obj_type == "lens":
            return UniversalLens(
                origin=params.get("origin", (0,0,0)),
                rotation_degrees=params.get("rotation", (0,0,0)),
                R1=params.get("R1"),
                R2=params.get("R2"),
                thickness=params.get("thickness", 0.5),
                edge_radius=params.get("edge_radius", 1.0),
                n=params.get("n", 1.5),
                reflection_range=params.get("reflection_range"),
                refraction_range=params.get("refraction_range", (0, np.inf)),
                absorption_range=params.get("absorption_range")
            )
        elif obj_type == "emitter":
            rot = params.get("rotation", (0,0,0))
            direction = R.from_euler('xyz', rot, degrees=True).apply([1,0,0])
            return BeamEmitter(
                origin=params.get("origin", (0,0,0)),
                direction=direction,
                num_rays=params.get("num_rays", 5),
                min_offset=params.get("min_offset", -0.5),
                max_offset=params.get("max_offset", 0.5),
                color=params.get("color", "yellow"),
                wavelength=params.get("wavelength", 550),
                energy=params.get("energy", 1.0),
                current_n=params.get("current_n", 1.0),
                pool=self.pool
            )
        else:
            raise ValueError(f"Неизвестный тип: {obj_type}")

    # ---------------------------------------------------------------
    # Обновление сцены (серверный рендеринг)
    # ---------------------------------------------------------------
    def update_scene(self):
        if self._updating:
            return
        self._updating = True
        try:
            # 1. Полная очистка
            self.plotter.clear()
            self.plotter.add_axes(color="white")
            self.plotter.set_background("#1a1a2e")
            self.plotter.enable_parallel_projection()
            self.plotter.view_isometric()

            # 2. Добавляем меши объектов (линзы, стрелки источников)
            for obj_entry in self.scene_objects:
                instance = obj_entry["instance"]
                if isinstance(instance, UniversalLens):
                    self.plotter.add_mesh(
                        instance.get_mesh(),
                        color="cyan", opacity=0.5, smooth_shading=True,
                        pickable=True, name=obj_entry["id"]
                    )
                elif isinstance(instance, BeamEmitter):
                    # Только стрелку, без генерации лучей
                    self.plotter.add_mesh(
                        instance.get_mesh(),
                        color="green", pickable=True, name=obj_entry["id"]
                    )
                else:
                    self.plotter.add_mesh(
                        instance.get_mesh(),
                        color="gray", opacity=0.7, pickable=True,
                        name=obj_entry["id"]
                    )

            # 3. Создаём RayTracer (облако НЕ ИСПОЛЬЗУЕМ)
            self.ray_tracer = RayTracer(
                self.plotter,
                mode="simple",  # SimpleMode для предсказуемости
                pool=self.pool,
                line_width=2.0,
                min_alpha=0.05,
                gamma=0.3
            )
            self.ray_tracer.mode.energy_color_type = 1  # энергия = непрозрачность

            # 4. Добавляем поверхности линз для трассировки
            for obj_entry in self.scene_objects:
                instance = obj_entry["instance"]
                if isinstance(instance, UniversalLens):
                    for surf in instance.get_surfaces():
                        self.ray_tracer.add_elements(surf)

            # 5. Добавляем лучи напрямую (без эмиттера!)
            # Здесь предполагается, что self.manual_rays заполнен в create_initial_objects
            for ray in self.manual_rays:
                self.ray_tracer.add_ray(ray)

            # 6. Трассируем
            segments = self.ray_tracer.trace_all()

            # 7. Рисуем сегменты правильно и быстро без ручной сборки lines
            if segments:
                # Собираем массив пар точек (N, 2, 3)
                segment_pairs = np.array([[seg.start, seg.end] for seg in segments], dtype=np.float32)

                # Изменяем форму в плоский массив точек (2*N, 3)
                all_points = segment_pairs.reshape(-1, 3)

                # Создаем массив ячеек VTK: для каждого отрезка [2, индекс_старта, индекс_конца]
                n_segments = len(segments)
                connectivity = np.empty((n_segments, 3), dtype=np.int64)
                connectivity[:, 0] = 2
                connectivity[:, 1] = np.arange(0, 2 * n_segments, 2)
                connectivity[:, 2] = np.arange(1, 2 * n_segments, 2)

                # Уплощаем массив ячеек, как требует PyVista
                lines_vtk = connectivity.ravel()

                # Строим чистый меш
                ray_mesh = pv.PolyData(all_points, lines=lines_vtk)

                # Добавляем на сцену с уникальным именем (исключает дублирование акторов)
                self.plotter.add_mesh(
                    ray_mesh,
                    color="yellow",
                    line_width=2,
                    render_lines_as_tubes=False,
                    name="traced_rays_geometry"  # Имя предотвращает утечки памяти
                )

            # 8. Принудительный рендер
            self.plotter.render()

            # 9. Отправка кадра
            if hasattr(self.ctrl, 'view_update'):
                self.ctrl.view_update()

        finally:
            self._updating = False

    # ---------------------------------------------------------------
    # Обработчики
    # ---------------------------------------------------------------
    def on_object_selected(self, *args, **kwargs):
        obj_id = args[0] if args else None
        if not obj_id:
            return
        obj_entry = self._find_object(obj_id)
        if obj_entry:
            self._load_params_to_state(obj_entry)

    def _find_object(self, obj_id):
        for o in self.scene_objects:
            if o["id"] == obj_id:
                return o
        return None

    def _load_params_to_state(self, obj_entry):
        p = obj_entry["params"]
        self.state.selected_object_type = obj_entry["type"]
        self.state.param_pos_x = float(p["origin"][0])
        self.state.param_pos_y = float(p["origin"][1])
        self.state.param_pos_z = float(p["origin"][2])
        self.state.param_rot_x = float(p["rotation"][0])
        self.state.param_rot_y = float(p["rotation"][1])
        self.state.param_rot_z = float(p["rotation"][2])
        if obj_entry["type"] == "lens":
            self.state.param_n = float(p.get("n", 1.5))
            self.state.param_R1 = float(p.get("R1", 5.0))
            self.state.param_R2 = float(p.get("R2", -5.0))
            self.state.param_thickness = float(p.get("thickness", 0.5))
            self.state.param_edge_radius = float(p.get("edge_radius", 1.0))
        elif obj_entry["type"] == "emitter":
            self.state.param_num_rays = int(p.get("num_rays", 5))
            self.state.param_min_offset = float(p.get("min_offset", -0.5))
            self.state.param_max_offset = float(p.get("max_offset", 0.5))
            self.state.param_wavelength = float(p.get("wavelength", 550.0))

    def on_pick_coords(self, *args, **kwargs):
        # Заглушка, при необходимости можно реализовать picking через события PyVista
        pass

    def on_trace_mode_changed(self, *args, **kwargs):
        mode = args[0] if args else "tree"
        if mode != self.state.trace_mode:
            self.state.trace_mode = mode
            self.update_scene()

    def update_selected_object(self, *args, **kwargs):
        obj_entry = self._find_object(self.state.selected_object_id)
        if not obj_entry:
            return
        p = obj_entry["params"]
        p["origin"] = (
            float(self.state.param_pos_x),
            float(self.state.param_pos_y),
            float(self.state.param_pos_z)
        )
        p["rotation"] = (
            float(self.state.param_rot_x),
            float(self.state.param_rot_y),
            float(self.state.param_rot_z)
        )
        if obj_entry["type"] == "lens":
            p["n"] = float(self.state.param_n)
            p["R1"] = float(self.state.param_R1)
            p["R2"] = float(self.state.param_R2)
            p["thickness"] = float(self.state.param_thickness)
            p["edge_radius"] = float(self.state.param_edge_radius)
        elif obj_entry["type"] == "emitter":
            p["num_rays"] = int(self.state.param_num_rays)
            p["min_offset"] = float(self.state.param_min_offset)
            p["max_offset"] = float(self.state.param_max_offset)
            p["wavelength"] = float(self.state.param_wavelength)
        obj_entry["instance"] = self._create_instance(obj_entry["type"], p)
        self.update_scene()

    def on_param_change(self, *args, **kwargs):
        self.update_selected_object()

# ----------------------------------------------------------------
# Запуск
# ----------------------------------------------------------------
server = get_server()
server.client_type = "vue3"
app = OpticsAppController(server)

# Привязка параметров
for param in [
    "param_pos_x", "param_pos_y", "param_pos_z",
    "param_rot_x", "param_rot_y", "param_rot_z",
    "param_n", "param_R1", "param_R2", "param_thickness",
    "param_edge_radius", "param_num_rays", "param_min_offset",
    "param_max_offset", "param_wavelength"
]:
    server.state.change(param)(app.on_param_change)

# UI
with SinglePageLayout(server) as layout:
    layout.title.set_text("Оптический симулятор (серверный рендеринг)")

    with layout.content:
        with vuetify.VContainer(fluid=True, classes="pa-0", style="height: 100vh; overflow: hidden;"):
            with vuetify.VRow(no_gutters=True, style="height: 100%;"):
                # Левая панель
                with vuetify.VCol(
                    cols=3,
                    classes="pa-4 bg-grey-darken-4",
                    style="height: 100%; overflow-y: auto; border-right: 1px solid #444; color: white;"
                ):
                    vuetify.VCardTitle("Объекты сцены", classes="text-h6 px-0")

                    with vuetify.VRow(classes="py-2", no_gutters=True):
                        vuetify.VBtn("Добавить линзу", color="cyan", block=True,
                                     click=app.add_lens_click)
                        vuetify.VBtn("Добавить источник", color="green", block=True,
                                     class_="mt-2", click=app.add_emitter_click)

                    vuetify.VDivider(class_="my-4")

                    # Список объектов
                    vuetify.VCard(flat=True, color="transparent", max_height="200",
                                  style="overflow-y: auto;")
                    with vuetify.VList(dense=True, nav=True):
                        for obj in app.scene_objects:
                            with vuetify.VListItem(
                                key=obj["id"],
                                active=(f"selected_object_id == '{obj['id']}'",),
                                click=f"selected_object_id = '{obj['id']}'",
                                style="cursor: pointer;"
                            ):
                                vuetify.VIcon("mdi-cube-outline", small=True, class_="mr-2")
                                vuetify.VListItemTitle(obj["name"])
                                vuetify.VBtn(
                                    icon="mdi-delete", x_small=True, variant="text",
                                    color="red", click=(app.remove_object, obj["id"])
                                )

                    vuetify.VDivider(class_="my-4")

                    vuetify.VCardTitle(
                        "Параметры: {{ selected_object_id ? selected_object_id : 'не выбран' }}",
                        classes="text-subtitle-1 px-0 text-cyan-lighten-2"
                    )

                    # Позиция
                    vuetify.VListSubheader("Позиция", class_="px-0")
                    with vuetify.VRow(no_gutters=True, align="center"):
                        with vuetify.VCol(cols=6):
                            vuetify.VTextField(v_model=("param_pos_x",), label="X",
                                               type="number", dense=True, class_="mb-2")
                        with vuetify.VCol(cols=6):
                            vuetify.VSlider(v_model=("param_pos_x",), min=-10, max=10,
                                            step=0.1, dense=True, hide_details=True)
                    with vuetify.VRow(no_gutters=True, align="center"):
                        with vuetify.VCol(cols=6):
                            vuetify.VTextField(v_model=("param_pos_y",), label="Y",
                                               type="number", dense=True, class_="mb-2")
                        with vuetify.VCol(cols=6):
                            vuetify.VSlider(v_model=("param_pos_y",), min=-5, max=5,
                                            step=0.1, dense=True, hide_details=True)
                    with vuetify.VRow(no_gutters=True, align="center"):
                        with vuetify.VCol(cols=6):
                            vuetify.VTextField(v_model=("param_pos_z",), label="Z",
                                               type="number", dense=True, class_="mb-2")
                        with vuetify.VCol(cols=6):
                            vuetify.VSlider(v_model=("param_pos_z",), min=-5, max=5,
                                            step=0.1, dense=True, hide_details=True)

                    # Поворот
                    vuetify.VListSubheader("Поворот (град)", class_="px-0")
                    with vuetify.VRow(no_gutters=True, align="center"):
                        with vuetify.VCol(cols=6):
                            vuetify.VTextField(v_model=("param_rot_x",), label="X",
                                               type="number", dense=True, class_="mb-2")
                        with vuetify.VCol(cols=6):
                            vuetify.VSlider(v_model=("param_rot_x",), min=-180, max=180,
                                            step=1, dense=True, hide_details=True)
                    with vuetify.VRow(no_gutters=True, align="center"):
                        with vuetify.VCol(cols=6):
                            vuetify.VTextField(v_model=("param_rot_y",), label="Y",
                                               type="number", dense=True, class_="mb-2")
                        with vuetify.VCol(cols=6):
                            vuetify.VSlider(v_model=("param_rot_y",), min=-180, max=180,
                                            step=1, dense=True, hide_details=True)
                    with vuetify.VRow(no_gutters=True, align="center"):
                        with vuetify.VCol(cols=6):
                            vuetify.VTextField(v_model=("param_rot_z",), label="Z",
                                               type="number", dense=True, class_="mb-2")
                        with vuetify.VCol(cols=6):
                            vuetify.VSlider(v_model=("param_rot_z",), min=-180, max=180,
                                            step=1, dense=True, hide_details=True)

                    vuetify.VDivider(class_="my-4")

                    # Параметры линзы
                    with vuetify.VContainer(v_if=("selected_object_type === 'lens'",),
                                            class_="pa-0"):
                        vuetify.VListSubheader("Параметры линзы", class_="px-0")
                        vuetify.VSlider(v_model=("param_n",), min=1.0, max=2.5,
                                        step=0.01, label="Показатель преломления",
                                        dense=True)
                        vuetify.VSlider(v_model=("param_R1",), min=-20, max=20,
                                        step=0.5, label="R1 (передняя)", dense=True)
                        vuetify.VSlider(v_model=("param_R2",), min=-20, max=20,
                                        step=0.5, label="R2 (задняя)", dense=True)
                        vuetify.VSlider(v_model=("param_thickness",), min=0.1, max=5.0,
                                        step=0.1, label="Толщина", dense=True)
                        vuetify.VSlider(v_model=("param_edge_radius",), min=0.1, max=5.0,
                                        step=0.1, label="Радиус апертуры", dense=True)

                    # Параметры источника
                    with vuetify.VContainer(v_if=("selected_object_type === 'emitter'",),
                                            class_="pa-0"):
                        vuetify.VListSubheader("Параметры источника", class_="px-0")
                        vuetify.VSlider(v_model=("param_num_rays",), min=1, max=20,
                                        step=1, label="Количество лучей", dense=True)
                        vuetify.VSlider(v_model=("param_min_offset",), min=-3.0, max=0.0,
                                        step=0.1, label="Мин. смещение", dense=True)
                        vuetify.VSlider(v_model=("param_max_offset",), min=0.0, max=3.0,
                                        step=0.1, label="Макс. смещение", dense=True)
                        vuetify.VSlider(v_model=("param_wavelength",), min=380, max=780,
                                        step=10, label="Длина волны (нм)", dense=True)

                    vuetify.VDivider(class_="my-4")
                    vuetify.VSelect(
                        v_model=("trace_mode",),
                        items=[
                            {"title": "Simple", "value": "simple"},
                            {"title": "Tree", "value": "tree"}
                        ],
                        label="Режим трассировки",
                        dense=True,
                        class_="mt-2"
                    )

                # Правая колонка: серверный рендеринг
                with vuetify.VCol(cols=9, style="height: 100%;"):
                    ui_view = plotter_ui(app.plotter, mode="server", add_menu=False)
                    app.ctrl.view_update = ui_view.update

# Первичное обновление
app.update_scene()
app.plotter.view_isometric()

if __name__ == "__main__":
    server.start(host="127.0.0.1", port=8085)