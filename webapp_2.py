import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")

import asyncio
import numpy as np
import pyvista as pv
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3 as vuetify
from trame.widgets import vtk as trame_vtk
from scipy.spatial.transform import Rotation as R
from pyvista.trame.ui import plotter_ui

from main import (
    RayTracer, RayPool, UniversalLens, BeamEmitter, Ray, SimpleMode, TreeMode
)

# Режим отрисовки: "client" или "server"
RENDER_MODE = "client"   # или "server"


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

        self.temp_plotter = pv.Plotter(off_screen=True)
        self.pool = RayPool(initial_size=0)
        self.scene_objects = []
        self.object_counter = 0
        self.initializing = True
        self._updating = False

        # Создаём RayTracer и увеличиваем лимиты для TreeMode
        self.ray_tracer = RayTracer(
            self.temp_plotter,
            mode="simple",
            pool=self.pool,
            line_width=2.0,
            min_alpha=0.0005,
            gamma=0.3
        )
        self.ray_tracer.mode.energy_color_type = 0

        self._debounce_delay = 0.03
        self._debounce_timer = None

        self.state.selected_object_id = None
        self.state.selected_object_type = None
        self.state.trace_mode = "simple"
        self.state.scene_objects_list = []
        self.state.trace_modes = [
            {"title": "Simple", "value": "simple"},
            {"title": "Tree", "value": "tree"},
        ]

        # Инициализация базовых полей параметров
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

        self.ctrl.trigger("delete_object_event")(self.remove_object)

        self.state.change("selected_object_id")(self.on_object_selected)
        self.state.change("trace_mode")(self.on_trace_mode_changed)

        self.create_initial_objects()
        self.initializing = False

    def _update_objects_list_state(self):
        self.state.scene_objects_list = [
            {"id": o["id"], "name": o["name"], "type": o["type"]}
            for o in self.scene_objects
        ]

    def create_initial_objects(self):
        obj1 = self.add_object("lens", "Линза 1", {
            "origin": (-2.0, 0.0, 0.0), "rotation": (0, 0, 0),
            "R1": 5.0, "R2": -5.0, "thickness": 0.5, "edge_radius": 1.0, "n": 1.5,
            "reflection_range": None, "refraction_range": (0, np.inf), "absorption_range": None
        })
        self.add_object("lens", "Линза 2", {
            "origin": (2.0, 0.0, 0.0), "rotation": (0, 15, 0),
            "R1": 5.0, "R2": 2.0, "thickness": 0.5, "edge_radius": 1.0, "n": 1.5,
            "reflection_range": None, "refraction_range": (0, np.inf), "absorption_range": None
        })

        for y in np.linspace(-0.5, 0.5, 5):
            self.manual_rays.append(
                Ray(origin=(-5.0, y, 0.0), direction=(1, 0, 0), energy=1.0, color="yellow", wavelength=550))

        # Исправление Бага №2: Принудительно загружаем параметры первого объекта
        if self.scene_objects:
            first_id = self.scene_objects[0]["id"]
            self.state.selected_object_id = first_id
            self.on_object_selected(first_id)
        self._update_objects_list_state()

    def add_object(self, obj_type, name, params):
        self.object_counter += 1
        obj_id = f"obj_{self.object_counter}"
        instance = self._create_instance(obj_type, params)
        self.scene_objects.append({
            "id": obj_id, "type": obj_type, "name": name, "params": params, "instance": instance
        })
        if not self.initializing:
            self._update_objects_list_state()
            self.state.selected_object_id = obj_id
            self.on_object_selected(obj_id)  # Исправление Бага №2 для новых объектов
            self.update_scene()
        return obj_id

    def add_lens_click(self):
        self.add_object("lens", f"Линза {len(self.scene_objects) + 1}", {
            "origin": (0, 0, 0), "rotation": (0, 0, 0), "R1": 5.0, "R2": -5.0,
            "thickness": 0.5, "edge_radius": 1.0, "n": 1.5,
            "reflection_range": None, "refraction_range": (0, np.inf), "absorption_range": None
        })

    def add_emitter_click(self):
        self.add_object("emitter", f"Источник {len(self.scene_objects) + 1}", {
            "origin": (0, 0, 0), "rotation": (0, 0, 0), "num_rays": 5, "min_offset": -0.5,
            "max_offset": 0.5, "wavelength": 550.0, "color": "yellow", "energy": 1.0, "current_n": 1.0
        })

    def remove_object(self, *args, **kwargs):
        """Метод удаления объекта, адаптированный под триггер из UI"""
        # Извлекаем ID: trame присылает массив параметров из JS в args[0]
        if args and isinstance(args[0], list):
            obj_id = args[0][0]
        elif args:
            obj_id = args[0]
        else:
            return

        # Удаляем объект из памяти сцены
        self.scene_objects = [o for o in self.scene_objects if o["id"] != obj_id]

        # Если удалили активный объект — переключаем фокус на первый оставшийся
        if self.state.selected_object_id == obj_id:
            next_id = self.scene_objects[0]["id"] if self.scene_objects else None
            self.state.selected_object_id = next_id
            if next_id:
                self.on_object_selected(next_id)
            else:
                self.state.selected_object_type = None

        # Обновляем UI-список и перерисовываем 3D-сцену PyVista
        self._update_objects_list_state()
        self.update_scene()

    def _create_instance(self, obj_type, params):
        if obj_type == "lens":
            return UniversalLens(
                origin=params.get("origin", (0, 0, 0)), rotation_degrees=params.get("rotation", (0, 0, 0)),
                R1=params.get("R1"), R2=params.get("R2"), thickness=params.get("thickness", 0.5),
                edge_radius=params.get("edge_radius", 1.0), n=params.get("n", 1.5),
                reflection_range=params.get("reflection_range"),
                refraction_range=params.get("refraction_range", (0, np.inf)),
                absorption_range=params.get("absorption_range")
            )
        elif obj_type == "emitter":
            rot = params.get("rotation", (0, 0, 0))
            direction = R.from_euler('xyz', rot, degrees=True).apply([1, 0, 0])
            return BeamEmitter(
                origin=params.get("origin", (0, 0, 0)), direction=direction, num_rays=params.get("num_rays", 5),
                min_offset=params.get("min_offset", -0.5), max_offset=params.get("max_offset", 0.5),
                color=params.get("color", "yellow"), wavelength=params.get("wavelength", 550),
                energy=params.get("energy", 1.0), current_n=params.get("current_n", 1.0), pool=self.pool
            )

    def on_trace_mode_changed(self, *args, **kwargs):
        mode = args[0] if args else "simple"
        if mode == "tree":
            # Исправление Бага №4: увеличиваем глубину дерева рекурсии для прохода через много линз
            self.ray_tracer.set_mode("tree")
            self.ray_tracer.mode.max_depth = 30
        else:
            self.ray_tracer.set_mode("simple")
        self.update_scene()

    def update_scene(self):
        if self._updating: return
        self._updating = True
        try:
            self.plotter.clear()
            self.plotter.add_axes(color="white")
            self.plotter.set_background("#1a1a2e")
            self.plotter.enable_parallel_projection()

            for obj_entry in self.scene_objects:
                instance = obj_entry["instance"]
                if isinstance(instance, UniversalLens):
                    self.plotter.add_mesh(instance.get_mesh(), color="cyan", opacity=0.5, smooth_shading=True,
                                          name=obj_entry["id"])
                elif isinstance(instance, BeamEmitter):
                    self.plotter.add_mesh(instance.get_mesh(), color="green", name=obj_entry["id"])

            self.ray_tracer.elements.clear()
            self.ray_tracer.rays.clear()

            for obj_entry in self.scene_objects:
                instance = obj_entry["instance"]
                if isinstance(instance, UniversalLens):
                    for surf in instance.get_surfaces():
                        self.ray_tracer.add_elements(surf)

            for ray in self.manual_rays:
                self.ray_tracer.add_ray(ray)

            segments = self.ray_tracer.trace_all()

            if segments:
                segment_pairs = np.array([[seg.start, seg.end] for seg in segments], dtype=np.float32)
                all_points = segment_pairs.reshape(-1, 3)
                n_segments = len(segments)
                connectivity = np.empty((n_segments, 3), dtype=np.int64)
                connectivity[:, 0] = 2
                connectivity[:, 1] = np.arange(0, 2 * n_segments, 2)
                connectivity[:, 2] = np.arange(1, 2 * n_segments, 2)
                ray_mesh = pv.PolyData(all_points, lines=connectivity.ravel())
            else:
                ray_mesh = pv.PolyData()

            self.plotter.add_mesh(ray_mesh, color="yellow", line_width=2, name="traced_rays_geometry")
            self.plotter.render()
            if hasattr(self.ctrl, 'view_update'):
                self.ctrl.view_update()
        finally:
            self._updating = False

    def on_object_selected(self, *args, **kwargs):
        obj_id = args[0] if args else self.state.selected_object_id
        if not obj_id: return
        obj_entry = self._find_object(obj_id)
        if obj_entry: self._load_params_to_state(obj_entry)

    def _find_object(self, obj_id):
        for o in self.scene_objects:
            if o["id"] == obj_id: return o
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

    def update_selected_object(self, *args, **kwargs):
        obj_entry = self._find_object(self.state.selected_object_id)
        if not obj_entry: return
        p = obj_entry["params"]
        p["origin"] = (float(self.state.param_pos_x), float(self.state.param_pos_y), float(self.state.param_pos_z))
        p["rotation"] = (float(self.state.param_rot_x), float(self.state.param_rot_y), float(self.state.param_rot_z))
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
        if self._debounce_timer is not None:
            self._debounce_timer.cancel()
        loop = asyncio.get_event_loop()
        self._debounce_timer = loop.call_later(self._debounce_delay, self.update_selected_object)

    def select_object(self, obj_id):
        self.state.selected_object_id = obj_id
        self.on_object_selected(obj_id)


# --- Сборка интерфейса Trame / Vuetify 3 ---
server = get_server()
server.client_type = "vue3"
app = OpticsAppController(server)

for param in ["param_pos_x", "param_pos_y", "param_pos_z", "param_rot_x", "param_rot_y", "param_rot_z",
              "param_n", "param_R1", "param_R2", "param_thickness", "param_edge_radius", "param_num_rays",
              "param_min_offset", "param_max_offset", "param_wavelength"]:
    server.state.change(param)(app.on_param_change)

with SinglePageLayout(server) as layout:
    layout.title.set_text("Оптический симулятор")
    with layout.content:
        with vuetify.VContainer(fluid=True, classes="pa-0", style="height: 100vh; overflow: hidden;"):
            with vuetify.VRow(no_gutters=True, style="height: 100%;"):
                # Левая колонка управления
                with vuetify.VCol(cols=3, classes="pa-4 bg-grey-darken-4",
                                  style="height: 100%; overflow-y: auto; border-right: 1px solid #444; color: white;"):
                    vuetify.VCardTitle("Объекты сцены", classes="text-h6 px-0")
                    with vuetify.VRow(classes="py-2", no_gutters=True):
                        vuetify.VBtn("Добавить линзу", color="cyan", block=True, click=app.add_lens_click)
                        vuetify.VBtn("Добавить источник", color="green", block=True, class_="mt-2",
                                     click=app.add_emitter_click)
                    vuetify.VDivider(class_="my-4")
                    # Исправление Бага №1: Исправлена вложенность и добавлен скроллбокс для списка
                    with vuetify.VCard(flat=True, color="transparent", style="max-height: 200px; overflow-y: auto;"):
                        with vuetify.VList(dense=True, nav=True):
                            with vuetify.VListItem(
                                    v_for="obj in scene_objects_list",
                                    key="obj.id",
                                    active="selected_object_id == obj.id",
                                    # Безопасное реактивное переключение ID на стороне JS (без вызова Python напрямую)
                                    click="selected_object_id = obj.id",
                                    style="cursor: pointer;"
                            ):
                                vuetify.VIcon("mdi-cube-outline", small=True, class_="mr-2")
                                vuetify.VListItemTitle("{{ obj.name }}")

                                # Для Vuetify 3 кнопка внутри VListItem выносится в специальный именованный слот append
                                with vuetify.Template(v_slot_append=True):
                                    vuetify.VBtn(
                                        icon="mdi-delete",
                                        size="x-small",
                                        variant="text",
                                        color="red",
                                        # Используем нативную JS-инструкцию trame.trigger для вызова метода Python с аргументом
                                        click="trigger('delete_object_event', [obj.id])"
                                    )
                    vuetify.VDivider(class_="my-4")
                    vuetify.VCardTitle("Параметры: {{ selected_object_id ? selected_object_id : 'не выбран' }}",
                                       classes="text-subtitle-1 px-0 text-cyan-lighten-2")
                    # Блок позиционирования
                    vuetify.VListSubheader("Позиция", class_="px-0")
                    for axis in ["x", "y", "z"]:
                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(v_model=f"param_pos_{axis}", label=axis.upper(), type="number",
                                                   dense=True, class_="mb-2")
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(v_model=f"param_pos_{axis}", min=-10 if axis == "x" else -5,
                                                max=10 if axis == "x" else 5, step=0.1, dense=True, hide_details=True)
                    # Блок вращения
                    vuetify.VListSubheader("Поворот (град)", class_="px-0")
                    for axis in ["x", "y", "z"]:
                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(v_model=f"param_rot_{axis}", label=axis.upper(), type="number",
                                                   dense=True, class_="mb-2")
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(v_model=f"param_rot_{axis}", min=-180, max=180, step=1, dense=True,
                                                hide_details=True)
                    vuetify.VDivider(class_="my-4")
                    # Исправление Бага №3: Исправлен синтаксис v_if для отображения настроек линз
                    with vuetify.VContainer(v_if="selected_object_type == 'lens'", class_="pa-0"):
                        vuetify.VListSubheader("Параметры линзы", class_="px-0")
                        vuetify.VSlider(v_model="param_n", min=1.0, max=2.5, step=0.01, label="Показатель преломления",
                                        dense=True)
                        vuetify.VSlider(v_model="param_R1", min=-20, max=20, step=0.5, label="R1 (передняя)",
                                        dense=True)
                        vuetify.VSlider(v_model="param_R2", min=-20, max=20, step=0.5, label="R2 (задняя)", dense=True)
                        vuetify.VSlider(v_model="param_thickness", min=0.1, max=5.0, step=0.1, label="Толщина",
                                        dense=True)
                        vuetify.VSlider(v_model="param_edge_radius", min=0.1, max=5.0, step=0.1,
                                        label="Радиус апертуры", dense=True)
                    with vuetify.VContainer(v_if="selected_object_type == 'emitter'", class_="pa-0"):
                        vuetify.VListSubheader("Параметры источника", class_="px-0")
                        vuetify.VSlider(v_model="param_num_rays", min=1, max=20, step=1, label="Количество лучей",
                                        dense=True)
                        vuetify.VSlider(v_model="param_min_offset", min=-3.0, max=0.0, step=0.1, label="Мин. смещение",
                                        dense=True)
                        vuetify.VSlider(v_model="param_max_offset", min=0.0, max=3.0, step=0.1, label="Макс. смещение",
                                        dense=True)
                        vuetify.VSlider(v_model="param_wavelength", min=380, max=780, step=10, label="Длина волны (нм)",
                                        dense=True)
                    vuetify.VDivider(class_="my-4")
                    vuetify.VSelect(label="Режим трассировки", v_model="trace_mode", items=("trace_modes",),
                                    item_title="title", item_value="value", dense=True, class_="mt-2")
                # Правая колонка с 3D окном pyvista
                with vuetify.VCol(cols=9, style="height: 100%; overflow: hidden;"):
                    ui_view = plotter_ui(app.plotter, mode=RENDER_MODE, add_menu=False, image_scale=1)
                    app.ctrl.view_update = ui_view.update

app.update_scene()

if __name__ == "__main__":
    server.start(host="127.0.0.1", port=8085)