import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import pyvista as pv
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3 as vuetify
from trame.widgets import vtk as trame_vtk
from scipy.spatial.transform import Rotation as R

from main import (
    RayTracer, RayPool, UniversalLens, BeamEmitter, Ray,
    SimpleMode, TreeMode
)

# -------------------------------------------------------------
# Контроллер приложения
# -------------------------------------------------------------
class OpticsApp:
    def __init__(self, server):
        self.server = server
        self.state = server.state
        self.ctrl = server.controller

        # ---- Реактивные переменные (синхронизируются с UI) ----
        self.state.objects = []                 # список объектов
        self.state.selected_id = None           # id выбранного объекта
        self.state.selected_type = None
        self.state.trace_mode = "tree"

        # Параметры выбранного объекта (привязаны к виджетам)
        self.state.pos_x = 0.0
        self.state.pos_y = 0.0
        self.state.pos_z = 0.0
        self.state.rot_x = 0.0
        self.state.rot_y = 0.0
        self.state.rot_z = 0.0
        self.state.n = 1.5
        self.state.R1 = 5.0
        self.state.R2 = -5.0
        self.state.thickness = 0.5
        self.state.edge_radius = 1.0
        self.state.num_rays = 5
        self.state.min_offset = -0.5
        self.state.max_offset = 0.5
        self.state.wavelength = 550.0

        # ---- Не-реактивные данные ----
        self.plotter = pv.Plotter(off_screen=True)
        self.plotter.set_background("#1a1a2e")
        self.plotter.add_axes(color="white")
        self.plotter.enable_parallel_projection()
        self.plotter.view_isometric()

        self.pool = RayPool(initial_size=200)
        self.ray_tracer = RayTracer(
            self.plotter,
            mode="tree",
            pool=self.pool,
            line_width=2.0,
            min_alpha=0.05,
            gamma=0.3
        )
        self.ray_tracer.mode.energy_color_type = 2

        self.object_actors = {}
        self.ray_actor = None
        self._updating = False
        self._initializing = True

        # ---- Привязка изменений состояния ----
        # При изменении selected_id загружаем параметры объекта
        self.state.change("selected_id")(self.on_selected_changed)
        self.state.change("trace_mode")(self.on_trace_mode_changed)
        # При изменении любого параметра обновляем объект
        for param in ["pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z",
                      "n", "R1", "R2", "thickness", "edge_radius",
                      "num_rays", "min_offset", "max_offset", "wavelength"]:
            self.state.change(param)(self.on_param_changed)

        # ---- Создаём начальную сцену ----
        self.create_initial_scene()
        self._initializing = False

    # ---------------------------------------------------------
    # Создание / удаление объектов
    # ---------------------------------------------------------
    def create_initial_scene(self):
        self.add_object("lens", "Линза 1", {
            "origin": (-2.0, 0.0, 0.0),
            "rotation": (0, 0, 0),
            "R1": 5.0, "R2": -5.0, "thickness": 0.5,
            "edge_radius": 1.0, "n": 1.5,
            "refraction_range": (0, np.inf)
        })
        self.add_object("lens", "Линза 2", {
            "origin": (2.0, 0.0, 0.0),
            "rotation": (0, 15, 0),
            "R1": 5.0, "R2": -5.0, "thickness": 0.5,
            "edge_radius": 1.0, "n": 1.5,
            "refraction_range": (0, np.inf)
        })
        self.manual_rays = []
        for y in np.linspace(-0.5, 0.5, 5):
            ray = Ray(origin=(-5.0, y, 0.0), direction=(1, 0, 0),
                      energy=1.0, color="yellow", wavelength=550)
            self.manual_rays.append(ray)

        if self.state.objects:
            self.state.selected_id = self.state.objects[0]["id"]

    def add_object(self, obj_type, name, params):
        obj_id = f"obj_{len(self.state.objects) + 1}_{obj_type}"
        instance = self._create_instance(obj_type, params)
        entry = {
            "id": obj_id,
            "type": obj_type,
            "name": name,
            "params": params.copy(),
            "instance": instance
        }
        # Добавляем в реактивный список
        self.state.objects.append(entry)
        if not self._initializing:
            self.state.selected_id = obj_id
            self.update_scene()

    def _create_instance(self, obj_type, params):
        if obj_type == "lens":
            return UniversalLens(
                origin=params["origin"],
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
                origin=params["origin"],
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

    def remove_object(self, obj_id):
        for i, entry in enumerate(self.state.objects):
            if entry["id"] == obj_id:
                # Удаляем актёр объекта
                if obj_id in self.object_actors:
                    self.plotter.remove_actor(self.object_actors.pop(obj_id))
                del self.state.objects[i]
                if self.state.selected_id == obj_id:
                    self.state.selected_id = self.state.objects[0]["id"] if self.state.objects else None
                self.update_scene()
                return

    # ---------------------------------------------------------
    # Обновление сцены (без clear!)
    # ---------------------------------------------------------
    def update_scene(self):
        if self._updating:
            return
        self._updating = True
        try:
            current_ids = {entry["id"] for entry in self.state.objects}

            # 1. Обновляем актёры объектов
            for entry in self.state.objects:
                obj_id = entry["id"]
                instance = entry["instance"]
                mesh = instance.get_mesh() if hasattr(instance, 'get_mesh') else None

                if obj_id in self.object_actors:
                    actor = self.object_actors[obj_id]
                    if mesh is not None:
                        actor.mapper.dataset.copy_from(mesh)
                        actor.mapper.Update()
                else:
                    if isinstance(instance, UniversalLens):
                        actor = self.plotter.add_mesh(
                            mesh, color="cyan", opacity=0.5, smooth_shading=True,
                            pickable=True, name=obj_id
                        )
                    elif isinstance(instance, BeamEmitter):
                        actor = self.plotter.add_mesh(
                            mesh, color="green", pickable=True, name=obj_id
                        )
                    else:
                        actor = self.plotter.add_mesh(
                            mesh, color="gray", opacity=0.7, pickable=True, name=obj_id
                        )
                    self.object_actors[obj_id] = actor

            # 2. Удаляем актёры для отсутствующих объектов
            for obj_id in list(self.object_actors.keys()):
                if obj_id not in current_ids:
                    actor = self.object_actors.pop(obj_id)
                    self.plotter.remove_actor(actor)

            # 3. Обновляем трассировщик
            self.ray_tracer.elements.clear()
            self.ray_tracer.rays.clear()
            if hasattr(self.ray_tracer, 'emitters'):
                self.ray_tracer.emitters.clear()

            for entry in self.state.objects:
                instance = entry["instance"]
                if isinstance(instance, UniversalLens):
                    for surf in instance.get_surfaces():
                        self.ray_tracer.add_elements(surf)

            for ray in getattr(self, "manual_rays", []):
                self.ray_tracer.add_ray(ray)

            segments = self.ray_tracer.trace_all()

            # 4. Строим геометрию лучей
            if segments:
                points = []
                lines = []
                offset = 0
                for seg in segments:
                    points.extend([seg.start, seg.end])
                    lines.extend([2, offset, offset + 1])
                    offset += 2
                points = np.array(points, dtype=np.float32)
                lines = np.array(lines, dtype=np.int64)
                ray_mesh = pv.PolyData(points, lines=lines)
            else:
                ray_mesh = pv.PolyData()

            # 5. Обновляем актёр лучей
            if self.ray_actor is None:
                self.ray_actor = self.plotter.add_mesh(
                    ray_mesh,
                    color="yellow",
                    line_width=2,
                    render_lines_as_tubes=False,
                    name="traced_rays"
                )
            else:
                self.ray_actor.mapper.dataset.copy_from(ray_mesh)
                self.ray_actor.mapper.Update()

            # 6. Рендерим и отправляем клиенту
            self.plotter.render()
            if hasattr(self.ctrl, 'view_update') and self.ctrl.view_update is not None:
                self.ctrl.view_update()

        finally:
            self._updating = False

    # ---------------------------------------------------------
    # Обработчики изменений состояния
    # ---------------------------------------------------------
    def on_selected_changed(self, obj_id, **kwargs):
        """При смене выбранного объекта загружаем его параметры в state."""
        if not obj_id:
            return
        entry = self._find_object(obj_id)
        if entry:
            self._load_params_to_state(entry)

    def _find_object(self, obj_id):
        for entry in self.state.objects:
            if entry["id"] == obj_id:
                return entry
        return None

    def _load_params_to_state(self, entry):
        """Загружает параметры объекта в реактивные переменные."""
        p = entry["params"]
        self.state.selected_type = entry["type"]
        # Позиция
        self.state.pos_x = float(p["origin"][0])
        self.state.pos_y = float(p["origin"][1])
        self.state.pos_z = float(p["origin"][2])
        # Поворот
        rot = p.get("rotation", (0,0,0))
        self.state.rot_x = float(rot[0])
        self.state.rot_y = float(rot[1])
        self.state.rot_z = float(rot[2])
        # Специфические параметры
        if entry["type"] == "lens":
            self.state.n = float(p.get("n", 1.5))
            self.state.R1 = float(p.get("R1", 5.0))
            self.state.R2 = float(p.get("R2", -5.0))
            self.state.thickness = float(p.get("thickness", 0.5))
            self.state.edge_radius = float(p.get("edge_radius", 1.0))
        elif entry["type"] == "emitter":
            self.state.num_rays = int(p.get("num_rays", 5))
            self.state.min_offset = float(p.get("min_offset", -0.5))
            self.state.max_offset = float(p.get("max_offset", 0.5))
            self.state.wavelength = float(p.get("wavelength", 550.0))

    def on_param_changed(self, **kwargs):
        """При изменении любого параметра обновляем выбранный объект."""
        obj_id = self.state.selected_id
        if not obj_id:
            return
        entry = self._find_object(obj_id)
        if not entry:
            return
        p = entry["params"]
        # Обновляем параметры из state
        p["origin"] = (self.state.pos_x, self.state.pos_y, self.state.pos_z)
        p["rotation"] = (self.state.rot_x, self.state.rot_y, self.state.rot_z)
        if entry["type"] == "lens":
            p["n"] = self.state.n
            p["R1"] = self.state.R1
            p["R2"] = self.state.R2
            p["thickness"] = self.state.thickness
            p["edge_radius"] = self.state.edge_radius
        elif entry["type"] == "emitter":
            p["num_rays"] = self.state.num_rays
            p["min_offset"] = self.state.min_offset
            p["max_offset"] = self.state.max_offset
            p["wavelength"] = self.state.wavelength
        # Пересоздаём экземпляр
        entry["instance"] = self._create_instance(entry["type"], p)
        self.update_scene()

    def on_trace_mode_changed(self, mode, **kwargs):
        if mode == "simple":
            self.ray_tracer.mode = SimpleMode(max_bounces=10, energy_color_type=1)
        else:
            self.ray_tracer.mode = TreeMode(max_depth=10, min_energy=0.001,
                                            energy_color_type=1)
        self.update_scene()

    # ---------------------------------------------------------
    # Действия пользователя (кнопки)
    # ---------------------------------------------------------
    def add_lens(self):
        self.add_object("lens", f"Линза {len(self.state.objects)+1}", {
            "origin": (0, 0, 0),
            "rotation": (0, 0, 0),
            "R1": 5.0, "R2": -5.0, "thickness": 0.5,
            "edge_radius": 1.0, "n": 1.5,
            "refraction_range": (0, np.inf)
        })

    def add_emitter(self):
        self.add_object("emitter", f"Источник {len(self.state.objects)+1}", {
            "origin": (0, 0, 0),
            "rotation": (0, 0, 0),
            "num_rays": 5, "min_offset": -0.5, "max_offset": 0.5,
            "wavelength": 550.0, "color": "yellow",
            "energy": 1.0, "current_n": 1.0
        })

    def remove_selected(self):
        if self.state.selected_id:
            self.remove_object(self.state.selected_id)

# -------------------------------------------------------------
# Запуск приложения
# -------------------------------------------------------------
server = get_server()
app = OpticsApp(server)

# UI
with SinglePageLayout(server) as layout:
    layout.title.set_text("Оптический симулятор (финальная версия)")

    with layout.content:
        with vuetify.VContainer(fluid=True, classes="pa-0", style="height: 100vh; overflow: hidden;"):
            with vuetify.VRow(no_gutters=True, style="height: 100%;"):
                # Левая панель
                with vuetify.VCol(cols=3, classes="pa-4 bg-grey-darken-4",
                                  style="height: 100%; overflow-y: auto; border-right: 1px solid #444; color: white;"):
                    vuetify.VCardTitle("Объекты сцены", classes="text-h6 px-0")

                    with vuetify.VRow(no_gutters=True, class_="py-2"):
                        vuetify.VBtn("Добавить линзу", color="cyan", block=True,
                                     click=app.add_lens)
                        vuetify.VBtn("Добавить источник", color="green", block=True,
                                     class_="mt-2", click=app.add_emitter)
                        vuetify.VBtn("Удалить выбранный", color="red", block=True,
                                     class_="mt-2", click=app.remove_selected)

                    vuetify.VDivider(class_="my-4")

                    # Список объектов (реактивный) – использует state.objects
                    vuetify.VList(
                        dense=True,
                        nav=True,
                        items=("objects",),
                        item_title="name",
                        item_value="id",
                        v_model=("selected_id",),
                    )

                    vuetify.VDivider(class_="my-4")

                    # Параметры (привязаны к state)
                    vuetify.VCardTitle("Параметры: {{ selected_id || 'не выбран' }}",
                                       classes="text-subtitle-1 px-0 text-cyan-lighten-2")

                    # Позиция
                    vuetify.VListSubheader("Позиция", class_="px-0")
                    with vuetify.VRow(no_gutters=True, align="center"):
                        with vuetify.VCol(cols=6):
                            vuetify.VTextField(v_model=("pos_x",), label="X",
                                               type="number", dense=True, class_="mb-2")
                        with vuetify.VCol(cols=6):
                            vuetify.VSlider(v_model=("pos_x",), min=-10, max=10, step=0.1,
                                            dense=True, hide_details=True)
                    with vuetify.VRow(no_gutters=True, align="center"):
                        with vuetify.VCol(cols=6):
                            vuetify.VTextField(v_model=("pos_y",), label="Y",
                                               type="number", dense=True, class_="mb-2")
                        with vuetify.VCol(cols=6):
                            vuetify.VSlider(v_model=("pos_y",), min=-5, max=5, step=0.1,
                                            dense=True, hide_details=True)
                    with vuetify.VRow(no_gutters=True, align="center"):
                        with vuetify.VCol(cols=6):
                            vuetify.VTextField(v_model=("pos_z",), label="Z",
                                               type="number", dense=True, class_="mb-2")
                        with vuetify.VCol(cols=6):
                            vuetify.VSlider(v_model=("pos_z",), min=-5, max=5, step=0.1,
                                            dense=True, hide_details=True)

                    vuetify.VListSubheader("Поворот (град)", class_="px-0")
                    with vuetify.VRow(no_gutters=True, align="center"):
                        with vuetify.VCol(cols=6):
                            vuetify.VTextField(v_model=("rot_x",), label="X",
                                               type="number", dense=True, class_="mb-2")
                        with vuetify.VCol(cols=6):
                            vuetify.VSlider(v_model=("rot_x",), min=-180, max=180, step=1,
                                            dense=True, hide_details=True)
                    with vuetify.VRow(no_gutters=True, align="center"):
                        with vuetify.VCol(cols=6):
                            vuetify.VTextField(v_model=("rot_y",), label="Y",
                                               type="number", dense=True, class_="mb-2")
                        with vuetify.VCol(cols=6):
                            vuetify.VSlider(v_model=("rot_y",), min=-180, max=180, step=1,
                                            dense=True, hide_details=True)
                    with vuetify.VRow(no_gutters=True, align="center"):
                        with vuetify.VCol(cols=6):
                            vuetify.VTextField(v_model=("rot_z",), label="Z",
                                               type="number", dense=True, class_="mb-2")
                        with vuetify.VCol(cols=6):
                            vuetify.VSlider(v_model=("rot_z",), min=-180, max=180, step=1,
                                            dense=True, hide_details=True)

                    vuetify.VDivider(class_="my-4")

                    # Параметры линзы
                    with vuetify.VContainer(v_if=("selected_type === 'lens'",), class_="pa-0"):
                        vuetify.VListSubheader("Параметры линзы", class_="px-0")
                        vuetify.VSlider(v_model=("n",), min=1.0, max=2.5, step=0.01,
                                        label="Показатель преломления", dense=True)
                        vuetify.VSlider(v_model=("R1",), min=-20, max=20, step=0.5,
                                        label="R1 (передняя)", dense=True)
                        vuetify.VSlider(v_model=("R2",), min=-20, max=20, step=0.5,
                                        label="R2 (задняя)", dense=True)
                        vuetify.VSlider(v_model=("thickness",), min=0.1, max=5.0, step=0.1,
                                        label="Толщина", dense=True)
                        vuetify.VSlider(v_model=("edge_radius",), min=0.1, max=5.0, step=0.1,
                                        label="Радиус апертуры", dense=True)

                    # Параметры источника
                    with vuetify.VContainer(v_if=("selected_type === 'emitter'",), class_="pa-0"):
                        vuetify.VListSubheader("Параметры источника", class_="px-0")
                        vuetify.VSlider(v_model=("num_rays",), min=1, max=20, step=1,
                                        label="Количество лучей", dense=True)
                        vuetify.VSlider(v_model=("min_offset",), min=-3.0, max=0.0, step=0.1,
                                        label="Мин. смещение", dense=True)
                        vuetify.VSlider(v_model=("max_offset",), min=0.0, max=3.0, step=0.1,
                                        label="Макс. смещение", dense=True)
                        vuetify.VSlider(v_model=("wavelength",), min=380, max=780, step=10,
                                        label="Длина волны (нм)", dense=True)

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

                # Правая колонка: 3D-вид (клиентский рендеринг)
                with vuetify.VCol(cols=9, style="height: 100%;"):
                    html_view = trame_vtk.VtkLocalView(app.plotter.render_window)
                    app.ctrl.view_update = html_view.update
                    app.ctrl.view_reset_camera = html_view.reset_camera

# Первичное обновление (после создания UI)
app._initializing = False
app.update_scene()
app.plotter.view_isometric()

if __name__ == "__main__":
    server.start(host="127.0.0.1", port=8085)