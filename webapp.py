"""
Оптический симулятор с GUI на Trame.
Интеграция основной программы (RayTracer, UniversalLens, BeamEmitter) с веб-интерфейсом.
"""

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")

import numpy as np
import pyvista as pv
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3 as vuetify
from trame.widgets import vtk as trame_vtk
from scipy.spatial.transform import Rotation as R

# Импорт из основной программы (файлы должны быть в той же директории)
from main import (
    RayTracer, RayPool, RayCloud, UniversalLens, BeamEmitter,
    PlaneSurface, SphereSurface, RAY_INFINITY_DISTANCE
)

# =============================================================================
# КОНТРОЛЛЕР ПРИЛОЖЕНИЯ
# =============================================================================

class OpticsAppController:
    def __init__(self, server):
        self.server = server
        self.state = server.state
        self.ctrl = server.controller

        # Инициализация 3D-плоттера
        self.plotter = pv.Plotter(off_screen=True)
        self.plotter.set_background("#1a1a2e")
        self.plotter.add_axes(color="white")
        self.plotter.enable_parallel_projection()
        self.plotter.view_isometric()

        # Пул лучей для производительности
        self.pool = RayPool(initial_size=200)

        # Хранилище объектов сцены
        self.scene_objects = []
        self.object_counter = 0

        # RayTracer будет пересоздаваться при обновлении сцены
        self.ray_tracer = None

        # Реактивное состояние Trame
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

        # Регистрация обработчиков
        self.state.change("selected_object_id")(self.on_object_selected)
        self.state.change("pick_coords")(self.on_pick_coords)
        self.state.change("trace_mode")(self.on_trace_mode_changed)

        # Создание начальной сцены (без вызова update_scene, он будет вызван после связывания view)
        self.create_initial_objects()

    # -------------------------------------------------------------------------
    # СОЗДАНИЕ И УПРАВЛЕНИЕ ОБЪЕКТАМИ
    # -------------------------------------------------------------------------
    def create_initial_objects(self):
        """Создаёт две линзы и источник лучей."""
        self.add_object(
            obj_type="lens",
            name="Линза 1",
            params={
                "origin": (-2.0, 0.0, 0.0),
                "rotation": (0, 0, 0),
                "R1": 5.0,
                "R2": -5.0,
                "thickness": 0.5,
                "edge_radius": 1.0,
                "n": 1.5,
                "reflection_range": None,
                "refraction_range": (0, np.inf),
                "absorption_range": None
            }
        )
        self.add_object(
            obj_type="lens",
            name="Линза 2",
            params={
                "origin": (2.0, 0.0, 0.0),
                "rotation": (0, 15, 0),
                "R1": 5.0,
                "R2": -5.0,
                "thickness": 0.5,
                "edge_radius": 1.0,
                "n": 1.5,
                "reflection_range": None,
                "refraction_range": (0, np.inf),
                "absorption_range": None
            }
        )
        self.add_object(
            obj_type="emitter",
            name="Источник",
            params={
                "origin": (-5.0, 0.0, 0.0),
                "rotation": (0, 0, 0),
                "num_rays": 5,
                "min_offset": -0.5,
                "max_offset": 0.5,
                "wavelength": 550.0,
                "color": "yellow",
                "energy": 1.0,
                "current_n": 1.0
            }
        )
        # Выбираем первый объект
        if self.scene_objects:
            self.state.selected_object_id = self.scene_objects[0]["id"]

    def add_object(self, obj_type, name, params):
        """Добавляет объект в сцену."""
        self.object_counter += 1
        obj_id = f"obj_{self.object_counter}"
        obj_entry = {
            "id": obj_id,
            "type": obj_type,
            "name": name,
            "params": params,
            "instance": None
        }
        instance = self._create_instance(obj_type, params)
        obj_entry["instance"] = instance
        self.scene_objects.append(obj_entry)
        return obj_entry

    def _create_instance(self, obj_type, params):
        """Создаёт экземпляр класса из основной программы на основе параметров."""
        if obj_type == "lens":
            rot = params.get("rotation", (0, 0, 0))
            origin = params.get("origin", (0, 0, 0))
            return UniversalLens(
                origin=origin,
                rotation_degrees=rot,
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
            rot = params.get("rotation", (0, 0, 0))
            origin = params.get("origin", (0, 0, 0))
            direction = np.array([1.0, 0.0, 0.0])
            direction = R.from_euler('xyz', rot, degrees=True).apply(direction)
            return BeamEmitter(
                origin=origin,
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
            raise ValueError(f"Неизвестный тип объекта: {obj_type}")

    def remove_object(self, obj_id):
        """Удаляет объект из сцены."""
        self.scene_objects = [obj for obj in self.scene_objects if obj["id"] != obj_id]
        if self.state.selected_object_id == obj_id:
            self.state.selected_object_id = self.scene_objects[0]["id"] if self.scene_objects else None
        self.update_scene()

    # -------------------------------------------------------------------------
    # ОБНОВЛЕНИЕ СЦЕНЫ
    # -------------------------------------------------------------------------
    def update_scene(self):
        """Полная перестройка сцены: очищает plotter, пересоздаёт RayTracer, добавляет объекты и лучи."""
        # Очистка старой сцены
        self.plotter.clear()
        self.plotter.add_axes(color="white")
        self.plotter.set_background("#1a1a2e")
        self.plotter.enable_parallel_projection()
        self.plotter.view_isometric()

        # Создаём новый RayTracer
        self.ray_tracer = RayTracer(
            self.plotter,
            mode=self.state.trace_mode,
            pool=self.pool,
            line_width=2.0,
            min_alpha=0.05,
            gamma=0.3
        )
        self.ray_tracer.mode.energy_color_type = self.state.energy_color_type

        # Добавляем все объекты в RayTracer и отображаем их меши
        for obj_entry in self.scene_objects:
            instance = obj_entry["instance"]
            if isinstance(instance, UniversalLens):
                # Линзы добавляются как набор поверхностей
                for surf in instance.get_surfaces():
                    self.ray_tracer.add_elements(surf)
                # Визуализация линзы
                mesh = instance.get_mesh()
                actor = self.plotter.add_mesh(
                    mesh,
                    color="cyan",
                    opacity=0.5,
                    smooth_shading=True,
                    pickable=True,
                    name=obj_entry["id"]
                )
                obj_entry["actor"] = actor
            elif isinstance(instance, BeamEmitter):
                # Источник добавляется в эмиттеры RayTracer
                self.ray_tracer.add_emitter(instance)
                # Визуализация источника (стрелка)
                arrow = instance.get_mesh()
                actor = self.plotter.add_mesh(
                    arrow,
                    color="green",
                    pickable=True,
                    name=obj_entry["id"]
                )
                obj_entry["actor"] = actor
            else:
                mesh = instance.get_mesh()
                actor = self.plotter.add_mesh(
                    mesh,
                    color="gray",
                    opacity=0.7,
                    pickable=True,
                    name=obj_entry["id"]
                )
                obj_entry["actor"] = actor

        # Запускаем трассировку лучей
        segments = self.ray_tracer.trace_all()
        self.ray_tracer.cloud.update(
            segments,
            energy_color_type=self.ray_tracer.mode.energy_color_type
        )

        # Обновляем вьюпорт (теперь view_update уже связан)
        if hasattr(self.ctrl, 'view_update'):
            self.ctrl.view_update()

    # -------------------------------------------------------------------------
    # ОБРАБОТЧИКИ СОБЫТИЙ
    # -------------------------------------------------------------------------
    def on_object_selected(self, obj_id, **kwargs):
        if obj_id is None:
            return
        obj_entry = self._find_object(obj_id)
        if obj_entry:
            self._load_params_to_state(obj_entry)

    def _find_object(self, obj_id):
        for obj in self.scene_objects:
            if obj["id"] == obj_id:
                return obj
        return None

    def _load_params_to_state(self, obj_entry):
        params = obj_entry["params"]
        self.state.selected_object_type = obj_entry["type"]
        origin = params.get("origin", (0, 0, 0))
        self.state.param_pos_x = float(origin[0])
        self.state.param_pos_y = float(origin[1])
        self.state.param_pos_z = float(origin[2])
        rot = params.get("rotation", (0, 0, 0))
        self.state.param_rot_x = float(rot[0])
        self.state.param_rot_y = float(rot[1])
        self.state.param_rot_z = float(rot[2])

        if obj_entry["type"] == "lens":
            self.state.param_n = float(params.get("n", 1.5))
            self.state.param_R1 = float(params.get("R1", 5.0))
            self.state.param_R2 = float(params.get("R2", -5.0))
            self.state.param_thickness = float(params.get("thickness", 0.5))
            self.state.param_edge_radius = float(params.get("edge_radius", 1.0))
        elif obj_entry["type"] == "emitter":
            self.state.param_num_rays = int(params.get("num_rays", 5))
            self.state.param_min_offset = float(params.get("min_offset", -0.5))
            self.state.param_max_offset = float(params.get("max_offset", 0.5))
            self.state.param_wavelength = float(params.get("wavelength", 550.0))

    def on_pick_coords(self, coords, **kwargs):
        if coords is None:
            return
        try:
            x, y = coords
            picked = self.plotter.pick(mouse_x=x, mouse_y=y)
            if picked is not None and hasattr(picked, "name"):
                obj_id = picked.name
                obj_entry = self._find_object(obj_id)
                if obj_entry:
                    self.state.selected_object_id = obj_id
                    self._load_params_to_state(obj_entry)
        except Exception as e:
            print(f"Ошибка picking: {e}")

    def on_trace_mode_changed(self, mode, **kwargs):
        self.update_scene()

    def update_selected_object(self):
        obj_entry = self._find_object(self.state.selected_object_id)
        if obj_entry is None:
            return
        params = obj_entry["params"]
        params["origin"] = (self.state.param_pos_x, self.state.param_pos_y, self.state.param_pos_z)
        params["rotation"] = (self.state.param_rot_x, self.state.param_rot_y, self.state.param_rot_z)

        if obj_entry["type"] == "lens":
            params["n"] = self.state.param_n
            params["R1"] = self.state.param_R1
            params["R2"] = self.state.param_R2
            params["thickness"] = self.state.param_thickness
            params["edge_radius"] = self.state.param_edge_radius
        elif obj_entry["type"] == "emitter":
            params["num_rays"] = self.state.param_num_rays
            params["min_offset"] = self.state.param_min_offset
            params["max_offset"] = self.state.param_max_offset
            params["wavelength"] = self.state.param_wavelength

        obj_entry["instance"] = self._create_instance(obj_entry["type"], params)
        self.update_scene()

    def on_param_change(self, *args, **kwargs):
        self.update_selected_object()

# =============================================================================
# НАСТРОЙКА TRAME И GUI
# =============================================================================

server = get_server()
server.client_type = "vue3"
app = OpticsAppController(server)

# Привязываем обработчики к изменению параметров
for param_name in [
    "param_pos_x", "param_pos_y", "param_pos_z",
    "param_rot_x", "param_rot_y", "param_rot_z",
    "param_n", "param_R1", "param_R2", "param_thickness",
    "param_edge_radius", "param_num_rays", "param_min_offset",
    "param_max_offset", "param_wavelength"
]:
    server.state.change(param_name)(app.on_param_change)

with SinglePageLayout(server) as layout:
    layout.title.set_text("Оптический симулятор")

    with layout.content:
        with vuetify.VContainer(fluid=True, classes="pa-0 fill-height", style="height: 100vh; overflow: hidden;"):
            with vuetify.VRow(no_gutters=True, classes="fill-height", style="height: 100%;"):
                # Левая панель управления
                with vuetify.VCol(cols=3, classes="pa-4 bg-grey-darken-4",
                                  style="border-right: 1px solid #444; overflow-y: auto; color: white;"):
                    vuetify.VCardTitle("Объекты сцены", classes="text-h6 px-0")

                    with vuetify.VRow(classes="py-2", no_gutters=True):
                        vuetify.VBtn(
                            "Добавить линзу",
                            color="cyan",
                            block=True,
                            click=lambda: app.add_object(
                                "lens",
                                f"Линза {len(app.scene_objects)+1}",
                                {
                                    "origin": (0, 0, 0),
                                    "rotation": (0, 0, 0),
                                    "R1": 5.0,
                                    "R2": -5.0,
                                    "thickness": 0.5,
                                    "edge_radius": 1.0,
                                    "n": 1.5,
                                    "reflection_range": None,
                                    "refraction_range": (0, np.inf),
                                    "absorption_range": None
                                }
                            )
                        )
                        vuetify.VBtn(
                            "Добавить источник",
                            color="green",
                            block=True,
                            class_="mt-2",
                            click=lambda: app.add_object(
                                "emitter",
                                f"Источник {len(app.scene_objects)+1}",
                                {
                                    "origin": (0, 0, 0),
                                    "rotation": (0, 0, 0),
                                    "num_rays": 5,
                                    "min_offset": -0.5,
                                    "max_offset": 0.5,
                                    "wavelength": 550.0,
                                    "color": "yellow",
                                    "energy": 1.0,
                                    "current_n": 1.0
                                }
                            )
                        )

                    vuetify.VDivider(class_="my-4")

                    # Список объектов
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
                                    icon="mdi-delete",
                                    x_small=True,
                                    variant="text",
                                    color="red",
                                    click=f"remove_object('{obj['id']}')"
                                )

                    vuetify.VDivider(class_="my-4")

                    vuetify.VCardTitle(
                        "Параметры: {{ selected_object_id ? selected_object_id : 'не выбран' }}",
                        classes="text-subtitle-1 px-0 text-cyan-lighten-2"
                    )

                    vuetify.VListSubheader("Позиция (X, Y, Z)", class_="px-0")
                    with vuetify.VRow(no_gutters=True):
                        with vuetify.VCol(cols=12):
                            vuetify.VSlider(
                                v_model=("param_pos_x",),
                                min=-10, max=10, step=0.1,
                                label="X", color="cyan", dense=True
                            )
                            vuetify.VSlider(
                                v_model=("param_pos_y",),
                                min=-5, max=5, step=0.1,
                                label="Y", color="cyan", dense=True
                            )
                            vuetify.VSlider(
                                v_model=("param_pos_z",),
                                min=-5, max=5, step=0.1,
                                label="Z", color="cyan", dense=True
                            )
                    vuetify.VListSubheader("Поворот (град)", class_="px-0")
                    with vuetify.VRow(no_gutters=True):
                        with vuetify.VCol(cols=12):
                            vuetify.VSlider(
                                v_model=("param_rot_x",),
                                min=-180, max=180, step=1,
                                label="X", color="amber", dense=True
                            )
                            vuetify.VSlider(
                                v_model=("param_rot_y",),
                                min=-180, max=180, step=1,
                                label="Y", color="amber", dense=True
                            )
                            vuetify.VSlider(
                                v_model=("param_rot_z",),
                                min=-180, max=180, step=1,
                                label="Z", color="amber", dense=True
                            )

                    vuetify.VDivider(class_="my-4")
                    with vuetify.VContainer(v_if=("selected_object_type === 'lens'",), class_="pa-0"):
                        vuetify.VListSubheader("Параметры линзы", class_="px-0")
                        vuetify.VSlider(
                            v_model=("param_n",),
                            min=1.0, max=2.5, step=0.01,
                            label="Показатель преломления", color="purple", dense=True
                        )
                        vuetify.VSlider(
                            v_model=("param_R1",),
                            min=-20, max=20, step=0.5,
                            label="R1 (передняя)", color="purple", dense=True
                        )
                        vuetify.VSlider(
                            v_model=("param_R2",),
                            min=-20, max=20, step=0.5,
                            label="R2 (задняя)", color="purple", dense=True
                        )
                        vuetify.VSlider(
                            v_model=("param_thickness",),
                            min=0.1, max=5.0, step=0.1,
                            label="Толщина", color="purple", dense=True
                        )
                        vuetify.VSlider(
                            v_model=("param_edge_radius",),
                            min=0.1, max=5.0, step=0.1,
                            label="Радиус апертуры", color="purple", dense=True
                        )

                    with vuetify.VContainer(v_if=("selected_object_type === 'emitter'",), class_="pa-0"):
                        vuetify.VListSubheader("Параметры источника", class_="px-0")
                        vuetify.VSlider(
                            v_model=("param_num_rays",),
                            min=1, max=20, step=1,
                            label="Количество лучей", color="green", dense=True
                        )
                        vuetify.VSlider(
                            v_model=("param_min_offset",),
                            min=-3.0, max=0.0, step=0.1,
                            label="Мин. смещение", color="green", dense=True
                        )
                        vuetify.VSlider(
                            v_model=("param_max_offset",),
                            min=0.0, max=3.0, step=0.1,
                            label="Макс. смещение", color="green", dense=True
                        )
                        vuetify.VSlider(
                            v_model=("param_wavelength",),
                            min=380, max=780, step=10,
                            label="Длина волны (нм)", color="green", dense=True
                        )

                    vuetify.VDivider(class_="my-4")
                    vuetify.VSelect(
                        v_model=("trace_mode",),
                        items=["simple", "tree"],
                        label="Режим трассировки",
                        dense=True,
                        class_="mt-2"
                    )

                # Правая колонка: 3D-вид
                with vuetify.VCol(cols=9, class_="fill-height", style="height: 100%; position: relative;"):
                    with vuetify.VContainer(
                        fluid=True,
                        classes="pa-0 fill-height",
                        style="height: 100%; position: relative;",
                        __properties=["onClick"],
                        onClick="""() => {
                            const canvas = document.querySelector('.vtk-container canvas');
                            if (canvas) {
                                const rect = canvas.getBoundingClientRect();
                                const x = event.clientX - rect.left;
                                const y = rect.bottom - event.clientY;
                                trame.state.pick_coords = [x, y];
                            }
                        }"""
                    ):
                        html_view = trame_vtk.VtkLocalView(app.plotter.render_window)
                        # Связываем методы обновления
                        app.ctrl.view_update = html_view.update
                        app.ctrl.view_reset_camera = html_view.reset_camera

# Функция удаления объекта (для вызова из шаблона)
@server.controller.trigger("remove_object")
def remove_object(obj_id):
    app.remove_object(obj_id)

# Теперь, когда view_update связан, обновляем сцену
app.update_scene()
app.plotter.view_isometric()

if __name__ == "__main__":
    server.start(host="127.0.0.1", port=8085)
