import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")

import json
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
    RayTracer, RayPool, UniversalLens, BeamEmitter, Ray, SimpleMode, TreeMode, MeshSurface, HyperbolicLens
)

# Режим отрисовки: "client" или "server"
RENDER_MODE = "client"   # или "client"
pv.OFF_SCREEN = True

class OpticsAppController:
    def __init__(self, server):
        self.server = server
        self.state = server.state
        self.ctrl = server.controller
        self.manual_rays = []

        self.plotter = pv.Plotter(off_screen=True)
        self.plotter.set_background("#1a1a2e")
        self.plotter.add_axes(color="white")
        self.plotter.enable_terrain_style()
        self.plotter.camera.roll = 0

        self.temp_plotter = pv.Plotter(off_screen=True)
        self.pool = RayPool(initial_size=0)
        self.scene_objects = []
        self.object_counter = 0
        self.initializing = True
        self._updating = False

        self.ray_tracer = RayTracer(
            self.temp_plotter,
            mode="simple",
            pool=self.pool,
            line_width=2.5,
            min_alpha=0.05,
            gamma=1,
        )
        self.ray_tracer.mode.energy_color_type = 0
        self.ray_tracer.mode.max_bounces = 100

        self._debounce_delay = 0.01  # Уменьшено для более плавного отклика при оптимизации
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
        # self.state.param_R_curvature = 5.0
        self.state.param_f_target = 10.0
        self.state.param_edge_radius = 1.0
        self.state.param_num_rays = 5
        self.state.param_min_offset = -0.5
        self.state.param_max_offset = 0.5
        self.state.param_wavelength = 550.0
        self.state.param_mesh_path = ""

        self.ctrl.trigger("delete_object_event")(self.remove_object)

        self.state.change("selected_object_id")(self.on_object_selected)
        self.state.change("trace_mode")(self.on_trace_mode_changed)

        self.plotter.add_mesh(pv.PolyData(), name="traced_rays_geometry")

        self.create_initial_objects()
        self.initializing = False

    def _update_objects_list_state(self):
        self.state.scene_objects_list = [
            {"id": o["id"], "name": o["name"], "type": o["type"]}
            for o in self.scene_objects
        ]

    def create_initial_objects(self):
        # self.add_object("lens", "Линза 1", {
        #     "origin": (-2.0, 0.0, 0.0), "rotation": (0, 0, 0),
        #     "R1": 5.0, "R2": -5.0, "thickness": 0.5, "edge_radius": 1.0, "n": 1.5,
        #     "reflection_range": (0, np.inf), "refraction_range": (0, np.inf), "absorption_range": None
        # })
        # self.add_object("lens", "Линза 2", {
        #     "origin": (2.0, 0.0, 0.0), "rotation": (0, 0, 45),
        #     "R1": 5.0, "R2": 2.0, "thickness": 0.5, "edge_radius": 1.0, "n": 1.5,
        #     "reflection_range": (0, np.inf), "refraction_range": (0, np.inf), "absorption_range": None
        # })
        # self.add_object("mesh", f"Меш {len(self.scene_objects) + 1}", {
        #     "origin": (10, 4, -16), "rotation": (0, -90, 0),
        #     "mesh_path": "Models/provod.stl",  # Значение по умолчанию или путь к вашей модели
        #     "n": 1.5,
        #     "reflection_range": (0, np.inf), "absorption_range": None
        # })

        self.add_object("hyperbolic_lens", f"Гиперб. линза {len(self.scene_objects) + 1}", {
            "origin": (0, 0, 0), "rotation": (0, 0, 0), "radius_of_curvature": -0.5,
            "thickness": 0.5, "edge_radius": 0.75, "n": 1.5,
            "reflection_range": (0, np.inf), "refraction_range": (0, np.inf), "absorption_range": None
        })

        for y in np.linspace(-1, 0.8, 50):
            self.manual_rays.append(
                Ray(origin=(-5.0, y, 0.0), direction=(1, 0, 0), energy=1.0, color="yellow", wavelength=550)
            )

        if self.scene_objects:
            first_id = self.scene_objects[0]["id"]
            self.state.selected_object_id = first_id
            self.on_object_selected(first_id)
        self._update_objects_list_state()

    def add_object(self, obj_type, name, params):
        self.object_counter += 1
        obj_id = f"obj_{self.object_counter}"
        instance = self._create_instance(obj_type, params)

        obj_entry = {
            "id": obj_id, "type": obj_type, "name": name, "params": params, "instance": instance
        }
        self.scene_objects.append(obj_entry)

        # Первичная отрисовка нового объекта на сцене.
        # Меш уже содержит в себе origin и rotation из main.py, поэтому
        # свойства actor.position/orientation не трогаем (они остаются 0, 0, 0)
        if isinstance(instance, (UniversalLens, HyperbolicLens)):
            self.plotter.add_mesh(
                instance.get_mesh(),
                color="magenta" if obj_type == "hyperbolic_lens" else "cyan",
                opacity=0.5,
                smooth_shading=True,
                name=obj_id
            )
        elif isinstance(instance, BeamEmitter):
            self.plotter.add_mesh(
                instance.get_mesh(),
                color="green",
                name=obj_id
            )
        elif isinstance(instance, MeshSurface):
            self.plotter.add_mesh(
                instance.get_mesh(),
                color="Blue",
                opacity=0.6,
                smooth_shading=True,
                name=obj_id
            )

        if not self.initializing:
            self._update_objects_list_state()
            self.state.selected_object_id = obj_id
            self.on_object_selected(obj_id)
            self.update_scene()
        return obj_id

    def add_lens_click(self):
        self.add_object("lens", f"Линза {len(self.scene_objects) + 1}", {
            "origin": (0, 0, 0), "rotation": (0, 0, 0), "R1": 5.0, "R2": -5.0,
            "thickness": 0.5, "edge_radius": 1.0, "n": 1.5,
            "reflection_range": (0, np.inf), "refraction_range": (0, np.inf), "absorption_range": None
        })

    def add_hyperbolic_lens_click(self):
        """Метод обработки добавления гиперболической линзы."""
        self.add_object("hyperbolic_lens", f"Гиперб. линза {len(self.scene_objects) + 1}", {
            "origin": (0, 0, 0), "rotation": (0, 0, 0), "radius_of_curvature": 5.0,
            "thickness": 0.5, "edge_radius": 1.0, "n": 1.5, "f_target": 10.0,
            "reflection_range": (0, np.inf), "refraction_range": (0, np.inf), "absorption_range": None
        })

    def add_emitter_click(self):
        self.add_object("emitter", f"Источник {len(self.scene_objects) + 1}", {
            "origin": (0, 0, 0), "rotation": (0, 0, 0), "num_rays": 5, "min_offset": -0.5,
            "max_offset": 0.5, "wavelength": 550.0, "color": "yellow", "energy": 1.0, "current_n": 1.0
        })

    def add_mesh_click(self):
        self.add_object("mesh", f"Меш {len(self.scene_objects) + 1}", {
            "origin": (0, 0, 0), "rotation": (0, 0, 0),
            "mesh_path": "Models/Prism.stl",  # Значение по умолчанию или путь к вашей модели
            "n": 1.5,
            "reflection_range": None, "refraction_range": (0, np.inf), "absorption_range": None
        })

    def compute_radius_from_n_click(self):
        """Кнопка: Считать Радиус по заданному N и Фокусу"""
        f = float(self.state.param_f_target)
        n = float(self.state.param_n)
        # Рассчитываем и обновляем ползунок радиуса в UI
        new_R = HyperbolicLens.calculate_radius_by_n(f, n)
        self.state.param_curvature = round(new_R, 3)

    def compute_n_from_radius_click(self):
        """Кнопка: Считать N по заданному Радиусу и Фокусу"""
        f = float(self.state.param_f_target)
        R_curv = float(self.state.param_R_curvature)
        # Рассчитываем и обновляем ползунок преломления в UI
        new_n = HyperbolicLens.calculate_n_by_radius(f, R_curv)
        # Ограничиваем разумными пределами для слайдера (1.0 - 2.5)
        self.state.param_n = round(max(1.0, min(2.5, new_n)), 3)

    def remove_object(self, *args, **kwargs):
        if args and isinstance(args[0], list):
            obj_id = args[0][0]
        elif args:
            obj_id = args[0]
        else:
            return

        # Удаляем актера с графика PyVista, чтобы он не висел в памяти
        self.plotter.remove_actor(obj_id)
        self.scene_objects = [o for o in self.scene_objects if o["id"] != obj_id]

        if self.state.selected_object_id == obj_id:
            next_id = self.scene_objects[0]["id"] if self.scene_objects else None
            self.state.selected_object_id = next_id
            if next_id:
                self.on_object_selected(next_id)
            else:
                self.state.selected_object_type = None

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
        elif obj_type == "hyperbolic_lens":
            return HyperbolicLens(
                origin=params.get("origin", (0, 0, 0)), rotation_degrees=params.get("rotation", (0, 0, 0)),
                thickness=params.get("thickness", 0.5), edge_radius=params.get("edge_radius", 1.0),
                n=params.get("n", 1.5), f_target=params.get("f_target", 10.0),
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
        elif obj_type == "mesh":
            return MeshSurface(
                mesh=params.get("mesh_path", ""),
                rotation_degrees=params.get("rotation", (0, 0, 0)),
                translation=params.get("origin", (0, 0, 0)),
                n_inside=params.get("n", 1.5),
                reflection_range=params.get("reflection_range", (0, np.inf)),
                # refraction_range=params.get("refraction_range", (0, np.inf)),
                absorption_range=params.get("absorption_range")
            )

    def on_trace_mode_changed(self, **kwargs):
        mode = self.state.trace_mode

        if mode == "tree":
            self.ray_tracer.set_mode("tree")
            # Увеличиваем глубину дерева рекурсии для прохода через много линз
            self.ray_tracer.mode.max_depth = 30
            self.ray_tracer.mode.total_limit = 1000
        else:
            self.ray_tracer.set_mode("simple")
            self.ray_tracer.mode.max_bounces = 100
            self.ray_tracer.mode.offset_distance = 0.01

        # Принудительно пересчитываем лучи для нового режима трассировки
        self.update_scene()

    def update_scene(self):
        """ОПТИМИЗИРОВАНО: Перестраивает геометрию лучей с учетом их энергии и прозрачности."""
        if self._updating: return
        self._updating = True
        try:
            self.ray_tracer.elements.clear()
            self.ray_tracer.rays.clear()

            # 1. Собираем оптические поверхности для трассировки
            for obj_entry in self.scene_objects:
                instance = obj_entry["instance"]
                if isinstance(instance, (UniversalLens, HyperbolicLens)):
                    for surf in instance.get_surfaces():
                        self.ray_tracer.add_elements(surf)
                elif isinstance(instance, MeshSurface):
                    self.ray_tracer.add_elements(instance)

            for ray in self.manual_rays:
                self.ray_tracer.add_ray(ray)

            # 2. Математический расчет лучей (возвращает список объектов Segment)
            segments = self.ray_tracer.trace_all()

            # 3. ИСПОЛЬЗУЕМ КЛАСС RayCloud ИЗ main.py ДЛЯ РАСЧЕТА ПРОЗРАЧНОСТИ (ALPHA)
            # Вызываем метод update у встроенного в ray_tracer объекта cloud.
            # Он самостоятельно запишет массив "colors" (RGBA) в свой внутренний меш.
            energy_type = self.ray_tracer.mode.energy_color_type
            self.ray_tracer.cloud.update(segments, energy_color_type=energy_type)

            # Извлекаем правильно сформированный меш с RGBA-информацией из RayCloud
            calculated_ray_mesh = self.ray_tracer.cloud.actor.mapper.dataset

            # 4. ОБНОВЛЕНИЕ UI-АКТОРА С АКТИВАЦИЕЙ ПРОЗРАЧНОСТИ
            if "traced_rays_geometry" in self.plotter.actors:
                # Удаляем старый актор, чтобы сбросить жесткий кэш непрозрачности на клиенте
                self.plotter.remove_actor("traced_rays_geometry")

            # Если меш содержит данные, добавляем его на сцену с поддержкой RGBA
            if calculated_ray_mesh and calculated_ray_mesh.n_points > 0:
                # Явно выставляем имя массива активных скаляров
                calculated_ray_mesh.active_scalars_name = "colors"

                self.plotter.add_mesh(
                    calculated_ray_mesh,
                    scalars="colors",  # Читаем цвета вершин
                    rgba=True,  # Активируем чтение альфа-канала (прозрачности)
                    opacity="linear",  # Отключаем дефолтные маски, используем чистый альфа-канал
                    line_width=4,
                    render_lines_as_tubes=False,
                    name="traced_rays_geometry"  # Регистрируем под тем же именем
                )
            self.plotter.render()
            if hasattr(self.ctrl, 'view_update'):
                self.ctrl.view_update()
        finally:
            self._updating = False

    def on_object_selected(self, *args, **kwargs):
        obj_id = args[0] if args else self.state.selected_object_id
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
        elif obj_entry["type"] == "hyperbolic_lens":
            p["n"] = float(self.state.param_n)
            p["f_target"] = float(self.state.param_f_target)
            p["thickness"] = float(self.state.param_thickness)
            p["edge_radius"] = float(self.state.param_edge_radius)
        elif obj_entry["type"] == "emitter":
            self.state.param_num_rays = int(p.get("num_rays", 5))
            self.state.param_min_offset = float(p.get("min_offset", -0.5))
            self.state.param_max_offset = float(p.get("max_offset", 0.5))
            self.state.param_wavelength = float(p.get("wavelength", 550.0))
        elif obj_entry["type"] == "mesh":
            self.state.param_n = float(p.get("n", 1.5))
            self.state.param_mesh_path = str(p.get("mesh_path", ""))

    def update_selected_object(self, *args, **kwargs):
        """Финальное исправление: устраняет преломление 'в воздухе' при вращении по 2+ осям.
        Синхронизирует внутренний порядок осей Эйлера с логикой main.py.
        """
        obj_id = self.state.selected_object_id
        obj_entry = self._find_object(obj_id)
        if not obj_entry: return

        p = obj_entry["params"]

        # Получаем целевые абсолютные значения из UI
        new_origin = np.array(
            [float(self.state.param_pos_x), float(self.state.param_pos_y), float(self.state.param_pos_z)])
        new_rotation = np.array(
            [float(self.state.param_rot_x), float(self.state.param_rot_y), float(self.state.param_rot_z)])

        # Проверяем, изменились ли конструктивные параметры формы (радиусы, толщина)
        shape_changed = False
        if obj_entry["type"] == "lens":
            if (p["n"] != float(self.state.param_n) or
                    p["R1"] != float(self.state.param_R1) or
                    p["R2"] != float(self.state.param_R2) or
                    p["thickness"] != float(self.state.param_thickness) or
                    p["edge_radius"] != float(self.state.param_edge_radius)):
                shape_changed = True
        elif obj_entry["type"] == "emitter":
            if (p["num_rays"] != int(self.state.param_num_rays) or
                    p["min_offset"] != float(self.state.param_min_offset) or
                    p["max_offset"] != float(self.state.param_max_offset) or
                    p["wavelength"] != float(self.state.param_wavelength)):
                shape_changed = True
        elif obj_entry["type"] == "mesh":
            if (p["n"] != float(self.state.param_n) or
                    p["mesh_path"] != str(self.state.param_mesh_path)):
                shape_changed = True

        # Сохраняем новые параметры в словарь состояния объекта
        p["origin"] = tuple(new_origin)
        p["rotation"] = tuple(new_rotation)
        if obj_entry["type"] == "lens":
            p["n"] = float(self.state.param_n)
            p["R1"] = float(self.state.param_R1)
            p["R2"] = float(self.state.param_R2)
            p["thickness"] = float(self.state.param_thickness)
            p["edge_radius"] = float(self.state.param_edge_radius)
        elif obj_entry["type"] == "hyperbolic_lens":
            p["n"] = float(self.state.param_n)
            # p["radius_of_curvature"] = float(self.state.param_R_curvature)
            p["f_target"] = float(self.state.param_f_target)  # Прямая запись без формул пересчета
            p["thickness"] = float(self.state.param_thickness)
            p["edge_radius"] = float(self.state.param_edge_radius)
        elif obj_entry["type"] == "emitter":
            p["num_rays"] = int(self.state.param_num_rays)
            p["min_offset"] = float(self.state.param_min_offset)
            p["max_offset"] = float(self.state.param_max_offset)
            p["wavelength"] = float(self.state.param_wavelength)
        elif obj_entry["type"] == "mesh":
            p["n"] = float(self.state.param_n)
            p["mesh_path"] = str(self.state.param_mesh_path)

        # РЕШЕНИЕ БАГА ПРЕЛОМЛЕНИЯ В ВОЗДУХЕ:
        # Вместо постепенного накопления дельт мы ВСЕГДА генерируем чистый инстанс в нулевых координатах
        # и с нулевым поворотом. После этого мы ОДНИМ вызовом .translate() и .rotate()
        # приводим его к целевому виду. Это гарантирует, что и математические поверхности,
        # и get_mesh() повернутся по абсолютно одинаковому математическому закону main.py.

        # 1. Создаем базовый инстанс в нуле (с учетом возможных изменений R1, R2, толщины)
        base_params = p.copy()
        base_params["origin"] = (0.0, 0.0, 0.0)
        base_params["rotation"] = (0.0, 0.0, 0.0)
        instance = self._create_instance(obj_entry["type"], base_params)

        # 2. Применяем абсолютный поворот встроенным методом из main.py
        if np.any(new_rotation != 0):
            instance.rotate(new_rotation)

        # 3. Применяем абсолютное смещение встроенным методом из main.py
        if np.any(new_origin != 0):
            instance.translate(new_origin)

        # Сохраняем собранный инстанс в память сцены
        obj_entry["instance"] = instance

        # 4. Обновляем визуальную 3D-модель объекта на сцене PyVista
        if obj_id in self.plotter.actors:
            self.plotter.remove_actor(obj_id)

            if obj_entry["type"] in ["lens", "hyperbolic_lens"]:
                self.plotter.add_mesh(
                    obj_entry["instance"].get_mesh(),
                    color="magenta" if obj_entry["type"] == "hyperbolic_lens" else "cyan",
                    opacity=0.5,
                    smooth_shading=True,
                    name=obj_id
                )
            elif obj_entry["type"] == "emitter":
                self.plotter.add_mesh(
                    obj_entry["instance"].get_mesh(),
                    color="green",
                    name=obj_id
                )
            elif obj_entry["type"] == "mesh":
                self.plotter.add_mesh(
                    obj_entry["instance"].get_mesh(),
                    color="blue",
                    opacity=0.6,
                    smooth_shading=True,
                    name=obj_id
                )

        # Пересчитываем трассировку лучей для новой сцены
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
              "param_n", "param_R1", "param_R2", "param_f_target", "param_thickness", "param_edge_radius", "param_num_rays",
              "param_min_offset", "param_max_offset", "param_wavelength", "param_mesh_path"]:
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
                        vuetify.VBtn("Добавить сферич. линзу", color="cyan", block=True, click=app.add_lens_click)
                        vuetify.VBtn("Добавить гиперб. линзу", color="magenta", block=True, click=app.add_hyperbolic_lens_click)
                        vuetify.VBtn("Добавить источник", color="green", block=True, class_="mt-2",
                                     click=app.add_emitter_click)
                        vuetify.VBtn("Добавить 3D-меш", color="yellow", block=True, class_="mt-2",
                                     click=app.add_mesh_click)
                    vuetify.VDivider(class_="my-4")
                    with vuetify.VCard(flat=True, color="transparent", style="max-height: 200px; overflow-y: auto;"):
                        with vuetify.VList(dense=True, nav=True):
                            with vuetify.VListItem(v_for="obj in scene_objects_list",
                                                   key="obj.id",
                                                   active="selected_object_id == obj.id",
                                                   click="selected_object_id = obj.id",
                                                   style="cursor: pointer;"):
                                vuetify.VIcon("mdi-cube-outline", small=True, class_="mr-2")
                                vuetify.VListItemTitle("{{ obj.name }}")
                                with vuetify.Template(v_slot_append=True):
                                    vuetify.VBtn(icon="mdi-delete",
                                                 size="x-small",
                                                 variant="text",
                                                 color="red",
                                                 click="trigger('delete_object_event', [obj.id])")

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

                    with vuetify.VContainer(v_if="selected_object_type == 'lens'", class_="pa-0"):
                        vuetify.VListSubheader("Параметры сферич. линзы", class_="px-0")
                        vuetify.VSlider(v_model="param_n", min=1.0, max=2.5, step=0.01,
                                        label="Показатель преломления", dense=True)
                        vuetify.VSlider(v_model="param_R1", min=-20, max=20, step=0.5,
                                        label="R1 (передняя)", dense=True)
                        vuetify.VSlider(v_model="param_R2", min=-20, max=20, step=0.5,
                                        label="R2 (задняя)", dense=True)
                        vuetify.VSlider(v_model="param_thickness", min=0.1, max=5.0, step=0.1,
                                        label="Толщина", dense=True)
                        vuetify.VSlider(v_model="param_edge_radius", min=0.1, max=5.0, step=0.1,
                                        label="Радиус апертуры", dense=True)

                    with vuetify.VContainer(v_if="selected_object_type == 'hyperbolic_lens'", class_="pa-0"):
                        vuetify.VListSubheader("Параметры гиперболической линзы", class_="px-0")
                        vuetify.VSlider(v_model="param_thickness", min=0.1, max=5.0, step=0.1,
                                        label="Толщина", dense=True)
                        vuetify.VSlider(v_model="param_edge_radius", min=0.1, max=5.0, step=0.1,
                                        label="Радиус апертуры", dense=True)
                        vuetify.VSlider(v_model="param_f_target", min=-30.0, max=30.0, step=0.1,
                                        label="Целевой фокус (f)", dense=True)
                        vuetify.VSlider(v_model="param_n", min=1.001, max=2.5, step=0.01,
                                        label="Показатель преломления (n)", dense=True)

                    with vuetify.VContainer(v_if="selected_object_type == 'emitter'", class_="pa-0"):
                        vuetify.VListSubheader("Параметры источника", class_="px-0")
                        vuetify.VSlider(v_model="param_num_rays", min=1, max=20, step=1,
                                        label="Количество лучей", dense=True)
                        vuetify.VSlider(v_model="param_min_offset", min=-3.0, max=0.0, step=0.1,
                                        label="Мин. смещение", dense=True)
                        vuetify.VSlider(v_model="param_max_offset", min=0.0, max=3.0, step=0.1,
                                        label="Макс. смещение", dense=True)
                        vuetify.VSlider(v_model="param_wavelength", min=380, max=780, step=10,
                                        label="Длина волны (нм)", dense=True)

                    with vuetify.VContainer(v_if="selected_object_type == 'mesh'", class_="pa-0"):
                        vuetify.VListSubheader("Параметры 3D-модели", class_="px-0")
                        vuetify.VTextField(v_model="param_mesh_path", label="Путь к файлу (.obj/.stl)", dense=True,
                                           class_="mb-2")
                        vuetify.VSlider(v_model="param_n", min=1.0, max=2.5, step=0.01, label="Показатель преломления",
                                        dense=True)

                    vuetify.VDivider(class_="my-4")
                    vuetify.VSelect(label="Режим трассировки", v_model="trace_mode", items=("trace_modes",),
                                    item_title="title", item_value="value", dense=True, class_="mt-2")

                # Правая колонка с 3D окном pyvista
                with vuetify.VCol(cols=9, style="height: 100%; overflow: hidden;"):

                    ui_view = plotter_ui(
                        app.plotter,
                        mode="client",
                        add_menu=False,
                        image_scale=1,
                        interactor_style="Terrain"
                    )
                    app.ctrl.view_update = ui_view.update

app.update_scene()

if __name__ == "__main__":
    server.start(host="127.0.0.1", port=8085)
