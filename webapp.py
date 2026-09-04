import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")

import io
import json
import asyncio
import numpy as np
import pyvista as pv
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3 as vuetify
from trame.widgets import vtk as trame_vtk
from scipy.spatial.transform import Rotation as R

# Для холста
from pyvista.trame.ui import plotter_ui
from trame_pyvista.widgets import PyVistaRemoteLocalView


from main import (
    RayTracer, RayPool, UniversalLens, BeamEmitter, Ray, SimpleMode, TreeMode, MeshSurface, HyperbolicLens,
    PlaneSurface, SphereSurface, CylinderSurface, DispersiveRay, WhiteRay
)

# Режим отрисовки: "client" или "server"
RENDER_MODE = "client"   # или "client"
pv.OFF_SCREEN = True

class OpticsAppController:
    def __init__(self, server):
        self.server = server
        self.state = server.state
        self.ctrl = server.controller

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
        self._loading_state = False

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

        self._debounce_delay = 0.01
        self._debounce_timer = None

        self.state.selected_object_id = None
        self.state.selected_object_type = None
        self.state.trace_mode = "simple"
        self.state.scene_objects_list = []
        self.state.render_mode = "local"
        self.state.trace_modes = [
            {"title": "Simple", "value": "simple"},
            {"title": "Tree", "value": "tree"},
        ]

        # Основные параметры
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
        self.state.param_f_target = 10.0
        self.state.param_edge_radius = 1.0
        self.state.param_num_rays = 5
        self.state.param_min_offset = -0.5
        self.state.param_max_offset = 0.5
        self.state.param_wavelength = 550.0
        self.state.param_current_n = 1.0
        self.state.param_mesh_path = ""
        self.state.param_scale_uniform = True  # Тумблер по умолчанию ВКЛ
        self.state.param_scale_all = 1.0  # Единый ползунок
        self.state.param_scale_x = 1.0  # Раздельные ползунки
        self.state.param_scale_y = 1.0
        self.state.param_scale_z = 1.0
        self.state.param_plane_shape = "rectangle"
        self.state.param_plane_width = 2.0
        self.state.param_plane_height = 2.0
        self.state.param_cylinder_capping = False
        self.state.param_ray_class = "Ray"  # Тип по умолчанию строкой для интерфейса
        self.state.param_num_spectral_bands = 7
        self.state.ray_classes_list = [
            {"title": "Обычный луч (Ray)", "value": "Ray"},
            {"title": "Дисперсионный луч (DispersiveRay)", "value": "DispersiveRay"},
            {"title": "Белый свет (WhiteRay)", "value": "WhiteRay"},
        ]

        # НАЧАЛО ИЗМЕНЕНИЙ: Инициализация спектральных параметров в состоянии
        for effect in ["reflection", "refraction", "absorption"]:
            self.state[f"param_{effect}_min"] = 0.0
            self.state[f"param_{effect}_max"] = np.inf
            self.state[f"param_{effect}_enabled"] = (effect == "refraction") # преломление включено по умолчанию
        # КОНЕЦ ИЗМЕНЕНИЙ

        self.state.temp_pos_delta_x = 0.0
        self.state.temp_pos_delta_y = 0.0
        self.state.temp_pos_delta_z = 0.0

        self.state.temp_last_pos_x = 0.0
        self.state.temp_last_pos_y = 0.0
        self.state.temp_last_pos_z = 0.0

        self.ctrl.trigger("apply_x_delta")(self.apply_x_delta)
        self.ctrl.trigger("apply_y_delta")(self.apply_y_delta)
        self.ctrl.trigger("apply_z_delta")(self.apply_z_delta)

        self.ctrl.trigger("set_last_pos")(self.set_last_pos)

        self.ctrl.trigger("delete_object_event")(self.remove_object)
        self.ctrl.trigger("focus_on_object_event")(self.focus_on_selected_object)
        self.state.change("selected_object_id")(self.on_object_selected)
        self.state.change("trace_mode")(self.on_trace_mode_changed)

        self.plotter.add_mesh(pv.PolyData(), name="traced_rays_geometry")

        self.ctrl.trigger("save_scene_to_file")(self.save_scene_to_file)
        self.ctrl.trigger("load_scene_from_file")(self.load_scene_from_file)

        self.create_initial_objects()
        self.initializing = False

    # ---------- Вспомогательные методы ----------
    def _update_objects_list_state(self):
        self.state.scene_objects_list = [
            {"id": o["id"], "name": o["name"], "type": o["type"]}
            for o in self.scene_objects
        ]

    # ---------- Сохранение и загрузка сцены ----------
    def save_scene_to_file(self, filename):
        """Сохраняет параметры всех объектов сцены в JSON-файл в корне проекта."""
        if not filename:
            return

        # Гарантируем расширение .json
        if not filename.endswith(".json"):
            filename += ".json"

        # Формируем плоский словарь параметров для сохранения
        serialized_data = []
        for obj in self.scene_objects:
            # Копируем параметры, чтобы не испортить текущие
            save_params = obj["params"].copy()

            # Преобразуем numpy-типы и специфичные объекты (например, Infinity) в JSON-совместимые
            for key, val in save_params.items():
                if isinstance(val, tuple):
                    save_params[key] = list(val)
                elif isinstance(val, type):  # Для ray_class (например, WhiteRay)
                    save_params[key] = val.__name__
                elif isinstance(val, (list, tuple)) and len(val) == 2:  # Для оптических диапазонов
                    # Заменяем np.inf на строковое представление для JSON
                    save_params[key] = [
                        "Infinity" if v == np.inf or (isinstance(v, float) and np.isinf(v)) else v
                        for v in val
                    ]

            serialized_data.append({
                "name": obj["name"],
                "type": obj["type"],
                "params": save_params
            })

        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(serialized_data, f, ensure_ascii=False, indent=4)
            print(f"Сцена успешно сохранена в файл: {filename}")
        except Exception as e:
            print(f"Ошибка при сохранении сцены: {e}")

    def load_scene_from_file(self, filename):
        """Загружает сцену из JSON-файла, полностью очищая текущую сцену."""
        if not filename:
            return

        if not filename.endswith(".json"):
            filename += ".json"

        import os
        if not os.path.exists(filename):
            print(f"Файл {filename} не найден.")
            return

        try:
            with open(filename, "r", encoding="utf-8") as f:
                loaded_data = json.load(f)
        except Exception as e:
            print(f"Ошибка чтения файла сцены: {e}")
            return

        # 1. Полностью очищаем текущую сцену
        # Удаляем всех акторов из PyVista плоттера
        for obj in self.scene_objects:
            self.plotter.remove_actor(obj["id"])
        self.scene_objects.clear()
        self.object_counter = 0
        self.state.selected_object_id = None
        self.state.selected_object_type = None

        # Отключаем триггеры обновления на время массовой загрузки
        self._loading_state = True
        self.initializing = True

        try:
            # 2. Восстанавливаем объекты из файла
            for obj_data in loaded_data:
                params = obj_data["params"]

                # Приводим массивы обратно к кортежам, где это необходимо
                if "origin" in params: params["origin"] = tuple(params["origin"])
                if "rotation" in params: params["rotation"] = tuple(params["rotation"])

                # Восстанавливаем классы лучей
                if "ray_class" in params:
                    if params["ray_class"] == "WhiteRay":
                        params["ray_class"] = WhiteRay
                    else:
                        params["ray_class"] = Ray

                # Восстанавливаем оптические диапазоны (строку "Infinity" преобразуем обратно в np.inf)
                for effect in ["reflection", "refraction", "absorption"]:
                    range_key = f"{effect}_range"
                    if range_key in params and params[range_key] is not None:
                        r_min = params[range_key][0]
                        r_max = params[range_key][1]
                        if r_max == "Infinity":
                            r_max = np.inf
                        params[range_key] = (r_min, r_max)

                # Создаем объект на сцене
                self.add_object(obj_data["type"], obj_data["name"], params)

        except Exception as e:
            print(f"Ошибка при разборе данных сцены: {e}")
        finally:
            self._loading_state = False
            self.initializing = False

        # Синхронизируем интерфейс
        self._update_objects_list_state()
        if self.scene_objects:
            first_id = self.scene_objects[0]["id"]
            self.state.selected_object_id = first_id
            self.on_object_selected(first_id)

        self.update_scene()

    # ---------- Создание объектов ----------
    def create_initial_objects(self):
        # 1. Входящий пучок света (параллельные лучи слева направо)
        self.add_object("emitter", "Входящий свет", {
            "origin": (-10.0, 0.0, 0.0),
            "rotation": (0.0, 0.0, 0.0),
            "num_rays": 1,
            "min_offset": -1.2,
            "max_offset": 1.2,
            "wavelength": 550.0,
            "color": "yellow",
            "energy": 1.0,
            "current_n": 1.0,
            "ray_class": WhiteRay,
            "num_spectral_bands": 7
        })

        # 2. Главное зеркало (вогнутая сферическая поверхность справа)
        # Отражает лучи обратно к центру трубы
        self.add_object("sphere_surf", "Главное зеркало (Primary)", {
            "origin": (5.0, 0.0, 0.0),
            "rotation": (0.0, 0.0, 0.0),  # Развернуто навстречу лучам
            "radius": 15.0,  # Радиус кривизны зеркала
            "edge_radius": 1.5,  # Радиус апертуры
            "n": 1.0,
            "reflection_range": (0.0, np.inf),
            "refraction_range": None,
            "absorption_range": None
        })

        # 3. Вторичное плоское зеркало (наклонное под 45 градусов)
        # Перенаправляет лучи от главного зеркала вертикально вверх
        self.add_object("plane", "Вторичное зеркало (Secondary)", {
            "origin": (-2.0, 0.0, 0.0),
            "rotation": (0.0, 0.0, 45.0),  # Наклон 45 градусов для отражения вверх
            "edge_radius": 0.5,
            "shape_type": "circle",
            "width": 1.0,
            "height": 1.0,
            "n": 1.0,
            "reflection_range": (0.0, np.inf),
            "refraction_range": None,
            "absorption_range": None
        })

        # 4. Линза окуляра (Eyepiece Lens сверху)
        # Собирает перенаправленный свет в финальный пучок
        self.add_object("lens", "Окуляр (Eyepiece)", {
            "origin": (-2.0, 3.17, 0.0),
            "rotation": (0.0, 0.0, 90.0),  # Повернута горизонтально (вдоль оси Y)
            "R1": 4.0,
            "R2": 2.13,
            "thickness": 0.2,
            "edge_radius": 0.6,
            "n": 1.5,
            "reflection_range": None,
            "refraction_range": (0.0, np.inf),
            "absorption_range": None
        })

        # if self.scene_objects:
        #     first_id = self.scene_objects[0]["id"]
        #     self.state.selected_object_id = first_id
        #     self.on_object_selected(first_id)
        self._update_objects_list_state()

    def add_object(self, obj_type, name, params):
        self.object_counter += 1
        obj_id = f"obj_{self.object_counter}"
        instance = self._create_instance(obj_type, params)

        obj_entry = {
            "id": obj_id, "type": obj_type, "name": name, "params": params, "instance": instance
        }
        self.scene_objects.append(obj_entry)

        if isinstance(instance, (UniversalLens, HyperbolicLens)):
            self.plotter.add_mesh(
                instance.get_mesh(),
                color="cyan",
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
                color="cyan",
                opacity=0.6,
                smooth_shading=True,
                name=obj_id
            )
        elif obj_type in ["plane", "sphere_surf", "cylinder_surf"]:
            self.plotter.add_mesh(
                instance.get_mesh(),
                color="cyan",
                opacity=1,
                smooth_shading=True,
                name=obj_id
            )

        if not self.initializing:
            self._update_objects_list_state()
            self.state.selected_object_id = obj_id
            self.on_object_selected(obj_id)
            self.update_scene()
        return obj_id

    def add_plane_click(self):
        self.add_object("plane", f"Плоскость {len(self.scene_objects) + 1}", {
            "origin": (0, 0, 0), "rotation": (0, 0, 0), "edge_radius": 1.5, "n": 1.0,
            "shape_type": "circle", "width": 2.0, "height": 2.0, # <-- Добавлено
            "reflection_range": None, "refraction_range": (0, np.inf), "absorption_range": None
        })

    def add_sphere_surf_click(self):
        self.add_object("sphere_surf", f"Сфера Поверхн. {len(self.scene_objects) + 1}", {
            "origin": (0, 0, 0), "rotation": (0, 0, 0), "radius": 3.0, "edge_radius": 1.5, "n": 1.5,
            "reflection_range": None, "refraction_range": (0, np.inf), "absorption_range": None
        })

    def add_cylinder_surf_click(self):
        self.add_object("cylinder_surf", f"Цилиндр Поверхн. {len(self.scene_objects) + 1}", {
            "origin": (0, 0, 0), "rotation": (0, 0, 0), "radius": 1.0, "half_length": 1.0, "n": 1.5,
            "capping": False, # <-- Добавлено
            "reflection_range": None, "refraction_range": (0, np.inf), "absorption_range": None
        })

    def add_lens_click(self):
        self.add_object("lens", f"Линза {len(self.scene_objects) + 1}", {
            "origin": (0, 0, 0), "rotation": (0, 0, 0), "R1": 5.0, "R2": -5.0,
            "thickness": 0.5, "edge_radius": 1.0, "n": 1.5,
            "reflection_range": (0, np.inf), "refraction_range": (0, np.inf), "absorption_range": None
        })

    def add_hyperbolic_lens_click(self):
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
            "mesh_path": "Models/Prism.stl",
            "n": 1.5,
            "reflection_range": None, "refraction_range": (0, np.inf), "absorption_range": None,
            "scale_uniform": True, "scale_all": 1.0, "scale_x": 1.0, "scale_y": 1.0, "scale_z": 1.0
        })

    # ---------- Удаление объекта ----------
    def remove_object(self, *args, **kwargs):
        if args and isinstance(args[0], list):
            obj_id = args[0][0]
        elif args:
            obj_id = args[0]
        else:
            return

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

    # ---------- Создание экземпляров ----------
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
            return BeamEmitter(
                origin=params.get("origin", (0, 0, 0)), direction=np.array([1.0, 0.0, 0.0]),
                num_rays=params.get("num_rays", 5),
                min_offset=params.get("min_offset", -0.5), max_offset=params.get("max_offset", 0.5),
                color=params.get("color", "yellow"), wavelength=params.get("wavelength", 550),
                energy=params.get("energy", 1.0), current_n=params.get("current_n", 1.0), pool=self.pool,
                ray_class=params.get("ray_class", Ray), num_spectral_bands=params.get("num_spectral_bands", 7)
            )
        elif obj_type == "mesh":
            if params.get("scale_uniform", True):
                s_val = float(params.get("scale_all", 1.0))
                s_factors = (s_val, s_val, s_val)
            else:
                s_factors = (
                    float(params.get("scale_x", 1.0)),
                    float(params.get("scale_y", 1.0)),
                    float(params.get("scale_z", 1.0))
                )

            # Передаем кортеж коэффициентов (X, Y, Z)
            return MeshSurface(
                mesh=params.get("mesh_path", ""),
                rotation_degrees=params.get("rotation", (0, 0, 0)),
                translation=params.get("origin", (0, 0, 0)),
                n_inside=params.get("n", 1.5),
                reflection_range=params.get("reflection_range"),
                refraction_range=params.get("refraction_range"),
                absorption_range=params.get("absorption_range"),
                scale_factors=s_factors
            )
        elif obj_type == "plane":
            return PlaneSurface(
                point=params.get("origin", (0, 0, 0)),
                rotation_degrees=params.get("rotation", (0, 0, 0)),
                n_inside=params.get("n", 1.0),
                edge_radius=params.get("edge_radius", 1.5),
                shape_type=params.get("shape_type", "circle"),  # <-- Добавлено
                width=params.get("width", 2.0),  # <-- Добавлено
                height=params.get("height", 2.0),  # <-- Добавлено
                reflection_range=params.get("reflection_range"),
                refraction_range=params.get("refraction_range", (0, np.inf)),
                absorption_range=params.get("absorption_range")
            )
        elif obj_type == "sphere_surf":
            return SphereSurface(
                radius=params.get("radius", 3.0),
                rotation_degrees=params.get("rotation", (0, 0, 0)),
                n_inside=params.get("n", 1.5),
                edge_radius=params.get("edge_radius", 1.5),
                reflection_range=params.get("reflection_range"),
                refraction_range=params.get("refraction_range"),
                absorption_range=params.get("absorption_range"),
                lens_origin=params.get("origin", (0, 0, 0))
            )
        elif obj_type == "cylinder_surf":
            return CylinderSurface(
                center=params.get("origin", (0, 0, 0)),
                axis_dir=np.array([1.0, 0.0, 0.0]),
                radius=params.get("radius", 1.0),
                half_length=params.get("half_length", 1.0),
                n_inside=params.get("n", 1.5),
                capping=params.get("capping", False),  # <-- Добавлено
                reflection_range=params.get("reflection_range"),
                refraction_range=params.get("refraction_range"),
                absorption_range=params.get("absorption_range")
            )

    # ---------- Трассировка ----------
    def on_trace_mode_changed(self, **kwargs):
        mode = self.state.trace_mode
        if mode == "tree":
            self.ray_tracer.set_mode("tree")
            self.ray_tracer.mode.max_depth = 30
            self.ray_tracer.mode.total_limit = 1000
        else:
            self.ray_tracer.set_mode("simple")
            self.ray_tracer.mode.max_bounces = 100
            self.ray_tracer.mode.offset_distance = 0.01
        self.update_scene()

    def update_scene(self):
        if self._updating: return
        self._updating = True
        try:
            self.ray_tracer.elements.clear()
            self.ray_tracer.rays.clear()

            if hasattr(self.ray_tracer, 'emitters'):
                self.ray_tracer.emitters.clear()

            for obj_entry in self.scene_objects:
                instance = obj_entry["instance"]
                if isinstance(instance, (UniversalLens, HyperbolicLens)):
                    for surf in instance.get_surfaces():
                        self.ray_tracer.add_elements(surf)
                elif isinstance(instance, MeshSurface):
                    self.ray_tracer.add_elements(instance)
                elif isinstance(instance, (PlaneSurface, SphereSurface, CylinderSurface, MeshSurface)):
                    self.ray_tracer.add_elements(instance)
                elif isinstance(instance, BeamEmitter):
                    self.ray_tracer.add_emitter(instance)


            segments = self.ray_tracer.trace_all()
            energy_type = self.ray_tracer.mode.energy_color_type
            self.ray_tracer.cloud.update(segments, energy_color_type=energy_type)
            calculated_ray_mesh = self.ray_tracer.cloud.actor.mapper.dataset

            if "traced_rays_geometry" in self.plotter.actors:
                self.plotter.remove_actor("traced_rays_geometry")

            if calculated_ray_mesh and calculated_ray_mesh.n_points > 0:
                calculated_ray_mesh.active_scalars_name = "colors"
                self.plotter.add_mesh(
                    calculated_ray_mesh,
                    scalars="colors",
                    rgba=True,
                    opacity="linear",
                    line_width=4,
                    render_lines_as_tubes=False,
                    name="traced_rays_geometry"
                )
            self.plotter.render()
            if hasattr(self.ctrl, 'view_update'):
                self.ctrl.view_update()
        finally:
            self._updating = False

    # ---------- Выбор объекта ----------
    def on_object_selected(self, *args, **kwargs):
        obj_id = args[0] if args else self.state.selected_object_id
        if not obj_id:
            return
        obj_entry = self._find_object(obj_id)
        if obj_entry:
            self._loading_state = True  # Блокируем триггеры update
            try:
                self._load_params_to_state(obj_entry)
            finally:
                self._loading_state = False  # Снимаем блокировку

    def focus_on_selected_object(self, *args, **kwargs):
        """Перемещает фокус камеры на текущие координаты выбранного объекта."""
        obj_id = self.state.selected_object_id
        if not obj_id:
            return

        obj_entry = self._find_object(obj_id)
        if not obj_entry:
            return

        # Считываем текущую мировую позицию объекта из его сохраненных параметров
        p = obj_entry["params"]
        origin = p.get("origin", (0.0, 0.0, 0.0))

        # Переносим фокальную точку камеры PyVista в центр объекта
        target_point = (float(origin[0]), float(origin[1]), float(origin[2]))

        # fly_to плавно смещает фокус камеры в эту точку
        self.plotter.fly_to(target_point)

        # Принудительно рендерим кадр и обновляем клиентский вид
        self.plotter.render()
        if hasattr(self.ctrl, 'view_update'):
            self.ctrl.view_update()

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
        # Сбрасываем временные смещения при выборе объекта
        self.state.temp_pos_delta_x = 0.0
        self.state.temp_pos_delta_y = 0.0
        self.state.temp_pos_delta_z = 0.0
        if obj_entry["type"] == "lens":
            self.state.param_n = float(p.get("n", 1.5))
            self.state.param_R1 = float(p.get("R1", 5.0))
            self.state.param_R2 = float(p.get("R2", -5.0))
            self.state.param_thickness = float(p.get("thickness", 0.5))
            self.state.param_edge_radius = float(p.get("edge_radius", 1.0))
        elif obj_entry["type"] == "hyperbolic_lens":
            self.state.param_n = float(p.get("n", 1.5))
            self.state.param_f_target = float(p.get("f_target", 10.0))
            self.state.param_thickness = float(p.get("thickness", 0.5))
            self.state.param_edge_radius = float(p.get("edge_radius", 1.0))
        elif obj_entry["type"] == "emitter":
            self.state.param_num_rays = int(p.get("num_rays", 5))
            self.state.param_min_offset = float(p.get("min_offset", -0.5))
            self.state.param_max_offset = float(p.get("max_offset", 0.5))
            self.state.param_wavelength = float(p.get("wavelength", 550.0))
            self.state.param_current_n = float(p.get("current_n", 1.0))

            # Восстанавливаем строковое имя класса луча для UI
            rc = p.get("ray_class", Ray)
            if rc.__name__ in ["Ray", "DispersiveRay", "WhiteRay"]:
                self.state.param_ray_class = rc.__name__
            else:
                self.state.param_ray_class = "Ray"

            self.state.param_num_spectral_bands = int(p.get("num_spectral_bands", 7))
        elif obj_entry["type"] == "mesh":
            self.state.param_n = float(p.get("n", 1.5))
            self.state.param_mesh_path = str(p.get("mesh_path", ""))
            # Загружаем масштабы
            self.state.param_scale_uniform = bool(p.get("scale_uniform", True))
            self.state.param_scale_all = float(p.get("scale_all", 1.0))
            self.state.param_scale_x = float(p.get("scale_x", 1.0))
            self.state.param_scale_y = float(p.get("scale_y", 1.0))
            self.state.param_scale_z = float(p.get("scale_z", 1.0))
        elif obj_entry["type"] == "plane":
            self.state.param_n = float(p.get("n", 1.0))
            self.state.param_edge_radius = float(p.get("edge_radius", 1.5))
            # НАЧАЛО ИЗМЕНЕНИЙ
            self.state.param_plane_shape = str(p.get("shape_type", "circle"))
            self.state.param_plane_width = float(p.get("width", 2.0))
            self.state.param_plane_height = float(p.get("height", 2.0))
            # КОНЕЦ ИЗМЕНЕНИЙ
        elif obj_entry["type"] == "sphere_surf":
            self.state.param_radius = float(p.get("radius", 3.0))
            self.state.param_edge_radius = float(p.get("edge_radius", 1.5))
            self.state.param_n = float(p.get("n", 1.5))
        elif obj_entry["type"] == "cylinder_surf":
            self.state.param_radius = float(p.get("radius", 1.0))
            self.state.param_thickness = float(p.get("half_length", 1.0))
            self.state.param_n = float(p.get("n", 1.5))
            # НАЧАЛО ИЗМЕНЕНИЙ
            self.state.param_cylinder_capping = bool(p.get("capping", False))
            # КОНЕЦ ИЗМЕНЕНИЙ

        for effect in ["reflection", "refraction", "absorption"]:
            r_range = p.get(f"{effect}_range")
            if r_range is None:
                self.state[f"param_{effect}_enabled"] = False
                self.state[f"param_{effect}_min"] = 0.0
                self.state[f"param_{effect}_max"] = np.inf
            else:
                self.state[f"param_{effect}_enabled"] = True
                self.state[f"param_{effect}_min"] = float(r_range[0]) if r_range[0] is not None else 0.0
                self.state[f"param_{effect}_max"] = float(r_range[1]) if r_range[1] is not None else np.inf

    # ---------- Обновление выбранного объекта ----------
    def update_selected_object(self, *args, **kwargs):
        obj_id = self.state.selected_object_id
        obj_entry = self._find_object(obj_id)
        if not obj_entry: return

        p = obj_entry["params"]
        # Базовые координаты из параметров
        base_origin = np.array([float(self.state.param_pos_x),
                                float(self.state.param_pos_y),
                                float(self.state.param_pos_z)])
        # Временное смещение от слайдера
        # temp_shift = np.array([float(self.state.temp_pos_delta_x),
        #                        float(self.state.temp_pos_delta_y),
        #                        float(self.state.temp_pos_delta_z)])
        # Итоговая позиция = базовая + временное смещение
        new_origin = base_origin# + temp_shift

        new_rotation = np.array([float(self.state.param_rot_x),
                                 float(self.state.param_rot_y),
                                 float(self.state.param_rot_z)])

        shape_changed = False
        if obj_entry["type"] == "lens":
            if (p["n"] != float(self.state.param_n) or
                p["R1"] != float(self.state.param_R1) or
                p["R2"] != float(self.state.param_R2) or
                p["thickness"] != float(self.state.param_thickness) or
                p["edge_radius"] != float(self.state.param_edge_radius)):
                shape_changed = True
        elif obj_entry["type"] == "emitter":
            # Проверяем, изменился ли класс или количество спектральных полос
            current_rc_name = p.get("ray_class").__name__ if p.get("ray_class") else "Ray"
            if (p["num_rays"] != int(self.state.param_num_rays) or
                p["min_offset"] != float(self.state.param_min_offset) or
                p["max_offset"] != float(self.state.param_max_offset) or
                p["wavelength"] != float(self.state.param_wavelength) or
                p["current_n"] != float(self.state.param_current_n) or
                current_rc_name != str(self.state.param_ray_class) or
                p.get("num_spectral_bands", 7) != int(self.state.param_num_spectral_bands)):
                shape_changed = True
        elif obj_entry["type"] == "mesh":
            if (p["n"] != float(self.state.param_n) or
                p["mesh_path"] != str(self.state.param_mesh_path) or
                p.get("scale_uniform") != self.state.param_scale_uniform or
                p.get("scale_all") != float(self.state.param_scale_all) or
                p.get("scale_x") != float(self.state.param_scale_x) or
                p.get("scale_y") != float(self.state.param_scale_y) or
                p.get("scale_z") != float(self.state.param_scale_z)):
                shape_changed = True
        elif obj_entry["type"] == "plane":
            if (p["n"] != float(self.state.param_n) or
                p["edge_radius"] != float(self.state.param_edge_radius) or
                str(p.get("shape_type")) != str(self.state.param_plane_shape) or
                p.get("width") != float(self.state.param_plane_width) or
                p.get("height") != float(self.state.param_plane_height)):
                shape_changed = True
        elif obj_entry["type"] == "sphere_surf":
            if (p["radius"] != float(self.state.param_radius) or
                    p["edge_radius"] != float(self.state.param_edge_radius) or p["n"] != float(self.state.param_n)):
                shape_changed = True
        elif obj_entry["type"] == "cylinder_surf":
            if (p["radius"] != float(self.state.param_radius) or
                p["half_length"] != float(self.state.param_thickness) or
                p["n"] != float(self.state.param_n) or
                p.get("capping") != self.state.param_cylinder_capping):
                shape_changed = True

        for effect in ["reflection", "refraction", "absorption"]:
            if not self.state[f"param_{effect}_enabled"]:
                p[f"{effect}_range"] = None
            else:
                # Преобразуем строки/числа из полей ввода, учитывая строки "Infinity"
                try:
                    v_min = float(self.state[f"param_{effect}_min"])
                except (ValueError, TypeError):
                    v_min = 0.0

                try:
                    v_max = float(self.state[f"param_{effect}_max"])
                except (ValueError, TypeError):
                    v_max = np.inf

                p[f"{effect}_range"] = (v_min, v_max)

        # Обновляем параметры (сохраняем только базовую позицию, без temp)
        p["origin"] = tuple(base_origin)  # сохраняем базовую, а не new_origin
        p["rotation"] = tuple(new_rotation)
        if obj_entry["type"] == "lens":
            p["n"] = float(self.state.param_n)
            p["R1"] = float(self.state.param_R1)
            p["R2"] = float(self.state.param_R2)
            p["thickness"] = float(self.state.param_thickness)
            p["edge_radius"] = float(self.state.param_edge_radius)
        elif obj_entry["type"] == "hyperbolic_lens":
            p["n"] = float(self.state.param_n)
            p["f_target"] = float(self.state.param_f_target)
            p["thickness"] = float(self.state.param_thickness)
            p["edge_radius"] = float(self.state.param_edge_radius)
        elif obj_entry["type"] == "emitter":
            p["num_rays"] = int(self.state.param_num_rays)
            p["min_offset"] = float(self.state.param_min_offset)
            p["max_offset"] = float(self.state.param_max_offset)
            p["wavelength"] = float(self.state.param_wavelength)
            p["current_n"] = float(self.state.param_current_n)
            p["num_spectral_bands"] = int(self.state.param_num_spectral_bands)
            # Маппинг строки из UI в реальный класс
            mapping = {"Ray": Ray, "DispersiveRay": DispersiveRay, "WhiteRay": WhiteRay}
            p["ray_class"] = mapping.get(self.state.param_ray_class, Ray)
        elif obj_entry["type"] == "mesh":
            p["n"] = float(self.state.param_n)
            p["mesh_path"] = str(self.state.param_mesh_path)
            # Сохраняем новые свойства масштаба
            p["scale_uniform"] = self.state.param_scale_uniform
            p["scale_all"] = float(self.state.param_scale_all)
            p["scale_x"] = float(self.state.param_scale_x)
            p["scale_y"] = float(self.state.param_scale_y)
            p["scale_z"] = float(self.state.param_scale_z)
        elif obj_entry["type"] == "plane":
            p["n"] = float(self.state.param_n)
            p["edge_radius"] = float(self.state.param_edge_radius)
            p["shape_type"] = str(self.state.param_plane_shape)
            p["width"] = float(self.state.param_plane_width)
            p["height"] = float(self.state.param_plane_height)
        elif obj_entry["type"] == "sphere_surf":
            p["radius"] = float(self.state.param_radius)
            p["edge_radius"] = float(self.state.param_edge_radius)
            p["n"] = float(self.state.param_n)
        elif obj_entry["type"] == "cylinder_surf":
            p["radius"] = float(self.state.param_radius)
            p["half_length"] = float(self.state.param_thickness)
            p["n"] = float(self.state.param_n)
            p["capping"] = self.state.param_cylinder_capping

        # Пересоздаём инстанс с новыми параметрами (используем new_origin, т.к. это реальная позиция)
        base_params = p.copy()
        base_params["origin"] = (0.0, 0.0, 0.0)
        base_params["rotation"] = (0.0, 0.0, 0.0)
        instance = self._create_instance(obj_entry["type"], base_params)

        # Применяем поворот и перенос на итоговую позицию
        if np.any(new_rotation != 0):
            instance.rotate(new_rotation)
        instance.translate(new_origin)

        obj_entry["instance"] = instance

        # Обновляем визуализацию
        if obj_id in self.plotter.actors:
            self.plotter.remove_actor(obj_id)
            if obj_entry["type"] in ["lens", "hyperbolic_lens"]:
                self.plotter.add_mesh(
                    obj_entry["instance"].get_mesh(),
                    color="cyan",
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
                    color="cyan",
                    opacity=0.6,
                    smooth_shading=True,
                    name=obj_id
                )
            elif obj_entry["type"] in ["plane", "sphere_surf", "cylinder_surf"]:
                self.plotter.add_mesh(
                    obj_entry["instance"].get_mesh(),
                    color="cyan",
                    opacity=0.5,
                    smooth_shading=True,
                    name=obj_id
                )

        self.update_scene()

    # Обработчик изменения любого параметра (кроме temp)
    def on_param_change(self, *args, **kwargs):
        # Если изменился param_pos, сбрасываем соответствующий temp, чтобы избежать двойного учёта
        # Определяем, какой параметр изменился, через kwargs
        # В trame при изменении состояния передаётся имя в kwargs.get('key')
        if getattr(self, '_loading_state', False):
            return  # Если мы просто выбираем объект, ничего не делаем

        key = kwargs.get('key')
        if key and key.startswith('param_pos_'):
            axis = key[-1]
            setattr(self.state, f"temp_pos_{axis}", 0.0)
        if self._debounce_timer is not None:
            self._debounce_timer.cancel()
        loop = asyncio.get_event_loop()
        self._debounce_timer = loop.call_later(self._debounce_delay, self.update_selected_object)

    # Обработчик изменения временных слайдеров – обновляет объект без задержки
    def on_temp_change(self, *args, **kwargs):
        key = kwargs.get('key')

        setattr(self.state, f"param_pos_x", float(float(getattr(self.state, f"temp_last_pos_x")) + kwargs["temp_pos_delta_x"]))
        setattr(self.state, f"param_pos_y", float(float(getattr(self.state, f"temp_last_pos_y")) + kwargs["temp_pos_delta_y"]))
        setattr(self.state, f"param_pos_z", float(float(getattr(self.state, f"temp_last_pos_z")) + kwargs["temp_pos_delta_z"]))

        # if key and key.startswith('temp_pos_delta_'):
        #     print(2222)
        #     axis = key[-1]
        #     temp = getattr(self.state, f"temp_pos_delta_{axis}")
        #     if temp == 0.0:
        #         return
        #
        #     current = getattr(self.state, f"param_pos_{axis}")
        #     new_val = current + temp
        #     setattr(self.state, f"param_pos_{axis}", new_val)
        self.update_selected_object()

    # Применение дельты при отпускании слайдера
    def apply_delta(self, axis):
        temp = getattr(self.state, f"temp_pos_{axis}")
        if temp == 0.0:
            return
        current = getattr(self.state, f"param_pos_{axis}")
        new_val = current + temp
        setattr(self.state, f"param_pos_{axis}", new_val)
        setattr(self.state, f"temp_pos_{axis}", 0.0)
        # Обновляем объект (он уже обновился при движении, но теперь базовая позиция изменилась)
        self.update_selected_object()

    def apply_x_delta(self):
        self.state.temp_last_pos_x = self.state.param_pos_x
        self.state.temp_pos_delta_x = 0

    def apply_y_delta(self):
        self.state.temp_last_pos_y = self.state.param_pos_y
        self.state.temp_pos_delta_y = 0

    def apply_z_delta(self):
        self.state.temp_last_pos_z = self.state.param_pos_z
        self.state.temp_pos_delta_z = 0

    def set_last_pos(self):
        self.state.temp_last_pos_x = self.state.param_pos_x
        self.state.temp_last_pos_y = self.state.param_pos_y
        self.state.temp_last_pos_z = self.state.param_pos_z

    def select_object(self, obj_id):
        self.state.selected_object_id = obj_id
        self.on_object_selected(obj_id)


# --- Сборка интерфейса Trame / Vuetify 3 ---
server = get_server()
server.client_type = "vue3"
app = OpticsAppController(server)

# Подписка на изменения параметров (кроме temp)
for param in ["param_n", "param_radius", "param_R1", "param_R2", "param_f_target", "param_thickness", "param_edge_radius",
              "param_num_rays", "param_min_offset", "param_max_offset", "param_wavelength", "param_current_n", "param_ray_class", "param_num_spectral_bands", "param_mesh_path",
              "param_plane_shape", "param_plane_width", "param_plane_height", "param_cylinder_capping"]:
    server.state.change(param)(app.on_param_change)

# Для позиционных координат и вращений тоже нужна подписка
for axis in ["x", "y", "z"]:
    server.state.change(f"param_pos_{axis}")(app.on_param_change)
    server.state.change(f"param_rot_{axis}")(app.on_param_change)

for scale_param in ["param_scale_uniform", "param_scale_all", "param_scale_x", "param_scale_y", "param_scale_z"]:
    server.state.change(scale_param)(app.on_param_change)

for effect in ["reflection", "refraction", "absorption"]:
    server.state.change(f"param_{effect}_enabled")(app.on_param_change)
    server.state.change(f"param_{effect}_min")(app.on_param_change)
    server.state.change(f"param_{effect}_max")(app.on_param_change)

# Подписка на временные переменные (для движения во время перетаскивания)
for axis in ["x", "y", "z"]:
    server.state.change(f"temp_pos_delta_{axis}")(app.on_temp_change)



with SinglePageLayout(server) as layout:
    # Заголовок убран
    # layout.title.set_text("Оптический симулятор")
    with layout.content:
        with vuetify.VContainer(fluid=True, classes="pa-0", style="height: 100vh; overflow: hidden;"):
            with vuetify.VRow(no_gutters=True, style="height: 100%;"):
                # Левая колонка управления
                with vuetify.VCol(cols=3, classes="pa-4 bg-grey-darken-4",
                                  style="height: 100%; overflow-y: auto; border-right: 1px solid #444; color: white;"):

                    server.state.setdefault("save_dialog_visible", False)
                    server.state.setdefault("load_dialog_visible", False)
                    server.state.setdefault("file_to_save", "scene_1")
                    server.state.setdefault("file_to_load", "scene_1")

                    with vuetify.VRow(class_="mb-2", no_gutters=True):
                        with vuetify.VCol(cols=6, class_="pr-1"):
                            with vuetify.VBtn(color="success", block=True, size="small",
                                              click="save_dialog_visible = true"):
                                vuetify.VIcon("mdi-content-save", class_="mr-1")
                                "Сохранить"
                        with vuetify.VCol(cols=6, class_="pl-1"):
                            with vuetify.VBtn(color="warning", block=True, size="small",
                                              click="load_dialog_visible = true"):
                                vuetify.VIcon("mdi-folder-open", class_="mr-1")
                                "Загрузить"

                    # Диалоговое окно сохранения файла
                    with vuetify.VDialog(v_model="save_dialog_visible", max_width="400px"):
                        with vuetify.VCard(class_="bg-grey-darken-3 text-white pa-4"):
                            vuetify.VCardTitle("Сохранить сцену", class_="px-0")
                            vuetify.VTextField(v_model="file_to_save", label="Имя файла (без расширения)",
                                               class_="my-3", dense=True)
                            with vuetify.VCardActions(class_="px-0"):
                                vuetify.VSpacer()
                                vuetify.VBtn("Отмена", color="grey", click="save_dialog_visible = false")
                                vuetify.VBtn("ОК", color="success",
                                             click="trigger('save_scene_to_file', [file_to_save]); save_dialog_visible = false")

                    # Диалоговое окно загрузки файла
                    with vuetify.VDialog(v_model="load_dialog_visible", max_width="400px"):
                        with vuetify.VCard(class_="bg-grey-darken-3 text-white pa-4"):
                            vuetify.VCardTitle("Загрузить сцену", class_="px-0")
                            vuetify.VTextField(v_model="file_to_load", label="Имя файла (.json)", class_="my-3",
                                               dense=True)
                            with vuetify.VCardActions(class_="px-0"):
                                vuetify.VSpacer()
                                vuetify.VBtn("Отмена", color="grey", click="load_dialog_visible = false")
                                vuetify.VBtn("ОК", color="warning",
                                             click="trigger('load_scene_from_file', [file_to_load]); load_dialog_visible = false")

                    vuetify.VDivider(class_="my-2")

                    vuetify.VCardTitle("Объекты сцены", classes="text-h6 px-0")

                    # Единая кнопка с выпадающим меню
                    with vuetify.VBtn(color="primary", block=True):
                        vuetify.VIcon("mdi-plus", class_="mr-2")
                        "Добавить объект"
                        with vuetify.VMenu(activator="parent"):
                            with vuetify.VList():
                                vuetify.VListItem("Сферическая линза", click=app.add_lens_click)
                                vuetify.VListItem("Гиперболическая линза", click=app.add_hyperbolic_lens_click)
                                vuetify.VListItem("Плоская поверхность", click=app.add_plane_click)
                                vuetify.VListItem("Сферическая поверхность", click=app.add_sphere_surf_click)
                                vuetify.VListItem("Цилиндрическая поверхность", click=app.add_cylinder_surf_click)
                                vuetify.VListItem("Источник", click=app.add_emitter_click)
                                vuetify.VListItem("3D-меш", click=app.add_mesh_click)

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
                    with vuetify.VRow(no_gutters=True, align="center", class_="mb-2"):
                        with vuetify.VCol(cols=9):
                            vuetify.VCardTitle("Параметры: {{ selected_object_id ? selected_object_id : 'не выбран' }}",
                                               classes="text-subtitle-1 px-0 text-cyan-lighten-2")
                        with vuetify.VCol(cols=3, class_="text-right"):
                            # Кнопка становится активной, только если объект выбран
                            vuetify.VBtn(icon="mdi-crosshairs-gps",
                                         size="small",
                                         color="cyan-darken-2",
                                         variant="flat",
                                         # disabled="selected_object_id == None",
                                         click="trigger('focus_on_object_event')",
                                         title="Сфокусировать камеру на объекте")

                    # --- Блок позиционирования ---
                    vuetify.VListSubheader("Позиция", class_="px-0")
                    # Ось X
                    with vuetify.VRow(v_if="selected_object_type != None", no_gutters=True, align="center"):
                        with vuetify.VCol(cols=4):
                            vuetify.VTextField(v_model="param_pos_x", label="X", type="number",
                                               dense=True, class_="mb-2")
                        with vuetify.VCol(cols=8):
                            vuetify.VSlider(v_model="temp_pos_delta_x", min=-10, max=10, step=0.01,
                                            dense=True, hide_details=True,
                                            thumb_label="always",
                                            mouseup="trigger('apply_x_delta')",
                                            mousedown="trigger('set_last_pos')",)
                    # Ось Y
                    with vuetify.VRow(v_if="selected_object_type != None", no_gutters=True, align="center"):
                        with vuetify.VCol(cols=4):
                            vuetify.VTextField(v_model="param_pos_y", label="Y", type="number",
                                               dense=True, class_="mb-2")
                        with vuetify.VCol(cols=8):
                            vuetify.VSlider(v_model="temp_pos_delta_y", min=-10, max=10, step=0.01,
                                            dense=True, hide_details=True,
                                            thumb_label="always",
                                            mouseup="trigger('apply_y_delta')",
                                            mousedown="trigger('set_last_pos')",)
                    # Ось Z
                    with vuetify.VRow(v_if="selected_object_type != None", no_gutters=True, align="center"):
                        with vuetify.VCol(cols=4):
                            vuetify.VTextField(v_model="param_pos_z", label="Z", type="number",
                                               dense=True, class_="mb-2")
                        with vuetify.VCol(cols=8):
                            vuetify.VSlider(v_model="temp_pos_delta_z", min=-10, max=10, step=0.01,
                                            dense=True, hide_details=True,
                                            thumb_label="always",
                                            mouseup="trigger('apply_z_delta')",
                                            mousedown="trigger('set_last_pos')",)

                    # --- Блок вращения ---
                    vuetify.VListSubheader("Поворот (град)", class_="px-0")
                    for axis in ["x", "y", "z"]:
                        with vuetify.VRow(v_if="selected_object_type != None", no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(v_model=f"param_rot_{axis}", label=axis.upper(), type="number",
                                                   dense=True, class_="mb-2")
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(v_model=f"param_rot_{axis}", min=-180, max=180, step=1, dense=True,
                                                hide_details=True)

                    vuetify.VDivider(class_="my-4")

                    with vuetify.VContainer(v_if="selected_object_type == 'plane'", class_="pa-0"):
                        vuetify.VListSubheader("Параметры плоскости", class_="px-0")

                        # ИСПРАВЛЕНИЕ: v_model перенесен строго на группу, убраны пустые блоки pass
                        with vuetify.VRadioGroup(v_model="param_plane_shape", inline=True, class_="mb-2",
                                                 hide_details=True):
                            vuetify.VRadio(label="Круг", value="circle", color="cyan")
                            vuetify.VRadio(label="Прямоугольник", value="rectangle", color="cyan")

                        # Если выбран КРУГ — показываем радиус апертуры
                        with vuetify.VContainer(v_if="param_plane_shape == 'circle'", class_="pa-0"):
                            with vuetify.VRow(no_gutters=True, align="center"):
                                with vuetify.VCol(cols=4):
                                    vuetify.VTextField(v_model="param_edge_radius", label="Радиус", type="number",
                                                       dense=True)
                                with vuetify.VCol(cols=8):
                                    vuetify.VSlider(v_model="param_edge_radius", min=0.1, max=5.0, step=0.1, dense=True,
                                                    hide_details=True)

                        # Если ПРЯМОУГОЛЬНИК — показываем настройки ширины и высоты
                        with vuetify.VContainer(v_if="param_plane_shape == 'rectangle'", class_="pa-0"):
                            with vuetify.VRow(no_gutters=True, align="center", class_="mb-1"):
                                with vuetify.VCol(cols=4):
                                    vuetify.VTextField(v_model="param_plane_width", label="Ширина", type="number",
                                                       dense=True)
                                with vuetify.VCol(cols=8):
                                    vuetify.VSlider(v_model="param_plane_width", min=0.1, max=10.0, step=0.1,
                                                    dense=True, hide_details=True)
                            with vuetify.VRow(no_gutters=True, align="center"):
                                with vuetify.VCol(cols=4):
                                    vuetify.VTextField(v_model="param_plane_height", label="Высота", type="number",
                                                       dense=True)
                                with vuetify.VCol(cols=8):
                                    vuetify.VSlider(v_model="param_plane_height", min=0.1, max=10.0, step=0.1,
                                                    dense=True, hide_details=True)

                        with vuetify.VRow(no_gutters=True, align="center", class_="mt-2"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(v_model="param_n", label="n (внутри)", type="number", dense=True,
                                                   step=0.01)
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(v_model="param_n", min=1.0, max=2.5, step=0.01, dense=True,
                                                hide_details=True)

                    # Параметры SphereSurface
                    with vuetify.VContainer(v_if="selected_object_type == 'sphere_surf'", class_="pa-0"):
                        vuetify.VListSubheader("Параметры сферической поверхности", class_="px-0")
                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(v_model="param_radius", label="Радиус кривизны", type="number",
                                                   dense=True, step=0.5)
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(v_model="param_radius", min=-20.0, max=20.0, step=0.5, dense=True,
                                                hide_details=True)
                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(v_model="param_edge_radius", label="Апертура", type="number",
                                                   dense=True, step=0.1)
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(v_model="param_edge_radius", min=0.1, max=5.0, step=0.1, dense=True,
                                                hide_details=True)
                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(v_model="param_n", label="n (внутри)", type="number", dense=True,
                                                   step=0.01)
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(v_model="param_n", min=1.0, max=2.5, step=0.01, dense=True,
                                                hide_details=True)

                    # Параметры CylinderSurface
                    with vuetify.VContainer(v_if="selected_object_type == 'cylinder_surf'", class_="pa-0"):
                        vuetify.VListSubheader("Параметры цилиндрической поверхности", class_="px-0")

                        # Тумблер включения/выключения оснований
                        vuetify.VSwitch(v_model="param_cylinder_capping", label="Включить основания (крышки)",
                                        dense=True, hide_details=True, class_="mb-2", color="orange")

                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(v_model="param_radius", label="Радиус", type="number", dense=True,
                                                   step=0.1)
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(v_model="param_radius", min=0.1, max=5.0, step=0.1, dense=True,
                                                hide_details=True)
                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(v_model="param_thickness", label="Полудлина", type="number",
                                                   dense=True, step=0.1)
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(v_model="param_thickness", min=0.1, max=5.0, step=0.1, dense=True,
                                                hide_details=True)
                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(v_model="param_n", label="n (внутри)", type="number", dense=True,
                                                   step=0.01)
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(v_model="param_n", min=1.0, max=2.5, step=0.01, dense=True,
                                                hide_details=True)

                    # --- Параметры сферической линзы ---
                    with vuetify.VContainer(v_if="selected_object_type == 'lens'", class_="pa-0"):
                        vuetify.VListSubheader("Параметры сферич. линзы", class_="px-0")
                        # n
                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(v_model="param_n", label="n", type="number", dense=True, step=0.01)
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(v_model="param_n", min=1.0, max=2.5, step=0.01,
                                                dense=True, hide_details=True)
                        # R1
                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(v_model="param_R1", label="R1", type="number", dense=True, step=0.5)
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(v_model="param_R1", min=-20, max=20, step=0.5,
                                                dense=True, hide_details=True)
                        # R2
                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(v_model="param_R2", label="R2", type="number", dense=True, step=0.5)
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(v_model="param_R2", min=-20, max=20, step=0.5,
                                                dense=True, hide_details=True)
                        # Толщина
                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(v_model="param_thickness", label="Толщина", type="number", dense=True, step=0.1)
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(v_model="param_thickness", min=0.1, max=5.0, step=0.1,
                                                dense=True, hide_details=True)
                        # Радиус апертуры
                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(v_model="param_edge_radius", label="Апертура", type="number", dense=True, step=0.1)
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(v_model="param_edge_radius", min=0.1, max=5.0, step=0.1,
                                                dense=True, hide_details=True)

                    # --- Параметры гиперболической линзы ---
                    with vuetify.VContainer(v_if="selected_object_type == 'hyperbolic_lens'", class_="pa-0"):
                        vuetify.VListSubheader("Параметры гиперболической линзы", class_="px-0")
                        # Толщина
                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(v_model="param_thickness", label="Толщина", type="number", dense=True, step=0.1)
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(v_model="param_thickness", min=0.1, max=5.0, step=0.1,
                                                dense=True, hide_details=True)
                        # Радиус апертуры
                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(v_model="param_edge_radius", label="Апертура", type="number", dense=True, step=0.1)
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(v_model="param_edge_radius", min=0.1, max=5.0, step=0.1,
                                                dense=True, hide_details=True)
                        # Фокус
                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(v_model="param_f_target", label="f", type="number", dense=True, step=0.1)
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(v_model="param_f_target", min=-30.0, max=30.0, step=0.1,
                                                dense=True, hide_details=True)
                        # n
                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(v_model="param_n", label="n", type="number", dense=True, step=0.01)
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(v_model="param_n", min=1.001, max=2.5, step=0.01,
                                                dense=True, hide_details=True)

                    # --- Параметры источника ---
                    with vuetify.VContainer(v_if="selected_object_type == 'emitter'", class_="pa-0"):
                        vuetify.VListSubheader("Параметры источника", class_="px-0")

                        # Выбор типа луча
                        vuetify.VSelect(label="Тип луча", v_model="param_ray_class",
                                        items=("ray_classes_list",),
                                        item_title="title", item_value="value",
                                        dense=True, class_="mb-2")

                        # Количество спектральных полос (показывается только для WhiteRay)
                        with vuetify.VContainer(v_if="param_ray_class == 'WhiteRay'", class_="pa-0 mb-2"):
                            with vuetify.VRow(no_gutters=True, align="center"):
                                with vuetify.VCol(cols=4):
                                    vuetify.VTextField(v_model="param_num_spectral_bands",
                                                       label="Спектр (кол-во)", type="number", dense=True,
                                                       step=1)
                                with vuetify.VCol(cols=8):
                                    vuetify.VSlider(v_model="param_num_spectral_bands", min=2, max=30,
                                                    step=1,
                                                    dense=True, hide_details=True)

                        # Количество лучей
                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(v_model="param_num_rays", label="Кол-во лучей",
                                                   type="number", dense=True, step=1)
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(v_model="param_num_rays", min=1, max=20, step=1,
                                                dense=True, hide_details=True)

                        # Длина волны (скрываем для Белого Света, так как он сам генерирует спектр)
                        with vuetify.VRow(v_if="param_ray_class != 'WhiteRay'", no_gutters=True,
                                          align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(v_model="param_wavelength", label="λ (нм)",
                                                   type="number", dense=True, step=10)
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(v_model="param_wavelength", min=380, max=780, step=10,
                                                dense=True, hide_details=True)

                        # Мин. смещение
                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(v_model="param_min_offset", label="Мин. смещ.",
                                                   type="number", dense=True, step=0.1)
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(v_model="param_min_offset", min=-3.0, max=0.0, step=0.1,
                                                dense=True, hide_details=True)
                        # Макс. смещение
                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(v_model="param_max_offset", label="Макс. смещ.",
                                                   type="number", dense=True, step=0.1)
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(v_model="param_max_offset", min=0.0, max=3.0, step=0.1,
                                                dense=True, hide_details=True)

                        # n среды
                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(v_model="param_current_n", label="n среды",
                                                   type="number", dense=True, step=0.1)
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(v_model="param_current_n", min=1.0, max=3.0, step=0.01,
                                                dense=True, hide_details=True)

                    # --- Параметры 3D-меша ---
                    with vuetify.VContainer(v_if="selected_object_type == 'mesh'", class_="pa-0"):
                        vuetify.VListSubheader("Параметры 3D-модели", class_="px-0")
                        vuetify.VTextField(v_model="param_mesh_path", label="Путь к файлу (.obj/.stl)", dense=True,
                                           class_="mb-2")
                        with vuetify.VRow(no_gutters=True, align="center", class_="mb-2"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(v_model="param_n", label="n", type="number", dense=True, step=0.01)
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(v_model="param_n", min=1.0, max=2.5, step=0.01,
                                                dense=True, hide_details=True)

                        vuetify.VDivider(class_="my-2")
                        # НАЧАЛО ИЗМЕНЕНИЙ ИНТЕРФЕЙСА ДЛЯ МАСШТАБИРОВАНИЯ
                        vuetify.VSwitch(v_model="param_scale_uniform", label="Пропорциональный масштаб",
                                        dense=True, hide_details=True, class_="mb-2", color="cyan")

                        # Если тумблер включен: один общий ползунок
                        with vuetify.VContainer(v_if="param_scale_uniform", class_="pa-0"):
                            with vuetify.VRow(no_gutters=True, align="center"):
                                with vuetify.VCol(cols=4):
                                    vuetify.VTextField(v_model="param_scale_all", label="Масштаб", type="number",
                                                       dense=True, step=0.1)
                                with vuetify.VCol(cols=8):
                                    vuetify.VSlider(v_model="param_scale_all", min=0.1, max=10.0, step=0.1,
                                                    dense=True, hide_details=True)

                        # Если тумблер выключен: три раздельных ползунка по осям X, Y, Z
                        with vuetify.VContainer(v_if="!param_scale_uniform", class_="pa-0"):
                            for axis in ["x", "y", "z"]:
                                with vuetify.VRow(no_gutters=True, align="center", class_="mb-1"):
                                    with vuetify.VCol(cols=4):
                                        vuetify.VTextField(v_model=f"param_scale_{axis}", label=f"Scale {axis.upper()}",
                                                           type="number", dense=True, step=0.1)
                                    with vuetify.VCol(cols=8):
                                        vuetify.VSlider(v_model=f"param_scale_{axis}", min=0.1, max=10.0, step=0.1,
                                                        dense=True, hide_details=True)

                    with vuetify.VContainer(v_if="selected_object_type != 'emitter'", class_="pa-0"):
                        vuetify.VListSubheader("Оптические свойства (λ, нм)", class_="px-0 text-cyan-lighten-2")

                        # Шаблон для генерации трех типов взаимодействия
                        effects = [
                            {"key": "refraction", "label": "Преломление"},
                            {"key": "reflection", "label": "Отражение"},
                            {"key": "absorption", "label": "Поглощение"}
                        ]


                        for eff in effects:
                            k = eff["key"]
                            vuetify.VCheckbox(v_model=f"param_{k}_enabled", v_if=f"selected_object_type != None", label=eff["label"], dense=True,
                                              hide_details=True)

                            with vuetify.VRow(v_if=f"param_{k}_enabled && selected_object_type != None", no_gutters=True, align="center",
                                              class_="pl-4 mb-2"):
                                # Поле MIN
                                with vuetify.VCol(cols=5, class_="pr-1"):
                                    vuetify.VTextField(v_model=f"param_{k}_min", label="Мин", type="number", dense=True,
                                                       hide_details=True)
                                # Поле MAX
                                with vuetify.VCol(cols=5, class_="px-1"):
                                    vuetify.VTextField(v_model=f"param_{k}_max", label="Макс", dense=True,
                                                       hide_details=True)
                                # Кнопка INFINITY
                                with vuetify.VCol(cols=2, class_="pl-1"):
                                    vuetify.VBtn(icon="mdi-infinity", size="small", color="grey-darken-2",
                                                 variant="flat",
                                                 click=f"param_{k}_max = Infinity")

                    vuetify.VDivider(class_="my-4")
                    vuetify.VSelect(label="Режим трассировки", v_model="trace_mode", items=("trace_modes",),
                                    item_title="title", item_value="value", dense=True, class_="mt-2")

                    vuetify.VListSubheader("Рендеринг 3D-сцены", class_="px-0 text-cyan-lighten-2")
                    with vuetify.VBtnToggle(v_model="render_mode", mandatory=True, block=True, class_="mb-2",
                                            color="cyan-darken-3"):
                        with vuetify.VBtn(value="remote", style="width: 50%"):
                            vuetify.VIcon("mdi-server", class_="mr-1")
                            "Server"
                        with vuetify.VBtn(value="local", style="width: 50%"):
                            vuetify.VIcon("mdi-monitor", class_="mr-1")
                            "Client"

                # Правая колонка с 3D окном pyvista
                with vuetify.VCol(cols=9, style="height: 100%; overflow: hidden;"):
                    # ui_view = plotter_ui(
                    #     app.plotter,
                    #     mode="trame",
                    #     default_server_rendering=False,
                    #     add_menu=False,
                    #     image_scale=1,
                    #     interactor_style="Terrain"
                    # )
                    ui_view = PyVistaRemoteLocalView(
                        app.plotter,
                        mode=("render_mode",)
                    )
                    # ui_view.set_mode(server.state.viewMode)
                    app.ctrl.view_update = ui_view.update


app.update_scene()

if __name__ == "__main__":
    server.start(host="127.0.0.1", port=8085)