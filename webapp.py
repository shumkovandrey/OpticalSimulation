import os
import warnings

import asyncio
import json
import time

import numpy as np
import pyvista as pv
import vtk

from matplotlib.colors import to_hex, to_rgb
from pyvista.trame.ui import plotter_ui
from scipy.spatial.transform import Rotation as R
from trame.app import get_server
from trame.widgets import vuetify3 as vuetify
from trame_vuetify.ui.vuetify3 import VAppLayout

from core.materials import Material, parse_range
from core.rays import DispersiveRay, Ray, RayPool, WhiteRay
from emitters import BeamEmitter
from surfaces.cylinder import CylinderSurface
from surfaces.hyperbolic_lens import HyperbolicLens
from surfaces.mesh import MeshSurface
from surfaces.plane import PlaneSurface
from surfaces.sphere import SphereSurface
from surfaces.universal_lens import UniversalLens
from visualization import RayTracer

pv.OFF_SCREEN = True
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")

class OpticsAppController:
    def __init__(self, server):
        self.server = server
        self.state = server.state
        self.ctrl = server.controller

        self.plotter = pv.Plotter(off_screen=True)
        self.plotter.set_background("black")
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
            line_width=1,
            min_alpha=0,
            gamma=1,
        )
        self.ray_tracer.mode.energy_color_type = 2
        self.ray_tracer.mode.max_bounces = 100
        self.ray_tracer.mode.offset_distance = 0.005

        self._interval_visual = 0.1
        self._interval_shape = 0.5
        self._interval_trace = 1
        self._interval_net = 0.1

        self._dirty_transform = False
        self._dirty_shape = False
        self._dirty_trace = False
        self._dirty_net = False

        self._last_visual_ts = 0.0
        self._last_shape_ts = 0.0
        self._last_trace_ts = 0.0
        self._last_net_ts = 0.0

        self._ticker_task = None

        self.state.selected_object_id = None
        self.state.selected_object_type = None
        self.state.trace_mode = "simple"
        self.state.scene_objects_list = []
        self.state.render_mode = "local"
        self.state.trace_modes = [
            {"title": "Simple", "value": "simple"},
            {"title": "Tree", "value": "tree"},
        ]

        # Параметры позиции/поворота
        self.state.param_pos_x = 0.0
        self.state.param_pos_y = 0.0
        self.state.param_pos_z = 0.0
        self.state.param_rot_x = 0.0
        self.state.param_rot_y = 0.0
        self.state.param_rot_z = 0.0

        # Параметры изменения позиции
        self.state.temp_pos_delta_x = 0.0
        self.state.temp_pos_delta_y = 0.0
        self.state.temp_pos_delta_z = 0.0
        self.state.temp_last_pos_x = 0.0
        self.state.temp_last_pos_y = 0.0
        self.state.temp_last_pos_z = 0.0

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
        self.state.param_scale_uniform = True
        self.state.param_scale_all = 1.0
        self.state.param_scale_x = 1.0
        self.state.param_scale_y = 1.0
        self.state.param_scale_z = 1.0
        self.state.param_plane_shape = "rectangle"
        self.state.param_plane_width = 2.0
        self.state.param_plane_height = 2.0
        self.state.param_cylinder_capping = False
        self.state.param_ray_class = "Ray"
        self.state.param_num_spectral_bands = 7
        self.state.ray_classes_list = [
            {"title": "Обычный луч (Ray)", "value": "Ray"},
            {"title": "Дисперсионный луч (DispersiveRay)", "value": "DispersiveRay"},
            {"title": "Белый свет (WhiteRay)", "value": "WhiteRay"},
        ]
        self.state.param_energy = 1.0
        self.state.param_ray_color = "#ffff00"
        self.state.param_ambient_n = 1.0
        self.state.param_ray_thickness = 3

        self.state.param_interval_visual = self._interval_visual
        self.state.param_interval_shape = self._interval_shape
        self.state.param_interval_trace = self._interval_trace
        self.state.param_interval_net = self._interval_net

        # Параметры материала
        self.state.param_mat_n = 1.5
        self.state.param_mat_empirical_coeff = 0.005
        self.state.param_mat_a_offset = 0.01
        self.state.param_mat_object_color = "#00ffff"
        self.state.param_mat_opacity = 0.5
        self.state.param_mat_reflectivity = 0.0

        for effect in ["reflection", "refraction", "absorption"]:
            self.state[f"param_mat_{effect}_min"] = 0.0
            self.state[f"param_mat_{effect}_max"] = np.inf
            self.state[f"param_mat_{effect}_enabled"] = effect == "refraction"

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
        self.ctrl.trigger("export_scene")(self.export_scene)

        # Для исправления бага с начальной толщиной лучей
        self.is_first_rays_visual = True

        self.initializing = False
        self._ensure_ticker()

    def _ensure_ticker(self):
        if self._ticker_task is not None and not self._ticker_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._ticker_task = loop.create_task(self._ticker())

    async def _ticker(self):
        while True:
            await asyncio.sleep(0.01)

            if self._loading_state or self.initializing:
                continue

            now = time.monotonic()

            if self._dirty_transform and (now - self._last_visual_ts) >= self._interval_visual:
                self._last_visual_ts = now
                self._dirty_transform = False
                try:
                    self._apply_visual_and_physics()
                except Exception as e:
                    print(f"[ticker visual] {e}")

            if self._dirty_shape and (now - self._last_shape_ts) >= self._interval_shape:
                self._last_shape_ts = now
                self._dirty_shape = False
                try:
                    self._apply_shape_visual()
                except Exception as e:
                    print(f"[ticker shape] {e}")

            if self._dirty_trace and (now - self._last_trace_ts) >= self._interval_trace:
                self._last_trace_ts = now
                self._dirty_trace = False
                try:
                    if self._dirty_transform:
                        self._apply_visual_and_physics()
                        self._dirty_transform = False
                        self._last_visual_ts = now
                    if self._dirty_shape:
                        self._apply_shape_visual()
                        self._dirty_shape = False
                        self._last_shape_ts = now
                    self._trace_and_render()
                except Exception as e:
                    print(f"[ticker trace] {e}")

            if self._dirty_net and (now - self._last_net_ts) >= self._interval_net:
                self._last_net_ts = now
                self._dirty_net = False
                self.ctrl.view_update()

    def _mark_transform_dirty(self):
        self._dirty_transform = True
        self._dirty_trace = True
        self._ensure_ticker()

    def _mark_shape_dirty(self):
        self._dirty_shape = True
        self._dirty_trace = True
        self._ensure_ticker()

    def _read_material_from_state(self) -> Material:
        mat = Material()
        mat.n = float(self.state.param_mat_n)
        mat.empirical_coeff = float(self.state.param_mat_empirical_coeff)
        mat.a_offset = float(self.state.param_mat_a_offset)
        mat.color = tuple(to_rgb(self.state.param_mat_object_color))
        mat.opacity = float(self.state.param_mat_opacity)
        mat.reflectivity = float(self.state.param_mat_reflectivity)
        for eff in ["reflection", "refraction", "absorption"]:
            if self.state[f"param_mat_{eff}_enabled"]:
                raw_max = self.state[f"param_mat_{eff}_max"]
                v_max = np.inf if raw_max in ["Infinity", "inf", None, ""] else float(raw_max)
                setattr(mat, f"{eff}_range", (float(self.state[f"param_mat_{eff}_min"]), v_max))
            else:
                setattr(mat, f"{eff}_range", None)
        return mat

    def on_transform_param_change(self, *args, **kwargs):
        if self._loading_state:
            return
        self._mark_transform_dirty()

    def on_temp_change(self, *args, **kwargs):
        self.state.param_pos_x = float(float(self.state.temp_last_pos_x) + kwargs["temp_pos_delta_x"])
        self.state.param_pos_y = float(float(self.state.temp_last_pos_y) + kwargs["temp_pos_delta_y"])
        self.state.param_pos_z = float(float(self.state.temp_last_pos_z) + kwargs["temp_pos_delta_z"])
        if self._loading_state:
            return
        self._mark_transform_dirty()

    def on_shape_param_change(self, *args, **kwargs):
        if self._loading_state:
            return
        self._mark_shape_dirty()

    def _apply_visual_and_physics(self):
        obj_id = self.state.selected_object_id
        obj_entry = self._find_object(obj_id)
        if not obj_entry or obj_id not in self.plotter.actors:
            return

        p = obj_entry["params"]
        new_origin = (float(self.state.param_pos_x),
                      float(self.state.param_pos_y),
                      float(self.state.param_pos_z))
        new_rotation = (float(self.state.param_rot_x),
                        float(self.state.param_rot_y),
                        float(self.state.param_rot_z))

        p["origin"] = new_origin
        p["rotation"] = new_rotation

        obj_entry["instance"] = self._create_instance(obj_entry["type"], p)

        actor = self.plotter.actors[obj_id]
        mat = np.eye(4)
        mat[:3, :3] = R.from_euler("xyz", new_rotation, degrees=True).as_matrix()
        mat[:3, 3] = new_origin
        actor.user_matrix = mat

        self._dirty_net = True

        # if hasattr(self.ctrl, "view_update"):
        #     self.ctrl.view_update()

    def _apply_shape_visual(self):
        obj_id = self.state.selected_object_id
        obj_entry = self._find_object(obj_id)
        if not obj_entry:
            return

        p = obj_entry["params"]
        current_origin = (float(self.state.param_pos_x), float(self.state.param_pos_y), float(self.state.param_pos_z))

        current_rotation = (float(self.state.param_rot_x), float(self.state.param_rot_y), float(self.state.param_rot_z))

        p["origin"] = current_origin
        p["rotation"] = current_rotation

        if obj_entry["type"] == "lens":
            p["material"] = self._read_material_from_state()
            p["R1"] = float(self.state.param_R1)
            p["R2"] = float(self.state.param_R2)
            p["thickness"] = float(self.state.param_thickness)
            p["edge_radius"] = float(self.state.param_edge_radius)
        elif obj_entry["type"] == "hyperbolic_lens":
            p["material"] = self._read_material_from_state()
            p["f_target"] = float(self.state.param_f_target)
            p["thickness"] = float(self.state.param_thickness)
            p["edge_radius"] = float(self.state.param_edge_radius)
        elif obj_entry["type"] == "plane":
            p["material"] = self._read_material_from_state()
            p["edge_radius"] = float(self.state.param_edge_radius)
            p["shape_type"] = str(self.state.param_plane_shape)
            p["width"] = float(self.state.param_plane_width)
            p["height"] = float(self.state.param_plane_height)
        elif obj_entry["type"] == "sphere_surf":
            p["radius"] = float(self.state.param_radius)
            p["edge_radius"] = float(self.state.param_edge_radius)
            p["material"] = self._read_material_from_state()
        elif obj_entry["type"] == "cylinder_surf":
            p["radius"] = float(self.state.param_radius)
            p["half_length"] = float(self.state.param_thickness)
            p["material"] = self._read_material_from_state()
            p["capping"] = bool(self.state.param_cylinder_capping)
        elif obj_entry["type"] == "mesh":
            p["material"] = self._read_material_from_state()
            p["mesh_path"] = str(self.state.param_mesh_path)
            p["scale_uniform"] = bool(self.state.param_scale_uniform)
            p["scale_all"] = float(self.state.param_scale_all)
            p["scale_x"] = float(self.state.param_scale_x)
            p["scale_y"] = float(self.state.param_scale_y)
            p["scale_z"] = float(self.state.param_scale_z)
        elif obj_entry["type"] == "emitter":
            p["num_rays"] = int(self.state.param_num_rays)
            p["min_offset"] = float(self.state.param_min_offset)
            p["max_offset"] = float(self.state.param_max_offset)
            p["wavelength"] = float(self.state.param_wavelength)
            p["energy"] = float(self.state.param_energy)
            p["color"] = tuple(to_rgb(self.state.param_ray_color))
            p["current_n"] = float(self.state.param_current_n)
            p["num_spectral_bands"] = int(self.state.param_num_spectral_bands)
            mapping = {"Ray": Ray, "DispersiveRay": DispersiveRay, "WhiteRay": WhiteRay}
            p["ray_class"] = mapping.get(self.state.param_ray_class, Ray)

        # for effect in ["reflection", "refraction", "absorption"]:
        #     if bool(self.state[f"param_{effect}_enabled"]):
        #         raw_max = self.state[f"param_{effect}_max"]
        #         v_max = np.inf if raw_max in ["Infinity", "inf", None, ""] else float(raw_max)
        #         p[f"{effect}_range"] = (float(self.state[f"param_{effect}_min"]), v_max)
        #     else:
        #         p[f"{effect}_range"] = None

        obj_entry["instance"] = self._create_instance(obj_entry["type"], p)

        if obj_id in self.plotter.actors:
            self.plotter.remove_actor(obj_id)

        local_mesh = obj_entry["instance"].get_mesh()
        if obj_entry["type"] == "emitter":
            color, opacity, reflectivity = "green", 0.5, 0.0
        else:
            color = self.state.param_mat_object_color
            opacity = self.state.param_mat_opacity
            reflectivity = self.state.param_mat_reflectivity

        actor = self.plotter.add_mesh(
            local_mesh, color=color, opacity=opacity,
            smooth_shading=True, name=obj_id,
        )

        # Зеркальность без PBR
        prop = actor.GetProperty()
        prop.SetSpecular(reflectivity)
        prop.SetSpecularPower(30)
        prop.SetAmbient(0.1 + 0.2 * reflectivity)
        prop.Modified()

        mat = np.eye(4)
        mat[:3, :3] = R.from_euler(
            "xyz",
            (self.state.param_rot_x, self.state.param_rot_y, self.state.param_rot_z),
            degrees=True,
        ).as_matrix()
        mat[:3, 3] = (self.state.param_pos_x, self.state.param_pos_y, self.state.param_pos_z)
        actor.user_matrix = mat

        self._dirty_net = True

        # if hasattr(self.ctrl, "view_update"):
        #     self.ctrl.view_update()

    def _trace_and_render(self):
        self.ray_tracer.mode.ambient_n = float(self.state.param_ambient_n)
        self.ray_tracer.elements.clear()
        self.ray_tracer.rays.clear()
        if hasattr(self.ray_tracer, "emitters"):
            self.ray_tracer.emitters.clear()

        for obj_entry in self.scene_objects:
            instance = obj_entry["instance"]
            if isinstance(instance, (UniversalLens, HyperbolicLens)):
                for surf in instance.get_surfaces():
                    self.ray_tracer.add_elements(surf)
            elif isinstance(instance, MeshSurface):
                self.ray_tracer.add_elements(instance)
            elif isinstance(instance, (PlaneSurface, SphereSurface, CylinderSurface)):
                self.ray_tracer.add_elements(instance)
            elif isinstance(instance, BeamEmitter):
                self.ray_tracer.add_emitter(instance)

        segments = self.ray_tracer.trace_all()

        if self.pool:
            for r in self.ray_tracer.rays:
                self.pool.release(r)
        self.ray_tracer.rays.clear()

        self.ray_tracer.cloud.update(segments, energy_color_type=self.ray_tracer.mode.energy_color_type)
        calculated_ray_mesh = self.ray_tracer.cloud.actor.mapper.dataset

        if "traced_rays_geometry" in self.plotter.actors:
            actor = self.plotter.actors["traced_rays_geometry"]
            actor.mapper.SetScalarVisibility(0)
            actor.mapper.SetLookupTable(None)
            if calculated_ray_mesh and calculated_ray_mesh.n_points > 0:
                actor.mapper.dataset.copy_from(calculated_ray_mesh)
                actor.mapper.dataset.point_data["colors"] = calculated_ray_mesh.point_data["colors"]
                actor.mapper.dataset.active_scalars_name = "colors"
                actor.mapper.SetColorModeToDirectScalars()
                actor.mapper.SetScalarModeToUsePointData()
                actor.mapper.ScalarVisibilityOn()

                # Исправление бага, из-за которого при запуске лучи имеют неправильную толщину
                if self.is_first_rays_visual:
                    actor.GetProperty().SetLineWidth(5)
                    self.is_first_rays_visual = False
                else:
                    actor.GetProperty().SetLineWidth(float(self.state.param_ray_thickness))

                actor.GetProperty().SetRenderLinesAsTubes(True)
                actor.mapper.dataset.Modified()
                actor.mapper.Modified()
            else:
                actor.mapper.dataset.copy_from(pv.PolyData())
                actor.mapper.Modified()
        elif calculated_ray_mesh and calculated_ray_mesh.n_points > 0:
            calculated_ray_mesh.active_scalars_name = "colors"
            self.plotter.add_mesh(
                calculated_ray_mesh,
                scalars="colors", rgba=True, opacity="linear",
                line_width=float(self.state.param_ray_thickness), render_lines_as_tubes=True,
                name="traced_rays_geometry",
            )

        self._dirty_net = True

        # if hasattr(self.ctrl, "view_update"):
        #     self.ctrl.view_update()

    def update_scene(self):
        self._apply_visual_and_physics()
        self._apply_shape_visual()
        self._trace_and_render()

        self._dirty_transform = self._dirty_shape = self._dirty_trace = False
        now = time.monotonic()
        self._last_visual_ts = self._last_shape_ts = self._last_trace_ts = now

    def _update_objects_list_state(self):
        self.state.scene_objects_list = [
            {"id": o["id"], "name": o["name"], "type": o["type"]} for o in self.scene_objects
        ]

    def save_scene_to_file(self, filename):
        """Сохраняет параметры всех объектов сцены в JSON-файл в корне проекта."""
        if not filename:
            return

        if not filename.endswith(".json"):
            filename += ".json"

        serialized_data = []
        for obj in self.scene_objects:
            save_params = obj["params"].copy()
            mat = save_params.pop("material", None)
            if mat is not None:
                save_params["material"] = mat.to_dict()

            for key, val in save_params.items():
                if isinstance(val, tuple):
                    save_params[key] = list(val)
                elif isinstance(val, type):
                    save_params[key] = val.__name__
                elif isinstance(val, (list, tuple)) and len(val) == 2:
                    save_params[key] = [
                        "Infinity" if v == np.inf or (isinstance(v, float) and np.isinf(v)) else v for v in val
                    ]

            serialized_data.append({"name": obj["name"], "type": obj["type"], "params": save_params})

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

        filename = "scenes/" + filename + ".json"

        if not os.path.exists(filename):
            print(f"Файл {filename} не найден.")
            return

        try:
            with open(filename, "r", encoding="utf-8") as f:
                loaded_data = json.load(f)
        except Exception as e:
            print(f"Ошибка чтения файла сцены: {e}")
            return

        for obj in self.scene_objects:
            self.plotter.remove_actor(obj["id"])
        self.scene_objects.clear()
        self.object_counter = 0
        self.state.selected_object_id = None
        self.state.selected_object_type = None

        self._loading_state = True
        self.initializing = True

        try:
            for obj_data in loaded_data:
                params = obj_data["params"]

                if "origin" in params:
                    params["origin"] = tuple(params["origin"])
                if "rotation" in params:
                    params["rotation"] = tuple(params["rotation"])

                if "ray_class" in params:
                    mapping = {"Ray": Ray, "DispersiveRay": DispersiveRay, "WhiteRay": WhiteRay}
                    params["ray_class"] = mapping.get(params["ray_class"], Ray)

                # if "ray_color" in params and isinstance(params["ray_color"], list):
                #     params["ray_color"] = tuple(params["ray_color"])

                if "material" in params and isinstance(params["material"], dict):
                    params["material"] = Material.from_dict(params["material"])
                else:
                    params["material"] = Material(
                        n=float(params.pop("n", 1.5)),
                        empirical_coeff=float(params.pop("empirical_coeff", 0.005)),
                        a_offset=float(params.pop("a_offset", 0.01)),
                        refraction_range=parse_range(params.pop("refraction_range", (0.0, np.inf))),
                        reflection_range=parse_range(params.pop("reflection_range", None)),
                        absorption_range=parse_range(params.pop("absorption_range", None)),
                        color=(0.0, 1.0, 1.0), opacity=0.5, reflectivity=0.0,
                    )

                for effect in ["reflection", "refraction", "absorption"]:
                    range_key = f"{effect}_range"
                    if range_key in params and params[range_key] is not None:
                        r_min = params[range_key][0]
                        r_max = params[range_key][1]
                        if r_max == "Infinity":
                            r_max = np.inf
                        params[range_key] = (r_min, r_max)

                self.add_object(obj_data["type"], obj_data["name"], params)

        except Exception as e:
            print(f"Ошибка при разборе данных сцены: {e}")
        finally:
            self._loading_state = False
            self.initializing = False

        self._update_objects_list_state()
        if self.scene_objects:
            first_id = self.scene_objects[0]["id"]
            self.state.selected_object_id = first_id
            self.on_object_selected(first_id)

        self.update_scene()

    def export_scene(self):
        self.plotter.export_gltf("my_scene.gltf")

    def add_object(self, obj_type, name, params):
        self.object_counter += 1
        obj_id = f"obj_{self.object_counter}"

        instance = self._create_instance(obj_type, params)
        obj_entry = {"id": obj_id, "type": obj_type, "name": name, "params": params, "instance": instance}
        self.scene_objects.append(obj_entry)

        local_mesh = instance.get_mesh()

        if obj_type == "emitter":
            color, opacity, reflectivity = "green", 0.5, 0.0
        else:
            mat = params.get("material", Material())
            color = mat.color
            opacity = mat.opacity
            reflectivity = mat.reflectivity

        actor = self.plotter.add_mesh(
            local_mesh, color=color, opacity=opacity,
            smooth_shading=True, name=obj_id,
        )

        # Зеркальность без PBR
        prop = actor.GetProperty()
        prop.SetSpecular(reflectivity)
        prop.SetSpecularPower(30)
        prop.SetAmbient(0.1 + 0.2 * reflectivity)
        prop.Modified()

        mat = np.eye(4)
        mat[:3, :3] = R.from_euler("xyz", params.get("rotation", (0, 0, 0)), degrees=True).as_matrix()
        mat[:3, 3] = params.get("origin", (0, 0, 0))
        actor.user_matrix = mat

        if not self.initializing:
            self._update_objects_list_state()
            self.state.selected_object_id = obj_id
            self.on_object_selected(obj_id)
            self.update_scene()
        return obj_id

    def add_plane_click(self):
        self.add_object(
            "plane",
            f"Плоскость {len(self.scene_objects) + 1}",
            {
                "origin": (0, 0, 0),
                "rotation": (0, 0, 0),
                "edge_radius": 1.5,
                "n": 1.0,
                "shape_type": "circle",
                "width": 2.0,
                "height": 2.0,
                "reflection_range": None,
                "refraction_range": (0, np.inf),
                "absorption_range": None,
            },
        )

    def add_sphere_surf_click(self):
        self.add_object(
            "sphere_surf",
            f"Сфера Поверхн. {len(self.scene_objects) + 1}",
            {
                "origin": (0, 0, 0),
                "rotation": (0, 0, 0),
                "radius": 3.0,
                "edge_radius": 1.5,
                "n": 1.5,
                "reflection_range": None,
                "refraction_range": (0, np.inf),
                "absorption_range": None,
            },
        )

    def add_cylinder_surf_click(self):
        self.add_object(
            "cylinder_surf",
            f"Цилиндр Поверхн. {len(self.scene_objects) + 1}",
            {
                "origin": (0, 0, 0),
                "rotation": (0, 0, 0),
                "radius": 1.0,
                "half_length": 1.0,
                "n": 1.5,
                "capping": False,
                "reflection_range": None,
                "refraction_range": (0, np.inf),
                "absorption_range": None,
            },
        )

    def add_lens_click(self):
        self.add_object(
            "lens",
            f"Линза {len(self.scene_objects) + 1}",
            {
                "origin": (0, 0, 0),
                "rotation": (0, 0, 0),
                "R1": 5.0,
                "R2": -5.0,
                "thickness": 0.5,
                "edge_radius": 1.0,
                "n": 1.5,
                "reflection_range": (0, np.inf),
                "refraction_range": (0, np.inf),
                "absorption_range": None,
            },
        )

    def add_hyperbolic_lens_click(self):
        self.add_object(
            "hyperbolic_lens",
            f"Гиперб. линза {len(self.scene_objects) + 1}",
            {
                "origin": (0, 0, 0),
                "rotation": (0, 0, 0),
                "radius_of_curvature": 5.0,
                "thickness": 0.5,
                "edge_radius": 1.0,
                "n": 1.5,
                "f_target": 10.0,
                "reflection_range": (0, np.inf),
                "refraction_range": (0, np.inf),
                "absorption_range": None,
            },
        )

    def add_emitter_click(self):
        self.add_object(
            "emitter",
            f"Источник {len(self.scene_objects) + 1}",
            {
                "origin": (0, 0, 0),
                "rotation": (0, 0, 0),
                "num_rays": 5,
                "min_offset": -0.5,
                "max_offset": 0.5,
                "wavelength": 550.0,
                "color": (1.0, 1.0, 0.0),
                "current_n": float(self.state.param_ambient_n),
                "energy": 1.0,
                "ray_class": Ray,
                "num_spectral_bands": 7,
            },
        )

    def add_mesh_click(self):
        self.add_object(
            "mesh",
            f"Меш {len(self.scene_objects) + 1}",
            {
                "origin": (0, 0, 0),
                "rotation": (0, 0, 0),
                "mesh_path": "Models/Prism.stl",
                "n": 1.5,
                "reflection_range": None,
                "refraction_range": (0, np.inf),
                "absorption_range": None,
                "scale_uniform": True,
                "scale_all": 1.0,
                "scale_x": 1.0,
                "scale_y": 1.0,
                "scale_z": 1.0,
            },
        )

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

    def _create_instance(self, obj_type, params):
        mat = params.get("material", Material())
        if obj_type == "lens":
            return UniversalLens(
                origin=params.get("origin", (0, 0, 0)),
                rotation_degrees=params.get("rotation", (0, 0, 0)),
                R1=params.get("R1"),
                R2=params.get("R2"),
                thickness=params.get("thickness", 0.5),
                edge_radius=params.get("edge_radius", 1.0),
                material=mat,
            )
        elif obj_type == "hyperbolic_lens":
            return HyperbolicLens(
                origin=params.get("origin", (0, 0, 0)),
                rotation_degrees=params.get("rotation", (0, 0, 0)),
                thickness=params.get("thickness", 0.5),
                edge_radius=params.get("edge_radius", 1.0),
                material=mat,
                f_target=params.get("f_target", 10.0),
            )
        elif obj_type == "emitter":
            return BeamEmitter(
                origin=params.get("origin", (0, 0, 0)),
                direction=np.array([1.0, 0.0, 0.0]),
                rotation_degrees=params.get("rotation", (0, 0, 0)),
                num_rays=params.get("num_rays", 5),
                min_offset=params.get("min_offset", -0.5),
                max_offset=params.get("max_offset", 0.5),
                color=params.get("color", (1.0, 1.0, 0.0)),
                wavelength=params.get("wavelength", 550),
                energy=params.get("energy", 1.0),
                current_n=float(params.get("current_n", self.state.param_ambient_n)),
                pool=self.pool,
                ray_class=params.get("ray_class", Ray),
                num_spectral_bands=params.get("num_spectral_bands", 7),
            )
        elif obj_type == "mesh":
            if params.get("scale_uniform", True):
                s_val = float(params.get("scale_all", 1.0))
                s_factors = (s_val, s_val, s_val)
            else:
                s_factors = (
                    float(params.get("scale_x", 1.0)),
                    float(params.get("scale_y", 1.0)),
                    float(params.get("scale_z", 1.0)),
                )

            # Передаем кортеж коэффициентов (X, Y, Z)
            return MeshSurface(
                mesh=params.get("mesh_path", ""),
                rotation_degrees=params.get("rotation", (0, 0, 0)),
                translation=params.get("origin", (0, 0, 0)),
                material=mat,
                scale_factors=s_factors,
            )
        elif obj_type == "plane":
            return PlaneSurface(
                point=params.get("origin", (0, 0, 0)),
                rotation_degrees=params.get("rotation", (0, 0, 0)),
                material=mat,
                edge_radius=params.get("edge_radius", 1.5),
                shape_type=params.get("shape_type", "circle"),
                width=params.get("width", 2.0),
                height=params.get("height", 2.0),
            )
        elif obj_type == "sphere_surf":
            return SphereSurface(
                radius=params.get("radius", 3.0),
                rotation_degrees=params.get("rotation", (0, 0, 0)),
                material=mat,
                edge_radius=params.get("edge_radius", 1.5),
                lens_origin=params.get("origin", (0, 0, 0)),
            )
        elif obj_type == "cylinder_surf":
            rot = R.from_euler("xyz", params.get("rotation", (0, 0, 0)), degrees=True).as_matrix()
            return CylinderSurface(
                center=params.get("origin", (0, 0, 0)),
                axis_dir=rot @ np.array([1.0, 0.0, 0.0]),
                radius=params.get("radius", 1.0),
                half_length=params.get("half_length", 1.0),
                material=mat,
                capping=params.get("capping", False),
            )

    def on_trace_mode_changed(self, **kwargs):
        mode = self.state.trace_mode
        if mode == "tree":
            self.ray_tracer.set_mode("tree")
            self.ray_tracer.mode.max_depth = 30
            self.ray_tracer.mode.total_limit = 2000
            self.ray_tracer.mode.offset_distance = 0.005
            self.ray_tracer.cloud.gamma = 0.9
        else:
            self.ray_tracer.set_mode("simple")
            self.ray_tracer.mode.max_bounces = 100
            self.ray_tracer.mode.offset_distance = 0.005
            self.ray_tracer.cloud.gamma = 1

        self.ray_tracer.mode.energy_color_type = 2
        self.update_scene()

    def on_scene_param_change(self, *args, **kwargs):
        if self._loading_state:
            return

        if "traced_rays_geometry" in self.plotter.actors:
            actor = self.plotter.actors["traced_rays_geometry"]
            actor.GetProperty().SetLineWidth(float(self.state.param_ray_thickness))
            actor.mapper.Modified()

        self._dirty_trace = True
        self._dirty_net = True
        self._ensure_ticker()

    def on_intervals_changed(self, *args, **kwargs):
        """Перечитывает интервалы throttle из state и применяет к контроллеру."""
        try:
            v = float(self.state.param_interval_visual)
            s = float(self.state.param_interval_shape)
            t = float(self.state.param_interval_trace)
            n = float(self.state.param_interval_net)
        except (TypeError, ValueError):
            return
        self._interval_visual = max(0.01, v)
        self._interval_shape = max(0.05, s)
        self._interval_trace = max(0.1, t)
        self._interval_net = max(0.01, n)

    def on_object_selected(self, *args, **kwargs):
        obj_id = args[0] if args else self.state.selected_object_id
        if not obj_id:
            return
        obj_entry = self._find_object(obj_id)
        if obj_entry:
            self._loading_state = True
            try:
                self._load_params_to_state(obj_entry)
            finally:
                self._loading_state = False

    def focus_on_selected_object(self, *args, **kwargs):
        """Перемещает фокус камеры на текущие координаты выбранного объекта."""
        obj_id = self.state.selected_object_id
        if not obj_id:
            return

        obj_entry = self._find_object(obj_id)
        if not obj_entry:
            return

        p = obj_entry["params"]
        origin = p.get("origin", (0.0, 0.0, 0.0))
        target_point = np.array(origin, dtype=float)

        camera = self.plotter.camera
        offset = np.array(camera.position) - np.array(camera.focal_point)
        dist = np.linalg.norm(offset)
        if dist < 1e-6:
            offset = np.array([5.0, 5.0, 5.0])
        else:
            offset = offset / dist * max(dist, 5.0)

        camera.focal_point = target_point
        camera.position = target_point + offset

        self.plotter.render()

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
        self.state.temp_pos_delta_x = 0.0
        self.state.temp_pos_delta_y = 0.0
        self.state.temp_pos_delta_z = 0.0
        if obj_entry["type"] == "lens":
            self.state.param_n = float(p.get("n", 1.5))
            self.state.param_empirical_coeff = float(p.get("empirical_coeff", 0.005))
            self.state.param_a_offset = float(p.get("a_offset", 0.01))
            self.state.param_R1 = float(p.get("R1", 5.0))
            self.state.param_R2 = float(p.get("R2", -5.0))
            self.state.param_thickness = float(p.get("thickness", 0.5))
            self.state.param_edge_radius = float(p.get("edge_radius", 1.0))
        elif obj_entry["type"] == "hyperbolic_lens":
            self.state.param_n = float(p.get("n", 1.5))
            self.state.param_empirical_coeff = float(p.get("empirical_coeff", 0.005))
            self.state.param_a_offset = float(p.get("a_offset", 0.01))
            self.state.param_f_target = float(p.get("f_target", 10.0))
            self.state.param_thickness = float(p.get("thickness", 0.5))
            self.state.param_edge_radius = float(p.get("edge_radius", 1.0))
        elif obj_entry["type"] == "emitter":
            self.state.param_num_rays = int(p.get("num_rays", 5))
            self.state.param_min_offset = float(p.get("min_offset", -0.5))
            self.state.param_max_offset = float(p.get("max_offset", 0.5))
            self.state.param_wavelength = float(p.get("wavelength", 550.0))
            self.state.param_current_n = float(p.get("current_n", 1.0))

            rc = p.get("ray_class", Ray)
            if rc.__name__ in ["Ray", "DispersiveRay", "WhiteRay"]:
                self.state.param_ray_class = rc.__name__
            else:
                self.state.param_ray_class = "Ray"

            self.state.param_num_spectral_bands = int(p.get("num_spectral_bands", 7))
            self.state.param_energy = float(p.get("energy", 1.0))
            color = p.get("color", "yellow")
            if isinstance(color, (tuple, list)):
                color_rgb = tuple(color[:3])
            else:
                color_rgb = to_rgb(color)
            self.state.param_ray_color = to_hex(color_rgb)

        elif obj_entry["type"] == "mesh":
            self.state.param_n = float(p.get("n", 1.5))
            self.state.param_empirical_coeff = float(p.get("empirical_coeff", 0.005))
            self.state.param_a_offset = float(p.get("a_offset", 0.01))
            self.state.param_mesh_path = str(p.get("mesh_path", ""))
            self.state.param_scale_uniform = bool(p.get("scale_uniform", True))
            self.state.param_scale_all = float(p.get("scale_all", 1.0))
            self.state.param_scale_x = float(p.get("scale_x", 1.0))
            self.state.param_scale_y = float(p.get("scale_y", 1.0))
            self.state.param_scale_z = float(p.get("scale_z", 1.0))
        elif obj_entry["type"] == "plane":
            self.state.param_n = float(p.get("n", 1.0))
            self.state.param_empirical_coeff = float(p.get("empirical_coeff", 0.005))
            self.state.param_a_offset = float(p.get("a_offset", 0.01))
            self.state.param_edge_radius = float(p.get("edge_radius", 1.5))
            self.state.param_plane_shape = str(p.get("shape_type", "circle"))
            self.state.param_plane_width = float(p.get("width", 2.0))
            self.state.param_plane_height = float(p.get("height", 2.0))
        elif obj_entry["type"] == "sphere_surf":
            self.state.param_radius = float(p.get("radius", 3.0))
            self.state.param_edge_radius = float(p.get("edge_radius", 1.5))
            self.state.param_n = float(p.get("n", 1.5))
            self.state.param_empirical_coeff = float(p.get("empirical_coeff", 0.005))
            self.state.param_a_offset = float(p.get("a_offset", 0.01))
        elif obj_entry["type"] == "cylinder_surf":
            self.state.param_radius = float(p.get("radius", 1.0))
            self.state.param_thickness = float(p.get("half_length", 1.0))
            self.state.param_n = float(p.get("n", 1.5))
            self.state.param_empirical_coeff = float(p.get("empirical_coeff", 0.005))
            self.state.param_a_offset = float(p.get("a_offset", 0.01))
            self.state.param_cylinder_capping = bool(p.get("capping", False))

        if obj_entry["type"] != "emitter":
            mat = p.get("material")
            if mat is None:
                # Миграция старого формата — если p["material"] нет, собираем из плоских полей
                mat = Material(
                    n=float(p.get("n", 1.5)),
                    empirical_coeff=float(p.get("empirical_coeff", 0.005)),
                    a_offset=float(p.get("a_offset", 0.01)),
                    refraction_range=parse_range(p.get("refraction_range", (0.0, np.inf))),
                    reflection_range=parse_range(p.get("reflection_range")),
                    absorption_range=parse_range(p.get("absorption_range")),
                )
                p["material"] = mat  # нормализуем на месте, чтобы _create_instance увидел

            self.state.param_mat_n = mat.n
            self.state.param_mat_empirical_coeff = mat.empirical_coeff
            self.state.param_mat_a_offset = mat.a_offset
            self.state.param_mat_object_color = to_hex(mat.color)
            self.state.param_mat_opacity = mat.opacity
            self.state.param_mat_reflectivity = mat.reflectivity

            for eff in ["reflection", "refraction", "absorption"]:
                rng = getattr(mat, f"{eff}_range")
                if rng is None:
                    self.state[f"param_mat_{eff}_enabled"] = False
                    self.state[f"param_mat_{eff}_min"] = 0.0
                    self.state[f"param_mat_{eff}_max"] = np.inf
                else:
                    self.state[f"param_mat_{eff}_enabled"] = True
                    self.state[f"param_mat_{eff}_min"] = float(rng[0])
                    self.state[f"param_mat_{eff}_max"] = float(rng[1])

    def update_object_transform(self):
        """Мгновенно двигает объект на GPU через user_matrix без пересоздания геометрии."""
        obj_id = self.state.selected_object_id
        obj_entry = self._find_object(obj_id)
        if not obj_entry or obj_id not in self.plotter.actors:
            return

        p = obj_entry["params"]

        new_origin = (float(self.state.param_pos_x), float(self.state.param_pos_y), float(self.state.param_pos_z))

        new_rotation = (float(self.state.param_rot_x), float(self.state.param_rot_y), float(self.state.param_rot_z))

        p["origin"] = new_origin
        p["rotation"] = new_rotation

        instance = self._create_instance(obj_entry["type"], p)
        obj_entry["instance"] = instance

        actor = self.plotter.actors[obj_id]
        mat = np.eye(4)
        mat[:3, :3] = R.from_euler("xyz", new_rotation, degrees=True).as_matrix()
        mat[:3, 3] = new_origin
        actor.user_matrix = mat

        self.update_scene()

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


server = get_server()
server.client_type = "vue3"
app = OpticsAppController(server)


transform_params = ["param_pos_x", "param_pos_y", "param_pos_z", "param_rot_x", "param_rot_y", "param_rot_z"]
for param in transform_params:
    server.state.change(param)(app.on_transform_param_change)

for axis in ["x", "y", "z"]:
    server.state.change(f"temp_pos_delta_{axis}")(app.on_temp_change)


shape_params = [
    "param_radius",
    "param_R1",
    "param_R2",
    "param_f_target",
    "param_thickness",
    "param_edge_radius",
    "param_num_rays",
    "param_min_offset",
    "param_max_offset",
    "param_wavelength",
    "param_current_n",
    "param_ray_class",
    "param_num_spectral_bands",
    "param_mesh_path",
    "param_plane_shape",
    "param_plane_width",
    "param_plane_height",
    "param_cylinder_capping",
    "param_scale_uniform",
    "param_scale_all",
    "param_scale_x",
    "param_scale_y",
    "param_scale_z",
    "param_energy",
    "param_ray_color",
    "param_mat_n",
    "param_mat_empirical_coeff",
    "param_mat_a_offset",
    "param_mat_object_color",
    "param_mat_opacity",
    "param_mat_reflectivity"
]
for param in shape_params:
    server.state.change(param)(app.on_shape_param_change)

for param in ["param_ambient_n", "param_ray_thickness"]:
    server.state.change(param)(app.on_scene_param_change)

for effect in ["reflection", "refraction", "absorption"]:
    server.state.change(f"param_mat_{effect}_enabled")(app.on_shape_param_change)
    server.state.change(f"param_mat_{effect}_min")(app.on_shape_param_change)
    server.state.change(f"param_mat_{effect}_max")(app.on_shape_param_change)

for key in ["param_interval_visual", "param_interval_shape", "param_interval_trace", "param_interval_net"]:
    server.state.change(key)(app.on_intervals_changed)

with VAppLayout(server) as layout:
    with vuetify.VMain():
        with vuetify.VContainer(fluid=True, classes="pa-0", style="height: 100vh; overflow: hidden;"):
            with vuetify.VRow(no_gutters=True, style="height: 100%;"):
                with vuetify.VCol(
                    cols=3,
                    classes="pa-4 bg-grey-darken-4",
                    style="height: 100%; overflow-y: auto; border-right: 1px solid #444; color: white;",
                ):
                    server.state.setdefault("save_dialog_visible", False)
                    server.state.setdefault("load_dialog_visible", False)
                    server.state.setdefault("file_to_save", "scene_1")
                    server.state.setdefault("file_to_load", "scene_1")

                    # Сохранение/Загрузка/Экспорт
                    with vuetify.VRow(class_="mb-2", no_gutters=True):
                        with vuetify.VCol(cols=4, class_="pr-1"):
                            with vuetify.VBtn(
                                color="success", block=True, size="small", click="save_dialog_visible = true"
                            ):
                                vuetify.VIcon("mdi-content-save", class_="mr-1")
                                "Сохранить"
                        with vuetify.VCol(cols=4, class_="pl-1"):
                            with vuetify.VBtn(
                                color="warning", block=True, size="small", click="load_dialog_visible = true"
                            ):
                                vuetify.VIcon("mdi-folder-open", class_="mr-1")
                                "Загрузить"
                        with vuetify.VCol(cols=4, class_="pl-1"):
                            with vuetify.VBtn(color="blue", block=True, size="small", click="trigger('export_scene')"):
                                vuetify.VLabel("Экспорт", class_="mr-1")
                                "Экспорт"

                    with vuetify.VDialog(v_model="save_dialog_visible", max_width="400px"):
                        with vuetify.VCard(class_="bg-grey-darken-3 text-white pa-4"):
                            vuetify.VCardTitle("Сохранить сцену", class_="px-0")
                            vuetify.VTextField(
                                v_model="file_to_save", label="Имя файла (без расширения)", class_="my-3", dense=True
                            )
                            with vuetify.VCardActions(class_="px-0"):
                                vuetify.VSpacer()
                                vuetify.VBtn("Отмена", color="grey", click="save_dialog_visible = false")
                                vuetify.VBtn(
                                    "ОК",
                                    color="success",
                                    click="trigger('save_scene_to_file', [file_to_save]); save_dialog_visible = false",
                                )

                    with vuetify.VDialog(v_model="load_dialog_visible", max_width="400px"):
                        with vuetify.VCard(class_="bg-grey-darken-3 text-white pa-4"):
                            vuetify.VCardTitle("Загрузить сцену", class_="px-0")
                            vuetify.VTextField(
                                v_model="file_to_load", label="Имя файла (.json)", class_="my-3", dense=True
                            )
                            with vuetify.VCardActions(class_="px-0"):
                                vuetify.VSpacer()
                                vuetify.VBtn("Отмена", color="grey", click="load_dialog_visible = false")
                                vuetify.VBtn(
                                    "ОК",
                                    color="warning",
                                    click="trigger('load_scene_from_file', [file_to_load]); load_dialog_visible = false",
                                )

                    vuetify.VDivider(class_="my-4")

                    # Список объектов
                    vuetify.VCardTitle("Объекты сцены", classes="text-h6 px-0")

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
                            with vuetify.VListItem(
                                v_for="obj in scene_objects_list",
                                key="obj.id",
                                active="selected_object_id == obj.id",
                                click="selected_object_id = obj.id",
                                style="cursor: pointer;",
                            ):
                                vuetify.VIcon("mdi-cube-outline", small=True, class_="mr-2")
                                vuetify.VListItemTitle("{{ obj.name }}")
                                with vuetify.Template(v_slot_append=True):
                                    vuetify.VBtn(
                                        icon="mdi-delete",
                                        size="x-small",
                                        variant="text",
                                        color="red",
                                        click="trigger('delete_object_event', [obj.id])",
                                    )

                    with vuetify.VRow(no_gutters=True, align="center", class_="mb-2"):
                        with vuetify.VCol(cols=9):
                            vuetify.VCardTitle(
                                "Параметры: {{ selected_object_id ? selected_object_id : 'не выбран' }}",
                                classes="text-subtitle-1 px-0 text-cyan-lighten-2",
                            )
                        with vuetify.VCol(cols=3, class_="text-right"):
                            vuetify.VBtn(
                                icon="mdi-crosshairs-gps",
                                size="small",
                                color="cyan-darken-2",
                                variant="flat",
                                click="trigger('focus_on_object_event')",
                                title="Сфокусировать камеру на объекте",
                            )

                    vuetify.VDivider(class_="my-4")

                    # Позиция/поворот
                    with vuetify.VContainer(v_if="selected_object_type != None", class_="pa-0"):
                        vuetify.VCardTitle("Позиция", class_="px-0", style="text-color: white")
                        for axis in ["x", "y", "z"]:
                            with vuetify.VRow(v_if="selected_object_type != None", no_gutters=True, align="center"):
                                with vuetify.VCol(cols=4):
                                    vuetify.VTextField(
                                        v_model=f"param_pos_{axis}",
                                        label=axis.upper(),
                                        type="number",
                                        dense=True,
                                        class_="mb-2",
                                    )
                                with vuetify.VCol(cols=8):
                                    vuetify.VSlider(
                                        v_model=f"temp_pos_delta_{axis}",
                                        min=-10,
                                        max=10,
                                        step=0.01,
                                        dense=True,
                                        hide_details=True,
                                        thumb_label="always",
                                        mouseup=f"trigger('apply_{axis}_delta')",
                                        mousedown="trigger('set_last_pos')",
                                    )

                        vuetify.VCardTitle("Поворот", class_="px-0", style="text-color: white")
                        for axis in ["x", "y", "z"]:
                            with vuetify.VRow(v_if="selected_object_type != None", no_gutters=True, align="center"):
                                with vuetify.VCol(cols=4):
                                    vuetify.VTextField(
                                        v_model=f"param_rot_{axis}",
                                        label=axis.upper(),
                                        type="number",
                                        dense=True,
                                        class_="mb-2",
                                    )
                                with vuetify.VCol(cols=8):
                                    vuetify.VSlider(
                                        v_model=f"param_rot_{axis}",
                                        min=-180,
                                        max=180,
                                        step=1,
                                        dense=True,
                                        hide_details=True,
                                    )

                        vuetify.VDivider(class_="my-4")

                    # Индивидуальные параметры объектов
                    with vuetify.VContainer(v_if="selected_object_type == 'plane'", class_="pa-0"):
                        vuetify.VCardTitle("Плоскость", class_="px-0", style="text-color: white")

                        with vuetify.VRadioGroup(
                            v_model="param_plane_shape", inline=True, class_="mb-2", hide_details=True
                        ):
                            vuetify.VRadio(label="Круг", value="circle", color="cyan")
                            vuetify.VRadio(label="Прямоугольник", value="rectangle", color="cyan")

                        with vuetify.VContainer(v_if="param_plane_shape == 'circle'", class_="pa-0"):
                            with vuetify.VRow(no_gutters=True, align="center"):
                                with vuetify.VCol(cols=4):
                                    vuetify.VTextField(
                                        v_model="param_edge_radius", label="Радиус", type="number", dense=True
                                    )
                                with vuetify.VCol(cols=8):
                                    vuetify.VSlider(
                                        v_model="param_edge_radius",
                                        min=0.1,
                                        max=5.0,
                                        step=0.1,
                                        dense=True,
                                        hide_details=True,
                                    )

                        with vuetify.VContainer(v_if="param_plane_shape == 'rectangle'", class_="pa-0"):
                            with vuetify.VRow(no_gutters=True, align="center", class_="mb-1"):
                                with vuetify.VCol(cols=4):
                                    vuetify.VTextField(
                                        v_model="param_plane_width", label="Ширина", type="number", dense=True
                                    )
                                with vuetify.VCol(cols=8):
                                    vuetify.VSlider(
                                        v_model="param_plane_width",
                                        min=0.1,
                                        max=10.0,
                                        step=0.1,
                                        dense=True,
                                        hide_details=True,
                                    )
                            with vuetify.VRow(no_gutters=True, align="center"):
                                with vuetify.VCol(cols=4):
                                    vuetify.VTextField(
                                        v_model="param_plane_height", label="Высота", type="number", dense=True
                                    )
                                with vuetify.VCol(cols=8):
                                    vuetify.VSlider(
                                        v_model="param_plane_height",
                                        min=0.1,
                                        max=10.0,
                                        step=0.1,
                                        dense=True,
                                        hide_details=True,
                                    )

                    with vuetify.VContainer(v_if="selected_object_type == 'sphere_surf'", class_="pa-0"):
                        vuetify.VCardTitle("Сферическая поверхность", class_="px-0", style="text-color: white")
                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(
                                    v_model="param_radius", label="Радиус кривизны", type="number", dense=True, step=0.5
                                )
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(
                                    v_model="param_radius", min=-20.0, max=20.0, step=0.5, dense=True, hide_details=True
                                )
                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(
                                    v_model="param_edge_radius", label="Апертура", type="number", dense=True, step=0.1
                                )
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(
                                    v_model="param_edge_radius",
                                    min=0.1,
                                    max=5.0,
                                    step=0.1,
                                    dense=True,
                                    hide_details=True,
                                )

                    with vuetify.VContainer(v_if="selected_object_type == 'cylinder_surf'", class_="pa-0"):
                        vuetify.VCardTitle("Цилиндрическая поверхность", class_="px-0", style="text-color: white")

                        vuetify.VSwitch(
                            v_model="param_cylinder_capping",
                            label="Включить основания (крышки)",
                            dense=True,
                            hide_details=True,
                            class_="mb-2",
                            color="orange",
                        )

                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(
                                    v_model="param_radius", label="Радиус", type="number", dense=True, step=0.1
                                )
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(
                                    v_model="param_radius", min=0.1, max=5.0, step=0.1, dense=True, hide_details=True
                                )
                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(
                                    v_model="param_thickness", label="Полудлина", type="number", dense=True, step=0.1
                                )
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(
                                    v_model="param_thickness", min=0.1, max=5.0, step=0.1, dense=True, hide_details=True
                                )

                    with vuetify.VContainer(v_if="selected_object_type == 'lens'", class_="pa-0"):
                        vuetify.VCardTitle("Сферическая линза", class_="px-0", style="text-color: white")

                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(v_model="param_R1", label="R1", type="number", dense=True, step=0.5)
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(
                                    v_model="param_R1", min=-20, max=20, step=0.5, dense=True, hide_details=True
                                )
                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(v_model="param_R2", label="R2", type="number", dense=True, step=0.5)
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(
                                    v_model="param_R2", min=-20, max=20, step=0.5, dense=True, hide_details=True
                                )
                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(
                                    v_model="param_thickness", label="Толщина", type="number", dense=True, step=0.1
                                )
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(
                                    v_model="param_thickness", min=0.1, max=5.0, step=0.1, dense=True, hide_details=True
                                )
                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(
                                    v_model="param_edge_radius", label="Апертура", type="number", dense=True, step=0.1
                                )
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(
                                    v_model="param_edge_radius",
                                    min=0.1,
                                    max=5.0,
                                    step=0.1,
                                    dense=True,
                                    hide_details=True,
                                )

                    with vuetify.VContainer(v_if="selected_object_type == 'hyperbolic_lens'", class_="pa-0"):
                        vuetify.VCardTitle("Гиперболическая линза", class_="px-0", style="text-color: white")
                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(
                                    v_model="param_thickness", label="Толщина", type="number", dense=True, step=0.1
                                )
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(
                                    v_model="param_thickness", min=0.1, max=5.0, step=0.1, dense=True, hide_details=True
                                )
                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(
                                    v_model="param_edge_radius", label="Апертура", type="number", dense=True, step=0.1
                                )
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(
                                    v_model="param_edge_radius",
                                    min=0.1,
                                    max=5.0,
                                    step=0.1,
                                    dense=True,
                                    hide_details=True,
                                )
                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(
                                    v_model="param_f_target", label="f", type="number", dense=True, step=0.1
                                )
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(
                                    v_model="param_f_target",
                                    min=-30.0,
                                    max=30.0,
                                    step=0.1,
                                    dense=True,
                                    hide_details=True,
                                )

                    with vuetify.VContainer(v_if="selected_object_type == 'mesh'", class_="pa-0"):
                        vuetify.VCardTitle("3D-модель", class_="px-0", style="text-color: white")
                        vuetify.VTextField(
                            v_model="param_mesh_path", label="Путь к файлу (.obj/.stl)", dense=True, class_="mb-2"
                        )

                        vuetify.VDivider(class_="my-2")
                        vuetify.VSwitch(
                            v_model="param_scale_uniform",
                            label="Пропорциональный масштаб",
                            dense=True,
                            hide_details=True,
                            class_="mb-2",
                            color="cyan",
                        )

                        with vuetify.VContainer(v_if="param_scale_uniform", class_="pa-0"):
                            with vuetify.VRow(no_gutters=True, align="center"):
                                with vuetify.VCol(cols=4):
                                    vuetify.VTextField(
                                        v_model="param_scale_all", label="Масштаб", type="number", dense=True, step=0.1
                                    )
                                with vuetify.VCol(cols=8):
                                    vuetify.VSlider(
                                        v_model="param_scale_all",
                                        min=0.1,
                                        max=10.0,
                                        step=0.1,
                                        dense=True,
                                        hide_details=True,
                                    )

                        with vuetify.VContainer(v_if="!param_scale_uniform", class_="pa-0"):
                            for axis in ["x", "y", "z"]:
                                with vuetify.VRow(no_gutters=True, align="center", class_="mb-1"):
                                    with vuetify.VCol(cols=4):
                                        vuetify.VTextField(
                                            v_model=f"param_scale_{axis}",
                                            label=f"Scale {axis.upper()}",
                                            type="number",
                                            dense=True,
                                            step=0.1,
                                        )
                                    with vuetify.VCol(cols=8):
                                        vuetify.VSlider(
                                            v_model=f"param_scale_{axis}",
                                            min=0.1,
                                            max=10.0,
                                            step=0.1,
                                            dense=True,
                                            hide_details=True,
                                        )

                    with vuetify.VContainer(v_if="selected_object_type == 'emitter'", class_="pa-0"):
                        vuetify.VCardTitle("Источник", class_="px-0", style="text-color: white")

                        vuetify.VSelect(
                            label="Тип луча",
                            v_model="param_ray_class",
                            items=("ray_classes_list",),
                            item_title="title",
                            item_value="value",
                            dense=True,
                            class_="mb-2",
                        )

                        with vuetify.VContainer(v_if="param_ray_class == 'WhiteRay'", class_="pa-0 mb-2"):
                            with vuetify.VRow(no_gutters=True, align="center"):
                                with vuetify.VCol(cols=4):
                                    vuetify.VTextField(
                                        v_model="param_num_spectral_bands",
                                        label="Спектр (кол-во)",
                                        type="number",
                                        dense=True,
                                        step=1,
                                    )
                                with vuetify.VCol(cols=8):
                                    vuetify.VSlider(
                                        v_model="param_num_spectral_bands",
                                        min=2,
                                        max=30,
                                        step=1,
                                        dense=True,
                                        hide_details=True,
                                    )

                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(
                                    v_model="param_num_rays", label="Кол-во лучей", type="number", dense=True, step=1
                                )
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(
                                    v_model="param_num_rays", min=1, max=20, step=1, dense=True, hide_details=True
                                )

                        with vuetify.VRow(v_if="param_ray_class != 'WhiteRay'", no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(
                                    v_model="param_wavelength", label="λ (нм)", type="number", dense=True, step=10
                                )
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(
                                    v_model="param_wavelength", min=380, max=780, step=10, dense=True, hide_details=True
                                )

                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(
                                    v_model="param_min_offset", label="Мин. смещ.", type="number", dense=True, step=0.1
                                )
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(
                                    v_model="param_min_offset",
                                    min=-3.0,
                                    max=0.0,
                                    step=0.1,
                                    dense=True,
                                    hide_details=True,
                                )
                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(
                                    v_model="param_max_offset", label="Макс. смещ.", type="number", dense=True, step=0.1
                                )
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(
                                    v_model="param_max_offset",
                                    min=0.0,
                                    max=3.0,
                                    step=0.1,
                                    dense=True,
                                    hide_details=True,
                                )

                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(
                                    v_model="param_energy", label="Энергия", type="number",
                                    dense=True, step=0.01, hide_details=True,
                                )
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(
                                    v_model="param_energy", min=0.0, max=2.0, step=0.01,
                                    dense=True, hide_details=True,
                                )

                        with vuetify.VContainer(v_if="param_ray_class == 'Ray'", class_="pa-0"):
                            vuetify.VColorPicker(
                                v_model="param_ray_color",
                                mode="hex",
                                hide_inputs=True,
                                width="100%",
                                elevation=0,
                            )

                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(
                                    v_model="param_current_n", label="n среды", type="number", dense=True, step=0.01
                                )
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(
                                    v_model="param_current_n",
                                    min=1.0,
                                    max=3.0,
                                    step=0.01,
                                    dense=True,
                                    hide_details=True,
                                )

                    vuetify.VDivider(class_="my-4")

                    # Генерация условия для всех объектов
                    object_types = ["plane", "sphere_surf", "cylinder_surf", "lens", "hyperbolic_lens", "mesh"]
                    mat_condition = ""
                    for object_type in object_types:
                        mat_condition += f"selected_object_type == '{object_type}' || "
                    mat_condition = mat_condition[:-4]

                    # Параметры материала
                    with vuetify.VContainer(v_if=mat_condition, class_="pa-0"):
                        vuetify.VCardTitle("Материал", class_="px-0", style="text-color: white")

                        with vuetify.VRow(no_gutters=True, align="center", class_="mb-2"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(v_model="param_mat_n", label="n", type="number", dense=True, step=0.01)
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(
                                    v_model="param_mat_n", min=1.0, max=2.5, step=0.01, dense=True, hide_details=True
                                )

                        with vuetify.VRow(no_gutters=True, align="center", class_="mt-2"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(
                                    v_model="param_mat_empirical_coeff", label="B", type="number", dense=True, step=0.001
                                )
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(
                                    v_model="param_mat_empirical_coeff", min=0, max=0.05, step=0.001, dense=True, hide_details=True
                                )

                        with vuetify.VRow(no_gutters=True, align="center", class_="mt-2"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(
                                    v_model="param_mat_a_offset", label="A", type="number", dense=True, step=0.001
                                )
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(
                                    v_model="param_mat_a_offset", min=0, max=0.05, step=0.001, dense=True, hide_details=True
                                )

                        vuetify.VColorPicker(
                            v_model="param_mat_object_color", mode="hex",
                            hide_inputs=True, width="100%", elevation=0,
                        )

                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(v_model="param_mat_opacity", label="Прозрачность",
                                                   type="number", dense=True, step=0.05, hide_details=True)
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(v_model="param_mat_opacity", min=0.0, max=1.0, step=0.01,
                                                dense=True, hide_details=True)
                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(v_model="param_mat_reflectivity", label="Зеркальность",
                                                   type="number", dense=True, step=0.05, hide_details=True)
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(v_model="param_mat_reflectivity", min=0.0, max=1.0, step=0.05,
                                                dense=True, hide_details=True)

                        effects = [
                            {"key": "refraction", "label": "Преломление"},
                            {"key": "reflection", "label": "Отражение"},
                            {"key": "absorption", "label": "Поглощение"},
                        ]

                        for eff in effects:
                            k = eff["key"]
                            vuetify.VCheckbox(
                                v_model=f"param_mat_{k}_enabled",
                                v_if=f"selected_object_type != None",
                                label=eff["label"],
                                dense=True,
                                hide_details=True,
                            )

                            with vuetify.VRow(
                                    v_if=f"param_mat_{k}_enabled && selected_object_type != None",
                                    no_gutters=True,
                                    align="center",
                                    class_="pl-4 mb-2",
                            ):
                                with vuetify.VCol(cols=5, class_="pr-1"):
                                    vuetify.VTextField(
                                        v_model=f"param_mat_{k}_min",
                                        label="Мин",
                                        type="number",
                                        dense=True,
                                        hide_details=True,
                                    )
                                with vuetify.VCol(cols=5, class_="px-1"):
                                    vuetify.VTextField(
                                        v_model=f"param_mat_{k}_max", label="Макс", dense=True, hide_details=True
                                    )
                                with vuetify.VCol(cols=2, class_="pl-1"):
                                    vuetify.VBtn(
                                        icon="mdi-infinity",
                                        size="small",
                                        color="grey-darken-2",
                                        variant="flat",
                                        click=f"param_mat_{k}_max = Infinity",
                                    )

                        vuetify.VDivider(class_="my-4")

                    # Параметры сцены
                    with vuetify.VContainer(class_="pa-0"):
                        vuetify.VCardTitle("Параметры сцены", class_="px-0", style="text-color: white")

                        vuetify.VSelect(
                            label="Режим трассировки",
                            v_model="trace_mode",
                            items=("trace_modes",),
                            item_title="title",
                            item_value="value",
                            dense=True,
                            class_="mt-2",
                        )

                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(
                                    v_model="param_ambient_n", label="n среды", type="number",
                                    dense=True, step=0.01, hide_details=True,
                                )
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(
                                    v_model="param_ambient_n", min=1.0, max=3.0, step=0.01,
                                    dense=True, hide_details=True,
                                )
                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(
                                    v_model="param_ray_thickness", label="толщина лучей", type="number",
                                    dense=True, step=0.01, hide_details=True,
                                )
                            with vuetify.VCol(cols=8):
                                vuetify.VSlider(
                                    v_model="param_ray_thickness", min=1.0, max=10, step=0.01,
                                    dense=True, hide_details=True,
                                )


                    vuetify.VDivider(class_="my-4")

                    # Частота обновления
                    with vuetify.VContainer(class_="pa-0"):
                        vuetify.VCardTitle("Частота обновления", class_="px-0", style="text-color: white")

                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(
                                    v_model="param_interval_visual",
                                    label="Позиция (с)",
                                    type="number",
                                    dense=True,
                                    step=0.01,
                                    hide_details=True,
                                )
                            with vuetify.VCol(cols=6):
                                vuetify.VSlider(
                                    v_model="param_interval_visual",
                                    min=0.01,
                                    max=0.2,
                                    step=0.01,
                                    dense=True,
                                    hide_details=True,
                                )

                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(
                                    v_model="param_interval_shape",
                                    label="Форма (с)",
                                    type="number",
                                    dense=True,
                                    step=0.05,
                                    hide_details=True,
                                )
                            with vuetify.VCol(cols=6):
                                vuetify.VSlider(
                                    v_model="param_interval_shape",
                                    min=0.05,
                                    max=1.0,
                                    step=0.05,
                                    dense=True,
                                    hide_details=True,
                                )

                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(
                                    v_model="param_interval_trace",
                                    label="Трассировка (с)",
                                    type="number",
                                    dense=True,
                                    step=0.1,
                                    hide_details=True,
                                )
                            with vuetify.VCol(cols=6):
                                vuetify.VSlider(
                                    v_model="param_interval_trace",
                                    min=0.1,
                                    max=3.0,
                                    step=0.1,
                                    dense=True,
                                    hide_details=True,
                                )

                        with vuetify.VRow(no_gutters=True, align="center"):
                            with vuetify.VCol(cols=4):
                                vuetify.VTextField(
                                    v_model="param_interval_net",
                                    label="Сеть (с)",
                                    type="number",
                                    dense=True,
                                    step=0.1,
                                    hide_details=True,
                                )
                            with vuetify.VCol(cols=6):
                                vuetify.VSlider(
                                    v_model="param_interval_net",
                                    min=0.01,
                                    max=1.0,
                                    step=0.01,
                                    dense=True,
                                    hide_details=True,
                                )

                # 3D-холст
                with vuetify.VCol(cols=9, style="height: 100%; overflow: hidden;"):
                    ui_view = plotter_ui(
                        app.plotter,
                        mode="client",
                        add_menu=False,
                        interactor_style="Terrain",
                    )
                    app.ctrl.view_update = ui_view.update

app.update_scene()

if __name__ == "__main__":
    port_env = int(os.environ.get("PORT", 8085))
    server.cli.set_defaults(
        host="0.0.0.0",
        port=port_env,
        timeout=0,
        open_browser=False,
    )

    print(f"--> Запуск на порту {port_env}...", flush=True)
    server.start()
