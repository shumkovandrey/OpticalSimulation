# tests/golden_test.py
"""
Golden-тест для проверки поведения трассировщика лучей после рефакторинга.

Использование:
    python tests/golden_test.py --generate    # один раз ДО рефакторинга
    python tests/golden_test.py --verify      # после каждого шага рефакторинга

Покрытие:
    - PlaneSurface (круг/прямоугольник, преломление/отражение/поглощение/ТИР)
    - SphereSurface (выпуклая/вогнутая)
    - CylinderSurface (бок + крышки)
    - UniversalLens (двояко-/плоско-выпуклая, повёрнутая)
    - HyperbolicLens (асферическая коллимация)
    - MeshSurface (треугольная призма)
    - ThinLens (параксиальная)
    - Ray, DispersiveRay, WhiteRay, BeamEmitter
    - SimpleMode, TreeMode
    - Френелевское отражение, полное внутреннее отражение, поглощение
    - Дисперсия показателя преломления

ВАЖНО:
    Этот тест фиксирует ТЕКУЩЕЕ поведение, включая известные баги
    (например, некорректная обработка ТИР в _trace_simple).
    Если при рефакторинге баг будет исправлен — тест зафиксирует изменение.
    Это ожидаемо: после осознанного фикса нужно перегенерировать baseline
    через --generate и сравнить diff вручную.
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

# ВАЖНО: выставляем флаги ДО импорта main.py,
# потому что main.py создаёт глобальный Plotter на уровне модуля.
import pyvista as pv
pv.OFF_SCREEN = True
pv.global_theme.allow_empty_mesh = True

import trimesh

from core.rays import Ray, DispersiveRay, WhiteRay, RayPool
from surfaces.plane import PlaneSurface
from surfaces.sphere import SphereSurface
from surfaces.cylinder import CylinderSurface
from surfaces.mesh import MeshSurface
from surfaces.thin_lens import ThinLens
from surfaces.universal_lens import UniversalLens
from surfaces.hyperbolic_lens import HyperbolicLens
from tracing.modes import SimpleMode, TreeMode
from emitters import BeamEmitter
from visualization import RayTracer


GOLDEN_PATH = Path(__file__).resolve().parent / "golden.json"

# Допуски сравнения. Жёсткие, т.к. numba-ядро детерминировано.
ATOL_POS = 1e-9
ATOL_ENERGY = 1e-10
ATOL_COLOR = 1e-9


# --------------------------------------------------------------------------
# Сериализация
# --------------------------------------------------------------------------
def serialize_segments(segments):
    """Приводит список Segment к JSON-сериализуемому виду."""
    result = []
    for seg in segments:
        start = np.asarray(seg.start, dtype=float)
        end = np.asarray(seg.end, dtype=float)
        energy = float(seg.energy)
        color = seg.color
        if isinstance(color, str):
            color_ser = color
        else:
            color_ser = [float(c) for c in color]
        result.append({
            "start": start.tolist(),
            "end": end.tolist(),
            "energy": energy,
            "color": color_ser,
        })
    return result


# --------------------------------------------------------------------------
# Инфраструктура трассировки
# --------------------------------------------------------------------------
def _make_tracer(mode="simple", **mode_kwargs):
    plotter = pv.Plotter(off_screen=True)
    plotter.set_background("black")
    # Пул с initial_size=0: первый acquire создаёт новый,
    # последующие переиспользуют — покрывает оба пути RayPool.
    pool = RayPool(initial_size=0)
    tracer = RayTracer(plotter, mode=mode, pool=pool)

    if mode == "simple":
        tracer.mode.max_bounces = mode_kwargs.get("max_bounces", 20)
        tracer.mode.offset_distance = mode_kwargs.get("offset_distance", 1e-3)
        tracer.mode.prioritize_refraction = mode_kwargs.get(
            "prioritize_refraction", True
        )
    elif mode == "tree":
        tracer.mode.max_depth = mode_kwargs.get("max_depth", 10)
        tracer.mode.min_energy = mode_kwargs.get("min_energy", 0.01)
        tracer.mode.offset_distance = mode_kwargs.get("offset_distance", 1e-3)
        tracer.mode.total_limit = mode_kwargs.get("total_limit", 5000)
    else:
        raise ValueError(f"unknown mode: {mode}")

    tracer.mode.energy_color_type = 1
    return tracer


def trace_scene(elements, rays, mode="simple", **mode_kwargs):
    """Прогоняет лучи через сцену и возвращает сериализованные сегменты."""
    tracer = _make_tracer(mode=mode, **mode_kwargs)
    for el in elements:
        tracer.add_elements(el)
    for r in rays:
        tracer.add_ray(r)
    segments = tracer.trace_all()
    return serialize_segments(segments)


def make_simple_prism_mesh():
    """Простая треугольная призма для MeshSurface."""
    vertices = np.array([
        [0.0, 0.0, -1.0],
        [0.0, 0.0,  1.0],
        [2.0, 0.0, -1.0],
        [2.0, 0.0,  1.0],
        [1.0, 1.5, -1.0],
        [1.0, 1.5,  1.0],
    ])
    faces = np.array([
        [0, 2, 4],
        [1, 5, 3],
        [0, 4, 5], [0, 5, 1],
        [2, 3, 5], [2, 5, 4],
        [0, 1, 3], [0, 3, 2],
    ])
    return trimesh.Trimesh(vertices=vertices, faces=faces)


# --------------------------------------------------------------------------
# Сценарии
# --------------------------------------------------------------------------
def scenario_plane_refraction():
    """Круглая плоскость, наклонный луч, преломление."""
    surf = PlaneSurface(
        point=(0, 0, 0), normal=(1, 0, 0), n_inside=1.5,
        edge_radius=5.0,
        reflection_range=None, refraction_range=(0, np.inf),
        absorption_range=None,
    )
    ray = Ray(origin=(-5.0, -1.0, 0.0), direction=(1.0, 0.3, 0.0),
              current_n=1.0)
    return trace_scene([surf], [ray], mode="simple",
                       prioritize_refraction=True)


def scenario_plane_rectangle():
    """Прямоугольная плоскость — один луч проходит, второй мимо."""
    surf = PlaneSurface(
        point=(0, 0, 0), normal=(1, 0, 0), n_inside=1.5,
        shape_type="rectangle", width=2.0, height=2.0,
        reflection_range=None, refraction_range=(0, np.inf),
        absorption_range=None,
    )
    rays = [
        Ray(origin=(-5.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0)),
        Ray(origin=(-5.0, 2.0, 0.0), direction=(1.0, 0.0, 0.0)),
    ]
    return trace_scene([surf], rays, mode="simple")


def scenario_plane_tir_simple():
    """
    Полное внутреннее отражение, SimpleMode.
    ВНИМАНИЕ: фиксирует текущее (багованное) поведение _trace_simple —
    current_n ошибочно меняется на 1.0 при ТИР. См. комментарий в шапке.
    """
    surf = PlaneSurface(
        point=(0, 0, 0), normal=(1, 0, 0), n_inside=1.0,
        edge_radius=5.0,
        reflection_range=None, refraction_range=(0, np.inf),
        absorption_range=None,
    )
    ray = Ray(origin=(-1.0, 0.0, 0.0), direction=(1.0, 2.0, 0.0),
              current_n=1.5)
    return trace_scene([surf], [ray], mode="simple",
                       prioritize_refraction=False)


def scenario_plane_tir_tree():
    """ТИР в TreeMode — обрабатывается корректно через split_ray."""
    surf = PlaneSurface(
        point=(0, 0, 0), normal=(1, 0, 0), n_inside=1.0,
        edge_radius=5.0,
        reflection_range=(0, np.inf), refraction_range=(0, np.inf),
        absorption_range=None,
    )
    ray = Ray(origin=(-1.0, 0.0, 0.0), direction=(1.0, 2.0, 0.0),
              current_n=1.5)
    return trace_scene([surf], [ray], mode="tree",
                       max_depth=2, min_energy=0.0)


def scenario_plane_absorption():
    """Объект поглощает луч — отрезок прерывается в точке удара."""
    surf = PlaneSurface(
        point=(0, 0, 0), normal=(1, 0, 0), n_inside=1.5,
        edge_radius=5.0,
        reflection_range=None, refraction_range=None,
        absorption_range=(0, np.inf),
    )
    ray = Ray(origin=(-5.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    return trace_scene([surf], [ray], mode="simple")


def scenario_sphere_refraction():
    """Выпуклая сферическая поверхность, преломление."""
    surf = SphereSurface(
        radius=5.0, n_inside=1.5, edge_radius=2.0,
        lens_origin=(0, 0, 0), lens_axis=(1, 0, 0),
        reflection_range=None, refraction_range=(0, np.inf),
        absorption_range=None,
    )
    ray = Ray(origin=(-5.0, 0.5, 0.0), direction=(1.0, 0.0, 0.0))
    return trace_scene([surf], [ray], mode="simple")


def scenario_sphere_reflection():
    """Вогнутое сферическое зеркало."""
    surf = SphereSurface(
        radius=-10.0, n_inside=1.0, edge_radius=3.0,
        lens_origin=(0, 0, 0), lens_axis=(1, 0, 0),
        reflection_range=(0, np.inf), refraction_range=None,
        absorption_range=None,
    )
    ray = Ray(origin=(-5.0, 0.5, 0.0), direction=(1.0, 0.0, 0.0))
    return trace_scene([surf], [ray], mode="simple")


def scenario_cylinder_side():
    """Цилиндр без крышек — попадание в боковую поверхность."""
    cyl = CylinderSurface(
        center=(0, 0, 0), axis_dir=(1, 0, 0),
        radius=1.0, half_length=2.0, n_inside=1.5, capping=False,
        reflection_range=None, refraction_range=(0, np.inf),
        absorption_range=None,
    )
    ray = Ray(origin=(-5.0, 1.0, 0.0), direction=(1.0, 0.0, 0.0))
    return trace_scene([cyl], [ray], mode="simple")


def scenario_cylinder_capping():
    """Цилиндр с крышками — луч проходит сквозь торец и бок."""
    cyl = CylinderSurface(
        center=(0, 0, 0), axis_dir=(1, 0, 0),
        radius=1.5, half_length=1.0, n_inside=1.5, capping=True,
        reflection_range=None, refraction_range=(0, np.inf),
        absorption_range=None,
    )
    ray = Ray(origin=(-5.0, 0.3, 0.0), direction=(1.0, 0.0, 0.0))
    return trace_scene([cyl], [ray], mode="simple")


def scenario_universal_lens_double_convex():
    """Двояковыпуклая линза: осевой, параксиальный и краевой лучи."""
    lens = UniversalLens(
        origin=(0, 0, 0), rotation_degrees=(0, 0, 0),
        R1=5.0, R2=-5.0, thickness=1.0, edge_radius=2.0, n=1.5,
        reflection_range=None, refraction_range=(0, np.inf),
        absorption_range=None,
    )
    rays = [
        Ray(origin=(-5.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0)),
        Ray(origin=(-5.0, 0.5, 0.0), direction=(1.0, 0.0, 0.0)),
        Ray(origin=(-5.0, 1.5, 0.0), direction=(1.0, 0.0, 0.0)),
    ]
    return trace_scene(lens.get_surfaces(), rays, mode="simple",
                       prioritize_refraction=True, max_bounces=10)


def scenario_universal_lens_plano_convex():
    """Плоско-выпуклая линза (R2=None)."""
    lens = UniversalLens(
        origin=(0, 0, 0), rotation_degrees=(0, 0, 0),
        R1=5.0, R2=None, thickness=1.0, edge_radius=2.0, n=1.5,
        reflection_range=None, refraction_range=(0, np.inf),
        absorption_range=None,
    )
    rays = [Ray(origin=(-5.0, 0.5, 0.0), direction=(1.0, 0.0, 0.0))]
    return trace_scene(lens.get_surfaces(), rays, mode="simple")


def scenario_universal_lens_rotated():
    """Линза, повёрнутая на 45° вокруг Z — проверка трансформаций."""
    lens = UniversalLens(
        origin=(0, 0, 0), rotation_degrees=(0, 0, 45),
        R1=5.0, R2=-5.0, thickness=1.0, edge_radius=2.0, n=1.5,
        reflection_range=None, refraction_range=(0, np.inf),
        absorption_range=None,
    )
    ray = Ray(origin=(-5.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    return trace_scene(lens.get_surfaces(), [ray], mode="simple")


def scenario_hyperbolic_lens_collimation():
    """Гиперболическая линза — три луча на разной высоте."""
    lens = HyperbolicLens(
        origin=(0, 0, 0), rotation_degrees=(0, 0, 0),
        thickness=1.0, edge_radius=1.0, n=1.5, f_target=10.0,
        reflection_range=None, refraction_range=(0, np.inf),
        absorption_range=None,
    )
    rays = [
        Ray(origin=(-10.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0)),
        Ray(origin=(-10.0, 0.5, 0.0), direction=(1.0, 0.0, 0.0)),
        Ray(origin=(-10.0, 0.9, 0.0), direction=(1.0, 0.0, 0.0)),
    ]
    return trace_scene(lens.get_surfaces(), rays, mode="simple")


def scenario_mesh_prism():
    """Треугольная призма (MeshSurface) — преломление через mesh."""
    mesh = make_simple_prism_mesh()
    mesh_surf = MeshSurface(
        mesh=mesh, rotation_degrees=(0, 0, 0), translation=(0, 0, 0),
        n_inside=1.5,
        reflection_range=None, refraction_range=(0, np.inf),
        absorption_range=None,
    )
    ray = Ray(origin=(-2.0, 0.5, 0.0), direction=(1.0, 0.0, 0.0))
    return trace_scene([mesh_surf], [ray], mode="simple",
                       prioritize_refraction=True)


def scenario_thin_lens():
    """Тонкая линза — параксиальное отклонение."""
    lens = ThinLens(
        center=(0, 0, 0), focal_length=5.0, edge_radius=2.0,
        axis_dir=(1, 0, 0), refraction_range=(0, np.inf),
    )
    rays = [
        Ray(origin=(-5.0, 0.5, 0.0), direction=(1.0, 0.0, 0.0)),
        Ray(origin=(-5.0, -0.5, 0.0), direction=(1.0, 0.0, 0.0)),
    ]
    return trace_scene([lens], rays, mode="simple")


def scenario_dispersive_ray():
    """DispersiveRay: две длины волны → разные n → разные углы."""
    lens = UniversalLens(
        origin=(0, 0, 0), rotation_degrees=(0, 0, 0),
        R1=5.0, R2=-5.0, thickness=1.0, edge_radius=2.0, n=1.5,
        reflection_range=None, refraction_range=(0, np.inf),
        absorption_range=None,
    )
    rays = [
        DispersiveRay(origin=(-5.0, 0.5, 0.0),
                      direction=(1.0, 0.0, 0.0), wavelength=450.0),
        DispersiveRay(origin=(-5.0, 0.5, 0.0),
                      direction=(1.0, 0.0, 0.0), wavelength=650.0),
    ]
    return trace_scene(lens.get_surfaces(), rays, mode="simple")


def scenario_white_ray():
    """WhiteRay разворачивается в 3 спектральные компоненты."""
    lens = UniversalLens(
        origin=(0, 0, 0), rotation_degrees=(0, 0, 0),
        R1=5.0, R2=-5.0, thickness=1.0, edge_radius=2.0, n=1.5,
        reflection_range=None, refraction_range=(0, np.inf),
        absorption_range=None,
    )
    white = WhiteRay(
        origin=(-5.0, 0.5, 0.0), direction=(1.0, 0.0, 0.0),
        num_spectral_bands=3, energy=1.0, current_n=1.0,
    )
    return trace_scene(lens.get_surfaces(), [white], mode="simple")


def scenario_beam_emitter():
    """BeamEmitter + UniversalLens."""
    emitter = BeamEmitter(
        origin=(-5.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0),
        rotation_degrees=(0, 0, 0), pool=None,
        num_rays=3, min_offset=-0.5, max_offset=0.5,
        color="yellow", wavelength=550.0, energy=1.0, current_n=1.0,
        ray_class=Ray,
    )
    rays = emitter.emit()
    lens = UniversalLens(
        origin=(0, 0, 0), R1=5.0, R2=-5.0, thickness=1.0,
        edge_radius=2.0, n=1.5,
        reflection_range=None, refraction_range=(0, np.inf),
        absorption_range=None,
    )
    return trace_scene(lens.get_surfaces(), rays, mode="simple")


def scenario_tree_mode_reflections():
    """TreeMode: линза с отражениями, ветвление по Френелю."""
    lens = UniversalLens(
        origin=(0, 0, 0), rotation_degrees=(0, 0, 0),
        R1=5.0, R2=-5.0, thickness=1.0, edge_radius=2.0, n=1.5,
        reflection_range=(0, np.inf), refraction_range=(0, np.inf),
        absorption_range=None,
    )
    ray = Ray(origin=(-5.0, 0.5, 0.0), direction=(1.0, 0.0, 0.0))
    return trace_scene(lens.get_surfaces(), [ray], mode="tree",
                       max_depth=4, min_energy=0.001, offset_distance=1e-3)


def scenario_fresnel_energy_tree():
    """TreeMode: проверка, что на границе R + T = 1."""
    surf = PlaneSurface(
        point=(0, 0, 0), normal=(1, 0, 0), n_inside=1.5,
        edge_radius=5.0,
        reflection_range=(0, np.inf), refraction_range=(0, np.inf),
        absorption_range=None,
    )
    ray = Ray(origin=(-5.0, 1.0, 0.0), direction=(1.0, 0.5, 0.0))
    return trace_scene([surf], [ray], mode="tree",
                       max_depth=2, min_energy=0.0)


def scenario_plane_tir_multi_surface():
    """
    Многослойный ТИР в SimpleMode.

    Луч стартует внутри материала (current_n=1.5), ТИРится на плоскости A,
    отражается к плоскости B, выходит в вакуум. Проверяет, что после ТИР
    current_n остаётся 1.5 (а не становится 1.0) — иначе на B преломление
    будет неверным.
    """
    surf_A = PlaneSurface(
        point=(1, 0, 0), normal=(1, 0, 0), n_inside=1.0,
        edge_radius=20.0, reflection_range=None,
        refraction_range=(0, np.inf), absorption_range=None,
    )
    surf_B = PlaneSurface(
        point=(0, 5, 0), normal=(0, -1, 0), n_inside=1.0,
        edge_radius=20.0, reflection_range=None,
        refraction_range=(0, np.inf), absorption_range=None,
    )
    ray = Ray(origin=(0, 0, 0), direction=(1, 2, 0), current_n=1.5)
    return trace_scene([surf_A, surf_B], [ray], mode="simple",
                       max_bounces=5, prioritize_refraction=True)


ALL_SCENARIOS = {
    "plane_refraction": scenario_plane_refraction,
    "plane_rectangle": scenario_plane_rectangle,
    "plane_tir_simple": scenario_plane_tir_simple,
    "plane_tir_tree": scenario_plane_tir_tree,
    "plane_absorption": scenario_plane_absorption,
    "sphere_refraction": scenario_sphere_refraction,
    "sphere_reflection": scenario_sphere_reflection,
    "cylinder_side": scenario_cylinder_side,
    "cylinder_capping": scenario_cylinder_capping,
    "universal_lens_double_convex": scenario_universal_lens_double_convex,
    "universal_lens_plano_convex": scenario_universal_lens_plano_convex,
    "universal_lens_rotated": scenario_universal_lens_rotated,
    "hyperbolic_lens": scenario_hyperbolic_lens_collimation,
    "mesh_prism": scenario_mesh_prism,
    "thin_lens": scenario_thin_lens,
    "dispersive_ray": scenario_dispersive_ray,
    "white_ray": scenario_white_ray,
    "beam_emitter": scenario_beam_emitter,
    "tree_mode_reflections": scenario_tree_mode_reflections,
    "fresnel_energy_tree": scenario_fresnel_energy_tree,
    "plane_tir_multi_surface": scenario_plane_tir_multi_surface,
}


# --------------------------------------------------------------------------
# Прогон и сравнение
# --------------------------------------------------------------------------
def run_all():
    """Возвращает dict: имя сценария -> сериализованные сегменты или ошибка."""
    results = {}
    for name, fn in ALL_SCENARIOS.items():
        try:
            results[name] = fn()
        except Exception as e:
            results[name] = {"__error__": f"{type(e).__name__}: {e}"}
    return results


def compare_segments(golden, current, path=""):
    """Возвращает список строк-ошибок (пустой, если всё совпало)."""
    errors = []

    if isinstance(golden, dict) and "__error__" in golden:
        if not isinstance(current, dict) or current.get("__error__") != golden["__error__"]:
            errors.append(
                f"{path}: ожидалась ошибка '{golden['__error__']}', получено {current!r}"
            )
        return errors

    if isinstance(current, dict) and "__error__" in current:
        errors.append(
            f"{path}: получена ошибка '{current['__error__']}', ожидались данные"
        )
        return errors

    if len(golden) != len(current):
        errors.append(
            f"{path}: разное число сегментов: {len(golden)} vs {len(current)}"
        )
        return errors

    for i, (g, c) in enumerate(zip(golden, current)):
        for key in ("start", "end"):
            g_arr = np.asarray(g[key])
            c_arr = np.asarray(c[key])
            if not np.allclose(g_arr, c_arr, atol=ATOL_POS, rtol=0):
                diff = float(np.max(np.abs(g_arr - c_arr)))
                errors.append(
                    f"{path}[{i}].{key}: {g_arr.tolist()} vs {c_arr.tolist()} "
                    f"(max diff = {diff:.3e})"
                )

        if not np.isclose(g["energy"], c["energy"], atol=ATOL_ENERGY, rtol=0):
            diff = abs(g["energy"] - c["energy"])
            errors.append(
                f"{path}[{i}].energy: {g['energy']} vs {c['energy']} "
                f"(diff = {diff:.3e})"
            )

        gc = g["color"]
        cc = c["color"]
        if isinstance(gc, str) and isinstance(cc, str):
            if gc != cc:
                errors.append(f"{path}[{i}].color: {gc!r} vs {cc!r}")
        elif isinstance(gc, list) and isinstance(cc, list):
            gc_arr = np.asarray(gc)
            cc_arr = np.asarray(cc)
            if not np.allclose(gc_arr, cc_arr, atol=ATOL_COLOR, rtol=0):
                diff = float(np.max(np.abs(gc_arr - cc_arr)))
                errors.append(
                    f"{path}[{i}].color: {gc} vs {cc} "
                    f"(max diff = {diff:.3e})"
                )
        else:
            if gc != cc:
                errors.append(f"{path}[{i}].color mismatch: {gc!r} vs {cc!r}")

    return errors


def _summary(results):
    """Печать краткой сводки по результатам прогона."""
    total_segments = 0
    total_energy = 0.0
    for name, segs in results.items():
        if isinstance(segs, dict) and "__error__" in segs:
            print(f"  {name:32s}  ОШИБКА: {segs['__error__']}")
        else:
            n = len(segs)
            energy = sum(s["energy"] for s in segs)
            total_segments += n
            total_energy += energy
            print(f"  {name:32s}  segments={n:3d}  energy={energy:.6f}")
    print(f"  {'ИТОГО':32s}  segments={total_segments:3d}  energy={total_energy:.6f}")


# --------------------------------------------------------------------------
# Режимы CLI
# --------------------------------------------------------------------------
def do_generate():
    print("=== Генерация baseline ===")
    results = run_all()
    _summary(results)

    with open(GOLDEN_PATH, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "scenarios": results}, f, indent=2)
    print(f"\nBaseline сохранён в {GOLDEN_PATH}")
    return 0


def do_verify():
    if not GOLDEN_PATH.exists():
        print(f"Файл {GOLDEN_PATH} не найден. Сначала запустите --generate")
        return 1

    print(f"=== Проверка против {GOLDEN_PATH.name} ===")
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        golden = json.load(f)["scenarios"]

    current = run_all()
    _summary(current)

    all_errors = {}
    for name, golden_segs in golden.items():
        if name not in current:
            all_errors[name] = [f"сценарий '{name}' отсутствует в текущем прогоне"]
            continue
        errs = compare_segments(golden_segs, current[name], path=name)
        if errs:
            all_errors[name] = errs

    for name in current:
        if name not in golden:
            print(f"  ПРЕДУПРЕЖДЕНИЕ: новый сценарий '{name}' отсутствует в baseline")

    if all_errors:
        print("\n=== ОБНАРУЖЕНЫ РАСХОЖДЕНИЯ ===")
        for name, errs in all_errors.items():
            print(f"\n[{name}]")
            for err in errs[:10]:
                print(f"  {err}")
            if len(errs) > 10:
                print(f"  ... и ещё {len(errs) - 10} ошибок")
        return 1

    print(f"\n✓ Все {len(golden)} сценариев совпадают с baseline.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Golden-тест трассировщика лучей")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--generate", action="store_true",
                       help="Сгенерировать baseline (запускать ДО рефакторинга)")
    group.add_argument("--verify", action="store_true",
                       help="Проверить текущее поведение против baseline")
    args = parser.parse_args()

    return do_generate() if args.generate else do_verify()


if __name__ == "__main__":
    sys.exit(main())
