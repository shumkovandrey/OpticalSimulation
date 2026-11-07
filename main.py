import pyvista as pv

pv.global_theme.allow_empty_mesh = True

# ---- Реэкспорт для обратной совместимости ----
from constants import RAY_INFINITY_DISTANCE                           # noqa: F401
from utils.geometry import (                                          # noqa: F401
    normalize, get_tangents, calculate_rotation_matrix, calculate_radius,
)
from utils.color import (                                             # noqa: F401
    wavelength_to_rgb, energy_to_alpha, to_rgb,
)                                                                     # noqa: F401
from core.rays import Ray, DispersiveRay, WhiteRay, RayPool           # noqa: F401
from core.segments import Segment, HitInfo                            # noqa: F401
from core.dispersion import get_dispersion_n                          # noqa: F401
from surfaces.base import BaseSurface, LensAssembly                   # noqa: F401
from surfaces.plane import PlaneSurface                               # noqa: F401
from surfaces.sphere import SphereSurface                             # noqa: F401
from surfaces.cylinder import CylinderSurface                         # noqa: F401
from surfaces.aspheric import AsphericSurface                         # noqa: F401
from surfaces.mesh import MeshSurface                                 # noqa: F401
from surfaces.thin_lens import ThinLens                               # noqa: F401
from surfaces.universal_lens import UniversalLens                     # noqa: F401
from surfaces.hyperbolic_lens import HyperbolicLens                   # noqa: F401
from tracing.modes import TraceMode, SimpleMode, TreeMode             # noqa: F401
from tracing.hit_finder import find_best_hit                          # noqa: F401
from tracing.interaction import split_ray                             # noqa: F401
from tracing.simple import _trace_simple                              # noqa: F401
from tracing.recursive import _trace_recursive                        # noqa: F401
from emitters import BeamEmitter                                      # noqa: F401
from visualization import RayCloud, RayTracer                         # noqa: F401



if __name__ == "__main__":
    plotter = pv.Plotter()
    plotter.set_background("black")
    plotter.view_isometric()
    plotter.enable_parallel_projection()
    plotter.enable_terrain_style(mouse_wheel_zooms=True)
    plotter.view_vector((0, 0, 1), viewup=(0, 1, 0))
    plotter.add_axes(color="white")
    plotter.show()
