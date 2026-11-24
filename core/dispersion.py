"""Дисперсия показателя преломления по формуле Коши."""

from core.materials import Material
from core.rays import DispersiveRay, Ray


def get_dispersion_n(material: "Material", ray: Ray) -> float:
    """Показатель преломления с учётом дисперсии (только для DispersiveRay)."""
    if not isinstance(ray, DispersiveRay) or ray.wavelength is None:
        return material.n
    lambda_mkm = ray.wavelength / 1000.0
    A = material.n - material.a_offset
    B = material.empirical_coeff
    return A + B / (lambda_mkm ** 2)
