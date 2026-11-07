"""Дисперсия показателя преломления по формуле Коши."""

from constants import CAUCHY_A_OFFSET, CAUCHY_B
from core.rays import DispersiveRay, Ray


def get_dispersion_n(base_n: float, ray: Ray) -> float:
    """Показатель преломления с учётом дисперсии (только для DispersiveRay)."""
    if not isinstance(ray, DispersiveRay) or ray.wavelength is None:
        return base_n

    lambda_mkm = ray.wavelength / 1000.0
    A = base_n - CAUCHY_A_OFFSET
    B = CAUCHY_B
    return A + B / (lambda_mkm ** 2)
