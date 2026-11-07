"""Геометрические утилиты: нормализация, тангенты, матрица поворота, радиус линзы."""

import numpy as np

from constants import MIN_DOT_FOR_TANGENT, VECTOR_ZERO_EPSILON


def normalize(vec):
    """Нормализует вектор. Возвращает исходный, если он нулевой."""
    n = np.linalg.norm(vec)
    if n > VECTOR_ZERO_EPSILON:
        return vec / n
    return vec


def get_tangents(normal):
    """Возвращает два ортонормированных вектора, перпендикулярных normal."""
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    if abs(normal[0]) < MIN_DOT_FOR_TANGENT:
        arbitrary = np.array([1.0, 0.0, 0.0])
    else:
        arbitrary = np.array([0.0, 1.0, 0.0])
    t1 = np.cross(normal, arbitrary)
    t1 = t1 / np.linalg.norm(t1)
    t2 = np.cross(normal, t1)
    t2 = t2 / np.linalg.norm(t2)
    return t1, t2


def calculate_rotation_matrix(v_to):
    """Матрица поворота, переводящая ось X в v_to."""
    v_to = np.array(v_to, dtype=float)
    norm = np.linalg.norm(v_to)
    if norm < 1e-10:
        return np.eye(3)
    v_to /= norm

    v_from = np.array([1.0, 0.0, 0.0])

    if np.allclose(v_from, v_to):
        return np.eye(3)

    if np.allclose(v_from, -v_to):
        return np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

    v = np.cross(v_from, v_to)
    c = np.dot(v_from, v_to)

    K = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    return np.eye(3) + K + (K @ K) * (1 / (1 + c))


def calculate_radius(target_f, n, thickness=0):
    """Радиус R для двояковыпуклой линзы (тонкая или с толщиной).

    Оставлено для будущего использования (UI/аналитика).
    """
    if thickness == 0:
        return 2 * target_f * (n - 1)

    A = ((n - 1) * thickness) / n
    B = 2
    C = -1 / (target_f * (n - 1))

    D = B ** 2 - 4 * A * C
    if D < 0:
        return None

    x = (-B + np.sqrt(D)) / (2 * A)
    return 1 / x
