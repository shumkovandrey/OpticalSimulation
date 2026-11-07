"""Цветовые утилиты: длина волны → RGB, энергия → альфа, приведение цвета."""

import numpy as np
from matplotlib.colors import to_rgb as _to_rgb_mpl


def to_rgb(color):
    """Приводит любой matplotlib-цвет (строка, tuple) к np.ndarray RGB float32."""
    return np.array(_to_rgb_mpl(color), dtype=np.float32)


def wavelength_to_rgb(wavelength_nm):
    """Перевод длины волны света (нм) в RGB для визуализации."""
    if 380 <= wavelength_nm < 440:
        R, G, B = -(wavelength_nm - 440) / (440 - 380), 0.0, 1.0
    elif 440 <= wavelength_nm < 490:
        R, G, B = 0.0, (wavelength_nm - 440) / (490 - 440), 1.0
    elif 490 <= wavelength_nm < 510:
        R, G, B = 0.0, 1.0, -(wavelength_nm - 510) / (510 - 490)
    elif 510 <= wavelength_nm < 580:
        R, G, B = (wavelength_nm - 510) / (580 - 510), 1.0, 0.0
    elif 580 <= wavelength_nm < 645:
        R, G, B = 1.0, -(wavelength_nm - 645) / (645 - 580), 0.0
    elif 645 <= wavelength_nm <= 780:
        R, G, B = 1.0, 0.0, 0.0
    else:
        R, G, B = 1.0, 1.0, 1.0

    factor = 1.0
    if 380 <= wavelength_nm < 420:
        factor = 0.3 + 0.7 * (wavelength_nm - 380) / (420 - 380)
    elif 700 < wavelength_nm <= 780:
        factor = 0.3 + 0.7 * (780 - wavelength_nm) / (780 - 700)

    return (R * factor, G * factor, B * factor)


def energy_to_alpha(energy, etype, min_alpha=0.0, gamma=1.0):
    """Преобразование энергии луча в альфа-канал.

    etype:
        0 — всегда непрозрачный
        1 — альфа линейно от энергии
        2 — альфа = energy ** gamma (с нижним порогом min_alpha)
        иначе — непрозрачный
    """
    if etype == 0:
        return 1.0
    elif etype == 1:
        return np.clip(energy, 0.0, 1.0)
    elif etype == 2:
        return max(min_alpha, energy ** gamma)
    else:
        return 1.0
