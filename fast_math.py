# fast_math.py
import numpy as np
from numba import njit


@njit
def dot(v1, v2):
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]


@njit
def fresnel_amplitudes(n1, n2, cos_i):
    sin2_i = max(0.0, 1.0 - cos_i*cos_i)
    eta = n1 / n2
    sin2_t = eta*eta * sin2_i
    if sin2_t > 1.0:
        return 1.0+0j, 1.0+0j, 0.0+0j, 0.0+0j
    cos_t = np.sqrt(1.0 - sin2_t)
    r_s = (n1*cos_i - n2*cos_t) / (n1*cos_i + n2*cos_t)
    r_p = (n2*cos_i - n1*cos_t) / (n2*cos_i + n1*cos_t)
    t_s = 2.0 * n1 * cos_i / (n1*cos_i + n2*cos_t)
    t_p = 2.0 * n1 * cos_i / (n2*cos_i + n1*cos_t)
    return complex(r_s), complex(r_p), complex(t_s), complex(t_p)

# ---------- Преломление (Снеллиус) ----------
@njit
def refract(ray_dir, normal, n1, n2):
    # Относительный показатель преломления
    eta = n1 / n2

    # Находим скалярное произведение падающего луча и базовой нормали
    cos_i = dot(normal, ray_dir)
    actual_normal = normal.copy()

    # Корректировка нормали:
    if cos_i > 0.0:
        # Если скалярное произведение положительно, луч выходит из объекта
        # В этом случае разворачиваем нормаль навстречу лучу
        actual_normal = -actual_normal
        cos_i = dot(actual_normal, ray_dir)

    cos_i = -cos_i

    # sin^2(t) = eta^2 * sin^2(i) = eta^2 * (1 - cos^2(i))
    sin2_t = eta * eta * (1.0 - cos_i * cos_i)

    # Проверка на Полное Внутреннее Отражение:
    if sin2_t > 1.0:
        # Возвращаем вектор идеального зеркального отражения от поверхности
        return ray_dir - 2.0 * dot(ray_dir, actual_normal) * actual_normal

    # Находим косинус угла преломления по формуле: cos(t) = sqrt(1 - sin^2(t))
    cos_t = np.sqrt(max(0.0, 1.0 - sin2_t))

    # Векторное уравнение преломления света
    return eta * ray_dir + (eta * cos_i - cos_t) * actual_normal

# ---------- Пересечение с плоскостью (круглая апертура) ----------
@njit
def plane_intersect_circular(ray_origin, ray_dir, point, normal, origin, axis, edge_radius):
    dot_dn = dot(ray_dir, normal)
    if abs(dot_dn) < 1e-6:
        return -1.0
    t = dot(point - ray_origin, normal) / dot_dn
    if t <= 1e-6:
        return -1.0
    p = ray_origin + t * ray_dir
    vec = p - origin
    proj = dot(vec, axis)
    dist = np.sqrt(dot(vec, vec) - proj * proj)
    if dist <= edge_radius + 1e-6:
        return t
    return -1.0


# ---------- Пересечение с плоскостью (прямоугольная апертура) ----------
@njit
def plane_intersect_rectangular(ray_origin, ray_dir, point, normal, origin, tangents, half_sizes):
    dot_dn = dot(ray_dir, normal)
    if abs(dot_dn) < 1e-6:
        return -1.0
    t = dot(point - ray_origin, normal) / dot_dn
    if t <= 1e-6:
        return -1.0
    p = ray_origin + t * ray_dir
    u = dot(p - origin, tangents[0])
    v = dot(p - origin, tangents[1])
    if abs(u) <= half_sizes[0] + 1e-6 and abs(v) <= half_sizes[1] + 1e-6:
        return t
    return -1.0


@njit
def cylinder_intersect(ray_origin, ray_dir, center, axis, radius, half_length):
    """
    Пересечение луча с боковой поверхностью цилиндра.
    Цилиндр задан центром основания на оси, направлением оси, радиусом и полудлиной.
    Возвращает наименьшее положительное t или -1.
    """
    oc = ray_origin - center
    # Проекция направления на плоскость, перпендикулярную оси
    a_dir = ray_dir - np.dot(ray_dir, axis) * axis
    a_oc = oc - np.dot(oc, axis) * axis

    A = np.dot(a_dir, a_dir)
    if A < 1e-12:
        # Луч параллелен оси цилиндра — пересечения с боковой поверхностью нет
        return -1.0

    B = 2.0 * np.dot(a_dir, a_oc)
    C = np.dot(a_oc, a_oc) - radius * radius
    disc = B * B - 4.0 * A * C
    if disc < 0.0:
        return -1.0

    sqrt_disc = np.sqrt(disc)
    t1 = (-B - sqrt_disc) / (2.0 * A)
    t2 = (-B + sqrt_disc) / (2.0 * A)

    best_t = -1.0
    for t in (t1, t2):
        if t <= 1e-6:
            continue
        p = ray_origin + t * ray_dir
        # Проверка, что точка лежит в пределах полудлины вдоль оси
        proj = np.dot(p - center, axis)
        if abs(proj) <= half_length + 1e-6:
            if best_t < 0.0 or t < best_t:
                best_t = t
    return best_t


@njit
def _sag_numba(r, R, k, coeffs):
    # 1. Вычисление базовой составляющей профиля (конического сечения)
    if abs(R) < 1e-12:
        # Если радиус кривизны бесконечен или близок к нулю — поверхность плоская
        sag = 0.0
    else:
        # Переходим к кривизне вершины поверхности c = 1/R
        c = 1.0 / R

        # Вычисляем подкоренное выражение для уравнения коноида: 1 - (1 + k) * c^2 * r^2
        discr = 1.0 - (1.0 + k) * c ** 2 * r ** 2

        if discr < 0.0:
            # Если дискриминант < 0, то профиль геометрически невозможен
            return np.inf

        sqrt_discr = np.sqrt(discr)
        sag = (c * r ** 2) / (1.0 + sqrt_discr)

    # 2. Добавление асферических деформаций
    # coeffs содержит коэффициенты при четных степенях радиуса: A4, A6, A8, A10...
    for i, A in enumerate(coeffs):
        sag += A * r ** (2 * (i + 1))

    # Возвращаем суммарную стрелку прогиба для текущего радиуса
    return sag


@njit
def aspheric_intersect_numba(origin, direction, radius, k, coeffs, edge_radius, sag_max):
    # Извлекаем локальные Y и Z координаты начала и направления луча
    oy, oz = origin[1], origin[2]
    dy, dz = direction[1], direction[2]

    # Квадрат проекции вектора направления на плоскость YZ (перпендикулярно оптической оси X)
    a_cyl = dy * dy + dz * dz

    if a_cyl < 1e-12:
        # Если луч строго параллелен локальной оптической оси X

        # Вычисляем радиальное расстояние от оси X до траектории луча
        r0 = np.sqrt(oy * oy + oz * oz)

        if r0 > edge_radius + 1e-6:
            # Если луч идет мимо поверхности
            return -1.0

        if abs(direction[0]) < 1e-12:
            return -1.0

        # Математически точное (явное) решение: t = (sag(r0) - x0) / dx
        sag0 = _sag_numba(r0, radius, k, coeffs)
        t = (sag0 - origin[0]) / direction[0]

        if t <= 1e-6:
            return -1.0  # Пересечение позади источника
        return t

    # === 2. ОБЩИЙ СЛУЧАЙ: Ограничение поиска через пересечение с бесконечным цилиндром апертуры ===
    # Решаем квадратное уравнение пересечения луча с цилиндром радиуса edge_radius вдоль оси X:
    # a_cyl * t^2 + b_cyl * t + c_cyl = 0
    b_cyl = 2.0 * (oy * dy + oz * dz)
    c_cyl = oy * oy + oz * oz - edge_radius * edge_radius
    disc_cyl = b_cyl * b_cyl - 4.0 * a_cyl * c_cyl

    # Если дискриминант отрицательный, луч гарантированно пролетает мимо цилиндрического объема линзы
    if disc_cyl < 0.0:
        return -1.0

    # Находим интервал [t1, t2], внутри которого луч находится внутри радиального цилиндра линзы
    sqrt_disc_cyl = np.sqrt(disc_cyl)
    t1 = (-b_cyl - sqrt_disc_cyl) / (2.0 * a_cyl)
    t2 = (-b_cyl + sqrt_disc_cyl) / (2.0 * a_cyl)
    t1, t2 = min(t1, t2), max(t1, t2)

    # Если точка выхода из цилиндра позади луча, пересечений нет
    if t2 <= 1e-6:
        return -1.0

    # Сужаем интервал поиска: входить в линзу мы можем не раньше, чем t = 0 (точка старта луча)
    t_enter = max(0.0, t1)
    t_exit = t2
    if t_enter > t_exit:
        return -1.0

    # Вычисляем 3D-координаты и радиальные расстояния r для точек входа и выхода из цилиндра
    x_enter = origin[0] + t_enter * direction[0]
    x_exit = origin[0] + t_exit * direction[0]
    r_enter = np.sqrt((origin[1] + t_enter * dy) ** 2 + (origin[2] + t_enter * dz) ** 2)
    r_exit = np.sqrt((origin[1] + t_exit * dy) ** 2 + (origin[2] + t_exit * dz) ** 2)

    # Вычисляем невязку F(t) = X_луча(t) - Стрелка_прогиба(r_луча(t)) на границах.
    # Корень F(t) == 0 будет являться точным физическим пересечением луча с асферикой.
    F_enter = x_enter - _sag_numba(r_enter, radius, k, coeffs)
    F_exit = x_exit - _sag_numba(r_exit, radius, k, coeffs)

    t_root = -1.0

    # === 3. ПОИСК КОРНЯ МЕТОДОМ БИСЕКЦИИ ===
    # Проверяем, не попали ли мы в корень прямо на входе
    if abs(F_enter) < 1e-9:
        t_root = t_enter
    # Проверяем корень на выходе
    elif abs(F_exit) < 1e-9:
        t_root = t_exit
    # Если знаки функции F(t) на краях разные, значит внутри интервала гарантированно есть корень
    elif F_enter * F_exit < 0.0:
        lo, hi = t_enter, t_exit
        flo, fhi = F_enter, F_exit

        # Запускаем фиксированные 60 итераций деления отрезка пополам (гарантия отсутствия вечных циклов в Numba)
        for _ in range(60):
            mid = (lo + hi) * 0.5
            x_mid = origin[0] + mid * direction[0]
            r_mid = np.sqrt((origin[1] + mid * dy) ** 2 + (origin[2] + mid * dz) ** 2)
            fmid = x_mid - _sag_numba(r_mid, radius, k, coeffs)

            # Если точность достигнута или интервал схлопнулся до машинного нуля — корень найден
            if abs(fmid) < 1e-9 or (hi - lo) < 1e-12:
                t_root = mid
                break

            # Сжимаем границы поиска на основе знака функции на середине интервала
            if flo * fmid < 0.0:
                hi = mid
                fhi = fmid
            else:
                lo = mid
                flo = fmid

        # Если цикл завершился по итерациям, берем среднее значение как лучшее приближение
        if t_root < 0.0:
            t_root = (lo + hi) * 0.5
    else:
        # === 4. ЗАЩИТА ОТ ОШИБОК ОКРУГЛЕНИЯ НА КРАЮ АПЕРТУРЫ ===
        # Если знаки одинаковые, корень мог «выскользнуть» из интервала из-за погрешностей float64 на самом краю.
        # Искусственно расширяем границы поиска наружу на дельту 0.01 и пытаемся найти корень повторно.
        t_lo_try = max(0.0, t_enter - 0.01)
        t_hi_try = t_exit + 0.01

        x_lo = origin[0] + t_lo_try * direction[0]
        r_lo = np.sqrt((origin[1] + t_lo_try * dy) ** 2 + (origin[2] + t_lo_try * dz) ** 2)
        F_lo = x_lo - _sag_numba(r_lo, radius, k, coeffs)

        x_hi = origin[0] + t_hi_try * direction[0]
        r_hi = np.sqrt((origin[1] + t_hi_try * dy) ** 2 + (origin[2] + t_hi_try * dz) ** 2)
        F_hi = x_hi - _sag_numba(r_hi, radius, k, coeffs)

        # Если в расширенном диапазоне знаки стали разными — запускаем аналогичную бисекцию
        if F_lo * F_hi < 0.0:
            lo, hi = t_lo_try, t_hi_try
            flo, fhi = F_lo, F_hi
            for _ in range(60):
                mid = (lo + hi) * 0.5
                x_mid = origin[0] + mid * direction[0]
                r_mid = np.sqrt((origin[1] + mid * dy) ** 2 + (origin[2] + mid * dz) ** 2)
                fmid = x_mid - _sag_numba(r_mid, radius, k, coeffs)
                if abs(fmid) < 1e-9 or (hi - lo) < 1e-12:
                    t_root = mid
                    break
                if flo * fmid < 0.0:
                    hi = mid
                    fhi = fmid
                else:
                    lo = mid
                    flo = fmid
            if t_root < 0.0:
                t_root = (lo + hi) * 0.5
        else:
            return -1.0  # Корня действительно нет (луч прошел мимо чаши асферики)

    # Если корень не был определен ни в одном блоке
    if t_root < 0.0:
        return -1.0

    # === 5. ФИНАЛЬНЫЕ ГЕОМЕТРИЧЕСКИЕ ПРОВЕРКИ И ФИЛЬТРАЦИЯ ===
    # Находим финальные мировые (в рамках локального базиса) координаты точки пересечения
    p_final = origin + t_root * direction

    # Проверяем, что точка на поверхности физически не вылезла за радиус апертуры
    r_final = np.sqrt(p_final[1] ** 2 + p_final[2] ** 2)
    if r_final > edge_radius + 1e-6:
        return -1.0

    # Защита от пересечения с "виртуальной" обратной стороной математической функции.
    # Ограничиваем координату X точки пересечения строгими рамками толщины «чаши» [sag_min, sag_max]
    sag_min = min(0.0, sag_max)
    sag_max_val = max(0.0, sag_max)
    if p_final[0] < sag_min - 1e-6 or p_final[0] > sag_max_val + 1e-6:
        return -1.0

    # Возвращаем успешно найденный параметр t
    return t_root
