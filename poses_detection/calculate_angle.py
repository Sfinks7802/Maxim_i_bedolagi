import numpy as np


# --- Универсальная функция для вычисления угла ---
def calculate_angle(a, b, c):
    """
    Вычисляет угол между тремя точками (в градусах). Угол измеряется в точке 'b'.

    Args:
        a, b, c (tuple or np.array): Координаты (x, y) трех точек.

    Returns:
        float: Угол в градусах от 0 до 180.
    """
    a = np.array(a) # Первая точка
    b = np.array(b) # Средняя точка (вершина угла)
    c = np.array(c) # Третья точка

    # Вычисляем векторы от средней точки
    ba = a - b
    bc = c - b

    # Используем формулу косинуса угла между векторами
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

    # Защита от выхода за пределы [-1, 1] из-за ошибок округления
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.arccos(cosine_angle)

    return np.degrees(angle)