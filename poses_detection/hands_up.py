# --- Функция для определения поднятых рук ---
def check_hand_raised(points, confs, confidence_threshold=0.5):
    """
    Проверяет, подняты ли руки у человека, на основе координат ключевых точек.

    Args:
        points (np.array): Массив координат (x, y) для 17 ключевых точек.
        confs (np.array): Массив уверенности для каждой из 17 точек.
        confidence_threshold (float): Порог уверенности для учета точки.

    Returns:
        str: Строка, описывающая состояние рук (" (Left Hand Up)", " (Right Hand Up)", " (Hands Up)" или "").
    """
    # Индексы ключевых точек COCO
    # 5: л_плечо, 6: п_плечо, 9: л_запястье, 10: п_запястье
    L_SHOULDER, R_SHOULDER = 5, 6
    L_WRIST, R_WRIST = 9, 10

    left_hand_up = False
    right_hand_up = False

    # Проверка левой руки
    # Условие: точки плеча и запястья должны быть распознаны с достаточной уверенностью
    if confs[L_WRIST] > confidence_threshold and confs[L_SHOULDER] > confidence_threshold:
        # Условие: Y-координата запястья должна быть меньше Y-координаты плеча
        if points[L_WRIST][1] < points[L_SHOULDER][1]:
            left_hand_up = True

    # Проверка правой руки
    if confs[R_WRIST] > confidence_threshold and confs[R_SHOULDER] > confidence_threshold:
        if points[R_WRIST][1] < points[R_SHOULDER][1]:
            right_hand_up = True

    # Формирование итоговой строки
    if left_hand_up and right_hand_up:
        return " (Hands Up)"
    elif left_hand_up:
        return " (Left Hand Up)"
    elif right_hand_up:
        return " (Right Hand Up)"
    else:
        return ""