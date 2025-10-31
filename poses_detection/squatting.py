from poses_detection.calculate_angle import calculate_angle


# --- Функция для определения приседа ---
def check_squat(
    points,
    confs,
    confidence_threshold=0.5,
    knee_angle_threshold=120,
    hip_angle_threshold=120,
):
    """
    Проверяет, находится ли человек в приседе.

    Args:
        points (np.array): Координаты 17 ключевых точек.
        confs (np.array): Уверенность для каждой точки.
        confidence_threshold (float): Порог уверенности для учета точки.
        knee_angle_threshold (float): Пороговый угол для колена.
        hip_angle_threshold (float): Пороговый угол для бедра.

    Returns:
        str: " (Squatting)" если поза соответствует, иначе "".
    """
    # Индексы ключевых точек COCO
    L_SHOULDER, R_SHOULDER = 5, 6
    L_HIP, R_HIP = 11, 12
    L_KNEE, R_KNEE = 13, 14
    L_ANKLE, R_ANKLE = 15, 16

    left_leg_squat = False
    right_leg_squat = False

    # Проверка левой ноги
    if all(
        confs[i] > confidence_threshold for i in [L_SHOULDER, L_HIP, L_KNEE, L_ANKLE]
    ):
        left_knee_angle = calculate_angle(
            points[L_HIP], points[L_KNEE], points[L_ANKLE]
        )
        left_hip_angle = calculate_angle(
            points[L_SHOULDER], points[L_HIP], points[L_KNEE]
        )
        if (
            left_knee_angle < knee_angle_threshold
            and left_hip_angle < hip_angle_threshold
        ):
            left_leg_squat = True

    # Проверка правой ноги
    if all(
        confs[i] > confidence_threshold for i in [R_SHOULDER, R_HIP, R_KNEE, R_ANKLE]
    ):
        right_knee_angle = calculate_angle(
            points[R_HIP], points[R_KNEE], points[R_ANKLE]
        )
        right_hip_angle = calculate_angle(
            points[R_SHOULDER], points[R_HIP], points[R_KNEE]
        )
        if (
            right_knee_angle < knee_angle_threshold
            and right_hip_angle < hip_angle_threshold
        ):
            right_leg_squat = True

    # Считаем, что человек приседает, если хотя бы одна нога в нужной позе
    if left_leg_squat or right_leg_squat:
        return " (Squatting)"
    else:
        return ""
