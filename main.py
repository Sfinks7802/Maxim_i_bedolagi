import cv2
import os
from ultralytics import YOLO
import torch
from poses_detection.hands_up import check_hand_raised
from poses_detection.squatting import check_squat
import datetime as dt


def process_video_with_pose_estimation(input_path, output_path, model_name):
    """
    Обрабатывает видеофайл для распознавания поз людей, рисует скелеты и сохраняет результат.

    Args:
        input_path (str): Путь к входному видеофайлу (.mp4).
        output_path (str): Путь для сохранения обработанного видеофайла (.mp4).
        model_name (str): Название модели YOLOv8 для оценки поз.
    """
    # Проверка, доступен ли CUDA, и выбор устройства
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используется устройство: {device}")

    # --- 1. Загрузка модели YOLO ---
    # Модель будет автоматически загружена из интернета при первом запуске
    print(f"Загрузка модели {model_name}...")
    try:
        model = YOLO(model_name)
        model.to(device)  # Перемещаем модель на выбранное устройство (GPU/CPU)
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return

    print("Модель успешно загружена.")

    # --- 2. Открытие видеофайла для чтения ---
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видеофайл по пути: {input_path}")
        return

    # Получение свойств видео (размеры кадра, FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(
        f"Видео '{os.path.basename(input_path)}': {frame_width}x{frame_height}, {fps:.2f} FPS, {total_frames} кадров."
    )

    # --- 3. Создание объекта для записи видео ---
    # Используем кодек 'mp4v' для формата .mp4
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    print(f"Результат будет сохранен в: {output_path}")

    # --- 4. Определение соединений для рисования скелета ---
    # Это стандартные соединения для 17 ключевых точек COCO
    # Индексы точек: 0:нос, 1:л_глаз, 2:п_глаз, 3:л_ухо, 4:п_ухо, 5:л_плечо, 6:п_плечо,
    # 7:л_локоть, 8:п_локоть, 9:л_запястье, 10:п_запястье, 11:л_бедро, 12:п_бедро,
    # 13:л_колено, 14:п_колено, 15:л_лодыжка, 16:п_лодыжка
    SKELETON_CONNECTIONS = [
        # Голова
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        # Тело
        (5, 6),
        (5, 11),
        (6, 12),
        (11, 12),
        # Левая рука
        (5, 7),
        (7, 9),
        # Правая рука
        (6, 8),
        (8, 10),
        # Левая нога
        (11, 13),
        (13, 15),
        # Правая нога
        (12, 14),
        (14, 16),
    ]

    # Цвета для точек и линий
    KEYPOINT_COLOR = (0, 0, 255)  # Красный для точек
    LINE_COLOR = (0, 255, 0)  # Зеленый для линий
    BOX_COLOR = (255, 0, 0)  # Синий для боксов

    # --- 5. Обработка каждого кадра видео ---
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Если кадров больше нет, выходим из цикла

        frame_count += 1
        time_start = dt.datetime.now()
        # Выводим прогресс в консоль
        print(f"\rОбработка кадра {frame_count}/{total_frames}...", end="")

        # Запускаем распознавание на кадре
        # verbose=False отключает вывод логов от YOLO для каждого кадра
        results = model(frame, verbose=False)

        # Копируем кадр, чтобы рисовать на нем
        annotated_frame = frame.copy()

        # Извлекаем результаты
        # results - это список, для одного изображения/кадра он содержит один элемент
        if results[0].boxes and results[0].keypoints:

            # --- Получаем и боксы, и ключевые точки ---

            keypoints_data = results[0].keypoints.cpu().numpy()
            boxes_data = results[0].boxes.cpu().numpy()

            # --- Итерируемся по индексу, чтобы связать бокс и скелет ---
            for i in range(len(boxes_data)):
                box = boxes_data[i]
                person_keypoints = keypoints_data[i]

                # Уверенность в распознавании человека
                confidence = box.conf[0]

                # Рисуем только если уверенность выше порога (например, 0.5)
                if confidence > 0.5:
                    points = person_keypoints.xy[0]
                    confs = person_keypoints.conf[0]
                    # --- Извлечение и рисование бокса ---
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

                    # --- Рисование подписи 'human' ---
                    hand_status = check_hand_raised(points, confs)
                    squat_status = check_squat(points, confs)
                    label = f"human {confidence:.2f}{hand_status}{squat_status}"
                    label_size, base_line = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                    )
                    y1_label = max(y1, label_size[1] + 10)
                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1_label - label_size[1] - 10),
                        (x1 + label_size[0], y1_label - base_line),
                        BOX_COLOR,
                        cv2.FILLED,
                    )
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, y1_label - 7),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                    # --- Код для рисования скелета ---

                    # Рисуем линии скелета
                    for p1_idx, p2_idx in SKELETON_CONNECTIONS:
                        # Рисуем линию, только если обе точки были распознаны с достаточной уверенностью
                        if confs[p1_idx] > 0.5 and confs[p2_idx] > 0.5:
                            point1 = tuple(map(int, points[p1_idx]))
                            point2 = tuple(map(int, points[p2_idx]))
                            cv2.line(annotated_frame, point1, point2, LINE_COLOR, 2)

                    # Рисуем ключевые точки
                    for i, point in enumerate(points):
                        # Рисуем точку, только если она распознана с достаточной уверенностью
                        if confs[i] > 0.5:
                            x, y = int(point[0]), int(point[1])
                            cv2.circle(annotated_frame, (x, y), 4, KEYPOINT_COLOR, -1)
        cv2.putText(
            annotated_frame,
            f"{1/float('0.0'+str((dt.datetime.now() - time_start).microseconds))} FPS",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

        # Записываем обработанный кадр в выходной файл
        out.write(annotated_frame)

    # --- 6. Освобождение ресурсов ---
    print("\nОбработка завершена.")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Видео успешно сохранено: {output_path}")


if __name__ == "__main__":
    input = "input_my.mp4"
    file_name, file_ext = os.path.splitext(os.path.basename(input))
    output = f"{file_name}_processed{file_ext}"

    process_video_with_pose_estimation(input, output, "yolo11n-pose.pt")
