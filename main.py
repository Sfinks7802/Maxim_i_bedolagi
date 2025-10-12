import cv2
import os
from ultralytics import YOLO
import torch


def process_video_with_pose_estimation(input_path, output_path, model_name='yolov8n-pose.pt'):
    """
    Обрабатывает видеофайл для распознавания поз людей, рисует скелеты и сохраняет результат.

    Args:
        input_path (str): Путь к входному видеофайлу (.mp4).
        output_path (str): Путь для сохранения обработанного видеофайла (.mp4).
        model_name (str): Название модели YOLOv8 для оценки поз.
    """
    # Проверка, доступен ли CUDA, и выбор устройства
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Используется устройство: {device}")

    # --- 1. Загрузка модели YOLOv8-pose ---
    # Модель будет автоматически загружена из интернета при первом запуске
    print(f"Загрузка модели {model_name}...")
    try:
        model = YOLO(model_name)
        model.to(device) # Перемещаем модель на выбранное устройство (GPU/CPU)
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

    print(f"Видео '{os.path.basename(input_path)}': {frame_width}x{frame_height}, {fps:.2f} FPS, {total_frames} кадров.")

    # --- 3. Создание объекта для записи видео ---
    # Используем кодек 'mp4v' для формата .mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    print(f"Результат будет сохранен в: {output_path}")

    # --- 4. Определение соединений для рисования скелета ---
    # Это стандартные соединения для 17 ключевых точек COCO
    # Индексы точек: 0:нос, 1:л_глаз, 2:п_глаз, 3:л_ухо, 4:п_ухо, 5:л_плечо, 6:п_плечо,
    # 7:л_локоть, 8:п_локоть, 9:л_запястье, 10:п_запястье, 11:л_бедро, 12:п_бедро,
    # 13:л_колено, 14:п_колено, 15:л_лодыжка, 16:п_лодыжка
    SKELETON_CONNECTIONS = [
        # Голова
        (0, 1), (0, 2), (1, 3), (2, 4),
        # Тело
        (5, 6), (5, 11), (6, 12), (11, 12),
        # Левая рука
        (5, 7), (7, 9),
        # Правая рука
        (6, 8), (8, 10),
        # Левая нога
        (11, 13), (13, 15),
        # Правая нога
        (12, 14), (14, 16)
    ]

    # Цвета для точек и линий
    KEYPOINT_COLOR = (0, 0, 255) # Красный для точек
    LINE_COLOR = (0, 255, 0) # Зеленый для линий

    # --- 5. Обработка каждого кадра видео ---
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # Если кадров больше нет, выходим из цикла

        frame_count += 1
        # Выводим прогресс в консоль
        print(f"\rОбработка кадра {frame_count}/{total_frames}...", end="")

        # Запускаем распознавание на кадре
        # verbose=False отключает вывод логов от YOLO для каждого кадра
        results = model(frame, verbose=False)

        # Копируем кадр, чтобы рисовать на нем
        annotated_frame = frame.copy()

        # Извлекаем результаты
        # results - это список, для одного изображения/кадра он содержит один элемент
        if results[0].keypoints:
            keypoints_data = results[0].keypoints.cpu().numpy()

            # Итерируемся по каждому распознанному человеку
            for person_keypoints in keypoints_data:
                # person_keypoints.xy содержит координаты [x, y] для всех 17 точек
                # person_keypoints.conf содержит уверенность в распознавании каждой точки

                points = person_keypoints.xy[0] # Получаем массив точек (17, 2)
                confs = person_keypoints.conf[0] # Получаем массив уверенностей (17,)

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

        # Записываем обработанный кадр в выходной файл
        out.write(annotated_frame)

    # --- 6. Освобождение ресурсов ---
    print("\nОбработка завершена.")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Видео успешно сохранено: {output_path}")

if __name__ == '__main__':
    file_name, file_ext = os.path.splitext(os.path.basename("input.mp4"))
    output = f"{file_name}_processed{file_ext}"

    process_video_with_pose_estimation("input.mp4", output, "yolo11n-pose.pt")
