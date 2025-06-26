"""
inference.py — Обработка видео и детекция людей с использованием YOLOv8.
"""

import cv2
from ultralytics import YOLO
import os

def run_inference(model_path: str, video_path: str, output_path: str) -> None:
    """
    Применяет обученную модель YOLO для инференса на видео.

    Args:
        model_path (str): Путь к модели (.pt)
        video_path (str): Входное видео
        output_path (str): Файл для финального видео с отрисовкой
    """
    # Проверяем наличие файлов
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Source video file not found: {video_path}")

    # Загружаем модель
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Проверяем, что папка output существует
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.5, iou=0.5, imgsz=960, verbose=False)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if model.names[cls_id] == 'person':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                color = (0, 255, 255)
                thickness = 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                label = f"person {conf:.2f}"
                font_scale = 0.5
                font_thickness = 1
                cv2.putText(
                    frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, font_thickness
                )

        out.write(frame)

    cap.release()
    out.release()
    print(f"Результат сохранен в: {output_path}")