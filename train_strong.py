
from ultralytics import YOLO

def main():
    # Берем yolov8n
    model = YOLO("yolov8n.pt")
    
    yaml_path = "C:/Users/Daniil/Desktop/mag1/opencv/bolts_dataset/dataset/bolt_nut.yaml"
    
    results = model.train(
        data=yaml_path,
        epochs=200,
        imgsz=640,
        batch=8,
        device=0,
        
        # --- СИЛЬНАЯ АУГМЕНТАЦИЯ ---
        # Заставляем модель забыть фон и смотреть на форму
        hsv_h=0.015,    # Изменение оттенка
        hsv_s=0.7,      # Изменение насыщенности (цвета)
        hsv_v=0.4,      # Изменение яркости (очень важно для теней)
        degrees=15.0,   # Повороты на 15 градусов
        translate=0.2,  # Сдвиг объекта в кадре
        scale=0.5,      # Изменение размера
        shear=2.0,      # Искажения
        perspective=0.0005, # Искажение перспективы
        mosaic=1.0,     # Склейка 4 фото в одно
        mixup=0.2,      # Наложение одного фото на другое
        erasing=0.4,    # Случайное стирание кусков фото (чтобы не смотрела на фон)
        
        project="runs",
        name="bolts_v8_strong"
    )

if __name__ == '__main__':
    main()