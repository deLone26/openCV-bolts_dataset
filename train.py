from ultralytics import YOLO

def main():
    # Загружаем YOLOv8 nano
    model = YOLO("yolov8n.yaml")
    
    yaml_path = "C:/Users/Daniil/Desktop/mag1/opencv/bolts_dataset/dataset/bolt_nut.yaml"
    
    results = model.train(
        data=yaml_path,
        epochs=250,            # Дадим побольше эпох
        imgsz=640,
        batch=8,
        device=0,              # Ваша RTX 5060
        project="runs",
        name="bolts_v8",       # Новое имя, чтобы сохранить историю
        
        # --- Аугментации для борьбы с переобучением на фон ---
        mosaic=1.0,
        mixup=0.3,             # Смешивает два фото, заставляя модель игнорировать фон
        degrees=15.0,          # Повороты фото
        scale=0.9,             # Масштабирование
        fliplr=0.5             # Отражение по горизонтали
    )

if __name__ == '__main__':
    main()