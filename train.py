from ultralytics import YOLO

def main():
    # Загружаем самую легкую и быструю модель
    model = YOLO("yolo11n.pt")
    
    # Запускаем обучение
    results = model.train(
        data="dataset/bolt_nut.yaml", # Путь к файлу конфигурации
        epochs=50,                    # 50 проходов по датасету
        imgsz=640,                    # Размер изображения
        batch=8,                      # Размер батча (для RTX 5060 можно и 16)
        device=0,                     # Использовать GPU (RTX 5060)
        project="runs",               # Куда сохранять результаты
        name="bolts_v1"               # Название папки с результатами
    )

if __name__ == '__main__':
    main()