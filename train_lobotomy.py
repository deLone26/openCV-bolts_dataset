
from ultralytics import YOLO

def main():
    # Берем yolov8n
    model = YOLO("yolov8n.pt")
    
    yaml_path = "C:/Users/Daniil/Desktop/mag1/opencv/bolts_dataset/dataset/bolt_nut.yaml"
    
    results = model.train(
        data=yaml_path,
        epochs=150,
        imgsz=640,
        batch=8,
        device=0,
        
        # --- САМАЯ ЖЕСТКАЯ АУГМЕНТАЦИЯ ---
        # 1. Вырезаем объекты и вставляем их в случайные места (помогает от переобучения на позицию)
        copy_paste=0.5,      
        # 2. Склеиваем 4 фото в одно (объекты в случайных местах)
        mosaic=1.0,          
        # 3. Сильно сдвигаем объекты по кадру
        translate=0.5,       
        # 4. Сильно меняем масштаб (от 20% до 200%)
        scale=0.8,           
        # 5. Крутим на 180 градусов
        degrees=180.0,       
        # 6. Отражения
        fliplr=0.5,
        flipud=0.5,
        # 7. Сильно меняем яркость и контраст (чтобы не запоминала тени)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.5,           
        
        project="runs",
        name="bolts_final_boss",
        # Отключаем раннюю остановку, чтобы модель училась все 300 эпох
        patience=300
    )

if __name__ == '__main__':
    main()