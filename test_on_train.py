
from ultralytics import YOLO
import os

model = YOLO("C:/Users/Daniil/runs/detect/runs/bolts_v3/weights/best.pt")

# Берем 4 фото из папки, на которых ТОЧНО училась модель
train_dir = "C:/Users/Daniil/Desktop/mag1/opencv/bolts_dataset/dataset/images/train"
images = os.listdir(train_dir)[:4] # берем первые 4 фото

for img_name in images:
    img_path = os.path.join(train_dir, img_name)
    results = model.predict(source=img_path, conf=0.25, save=True, project="runs", name="test_train")
    print(f"Обработано: {img_name}")

print("Результаты в папке runs/test_train")