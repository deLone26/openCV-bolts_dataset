
import cv2
import os
import random

# Путь к папке, где лежат картинки и txt (после split_dataset)
IMG_DIR = "dataset/images/train"
LBL_DIR = "dataset/labels/train"

# Куда сохранить картинки с проверкой
OUT_DIR = "check_labels_visual"
os.makedirs(OUT_DIR, exist_ok=True)

# Берем 10 случайных фото
images = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
random.seed(42)
random.shuffle(images)
images = images[:10]

for img_name in images:
    img_path = os.path.join(IMG_DIR, img_name)
    lbl_name = os.path.splitext(img_name)[0] + ".txt"
    lbl_path = os.path.join(LBL_DIR, lbl_name)
    
    if not os.path.exists(lbl_path):
        print(f"Нет txt для {img_name}")
        continue
        
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    with open(lbl_path, 'r') as f:
        lines = f.readlines()
        
    if not lines:
        print(f"⚠️ Пустой txt для {img_name}")
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            class_id, cx, cy, bw, bh = map(float, parts)
            class_id = int(class_id)
            
            # Переводим из YOLO формата (центр, ширина, высота) в пиксели (x1, y1, x2, y2)
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            
            # Цвет: 0 (bolt) - зеленый, 1 (nut) - красный
            color = (0, 255, 0) if class_id == 0 else (0, 0, 255)
            label = "bolt" if class_id == 0 else "nut"
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
    # Уменьшаем для показа, если фото большое
    scale = min(800 / w, 800 / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        
    out_path = os.path.join(OUT_DIR, f"check_{img_name}")
    cv2.imwrite(out_path, img)
    print(f"Сохранено: {out_path}")

print(f"\nГотово! Откройте папку {OUT_DIR} и проверьте картинки.")