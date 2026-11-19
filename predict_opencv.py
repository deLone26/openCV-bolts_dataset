import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

# Скрываем главное окно tkinter
root = tk.Tk()
root.withdraw()

# Путь к финальной модели
MODEL_PATH = "C:/Users/Daniil/runs/detect/runs/bolts_final_boss/weights/best.onnx"

print("Выберите тестовое фото...")
IMAGE_PATH = filedialog.askopenfilename(
    title="Выберите фото для теста",
    filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp"), ("Все файлы", "*.*")]
)

if not IMAGE_PATH:
    print("Фото не выбрано. Выход.")
    exit()

print(f"Выбрано: {IMAGE_PATH}")

# Загружаем модель
net = cv2.dnn.readNetFromONNX(MODEL_PATH)
classes = ["bolt", "nut"]

img_original = cv2.imread(IMAGE_PATH)
h, w = img_original.shape[:2]

# --- ПРЕПРОЦЕССИНГ (Letterbox) ---
# Масштабируем так, чтобы сохранить пропорции, и вписываем в квадрат 640x640
scale = min(640 / w, 640 / h)
new_w, new_h = int(w * scale), int(h * scale)

# Создаем квадратное полотно 640x640, залитое серым (114, 114, 114)
padded_img = np.full((640, 640, 3), 114, dtype=np.uint8)
pad_x = (640 - new_w) // 2
pad_y = (640 - new_h) // 2

# Вставляем уменьшенное оригинальное фото в центр
padded_img[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = cv2.resize(img_original, (new_w, new_h))

# Подаем в сеть
blob = cv2.dnn.blobFromImage(padded_img, 1/255.0, (640, 640), swapRB=True, crop=False)
net.setInput(blob)
outputs = net.forward()[0].transpose()

boxes, confidences, class_ids = [], [], []

for row in outputs:
    scores = row[4:]
    class_id = int(np.argmax(scores))
    confidence = float(scores[class_id])
    
    # Порог уверенности 0.4 отсекает случайные срабатывания
    if confidence > 0.4: 
        cx, cy, bw, bh = row[0], row[1], row[2], row[3]
        
        # Возвращаем координаты к исходному размеру фото
        x = cx - bw / 2
        y = cy - bh / 2
        
        x = (x - pad_x) / scale
        y = (y - pad_y) / scale
        bw = bw / scale
        bh = bh / scale
        
        # Ограничиваем рамки границами фото
        x = max(0, min(x, w))
        y = max(0, min(y, h))
        bw = min(bw, w - x)
        bh = min(bh, h - y)
        
        boxes.append([int(x), int(y), int(bw), int(bh)])
        confidences.append(float(confidence))
        class_ids.append(class_id)

# NMS (убираем дубликаты)
indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.4, nms_threshold=0.4)

counts = {"bolt": 0, "nut": 0}
if len(indices) > 0:
    for i in indices.flatten():
        x, y, bw, bh = boxes[i]
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        color = (0, 255, 0) if class_ids[i] == 0 else (0, 0, 255)
        
        cv2.rectangle(img_original, (x, y), (x + bw, y + bh), color, 3)
        cv2.putText(img_original, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        counts[classes[class_ids[i]]] += 1

print(f"\n--- Результат ---")
print(f"Болтов: {counts['bolt']}")
print(f"Гаек:   {counts['nut']}")
print(f"Всего:  {counts['bolt'] + counts['nut']}")

# Выводим счетчик прямо на фото
total_text = f"Bolts: {counts['bolt']} | Nuts: {counts['nut']}"
cv2.putText(img_original, total_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5)
cv2.putText(img_original, total_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

# --- Уменьшение размера окна для показа ---
screen_h = root.winfo_screenheight()
screen_w = root.winfo_screenwidth()
max_h = screen_h - 120
max_w = screen_w - 80
scale_display = min(max_w / w, max_h / h, 1.0)

if scale_display < 1.0:
    new_w_show = int(w * scale_display)
    new_h_show = int(h * scale_display)
    img_show = cv2.resize(img_original, (new_w_show, new_h_show), interpolation=cv2.INTER_AREA)
else:
    img_show = img_original

cv2.namedWindow("Result", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Result", img_show)
cv2.waitKey(0)
cv2.destroyAllWindows()