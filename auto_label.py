import cv2
import os
import glob
import numpy as np

# --- НАСТРОЙКИ ПУТЕЙ ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIRS = [
    os.path.join(BASE_DIR, "raw_photos", "single"), 
    os.path.join(BASE_DIR, "raw_photos", "mixed")
]
OUTPUT_LABELS = os.path.join(BASE_DIR, "labels_raw")
OUTPUT_PREVIEW = os.path.join(BASE_DIR, "preview")

os.makedirs(OUTPUT_LABELS, exist_ok=True)
os.makedirs(OUTPUT_PREVIEW, exist_ok=True)

# Базовый порог площади (10 000 пикселей)
MIN_AREA = 10000  

def process_image(img_path, prefix):
    img = cv2.imread(img_path)
    if img is None:
        return
    
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Базовый пайплайн: размытие -> порог -> морфология
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    txt_lines = []
    img_preview = img.copy()
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue # Отбрасываем мелкий шум
            
        x, y, bw, bh = cv2.boundingRect(cnt)
        
        # Просто базовая эвристика для класса (потом поправите в LabelImg)
        aspect_ratio = float(bw) / bh
        epsilon = 0.03 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        if 5 <= len(approx) <= 8 or (0.8 < aspect_ratio < 1.2):
            class_id = 1 # Гайка
        else:
            class_id = 0 # Болт
        
        # Перевод в YOLO
        x_center = (x + bw / 2.0) / w
        y_center = (y + bh / 2.0) / h
        norm_w = bw / w
        norm_h = bh / h
        
        txt_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
        
        # Рисуем (0 - зеленый, 1 - красный)
        color = (0, 255, 0) if class_id == 0 else (0, 0, 255) 
        cv2.rectangle(img_preview, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(img_preview, str(class_id), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Сохраняем результаты
    filename = os.path.basename(img_path)
    name, _ = os.path.splitext(filename)
    out_name = f"{prefix}_{name}"
    
    if txt_lines:
        with open(os.path.join(OUTPUT_LABELS, f"{out_name}.txt"), "w") as f:
            f.write("\n".join(txt_lines))
        cv2.imwrite(os.path.join(OUTPUT_PREVIEW, f"{out_name}_preview.jpg"), img_preview)
        print(f"OK: {out_name} | Объектов: {len(txt_lines)}")
    else:
        print(f"ПРОПУЩЕНО: {out_name}")

# --- ЗАПУСК ---
if __name__ == "__main__":
    for folder in INPUT_DIRS:
        if os.path.exists(folder):
            prefix = os.path.basename(folder)
            print(f"\n--- Сканирую папку: {folder} ---")
            extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
            for ext in extensions:
                for img_path in glob.glob(os.path.join(folder, ext)):
                    process_image(img_path, prefix)
        else:
            print(f"ВНИМАНИЕ: Папка {folder} не существует!")
    print("\nОбработка завершена. Проверьте папку preview.")