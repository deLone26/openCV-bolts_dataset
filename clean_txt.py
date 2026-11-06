import os
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_DIR = os.path.join(BASE_DIR, "labels_raw")

for txt_file in glob.glob(os.path.join(LABELS_DIR, "*.txt")):
    if "classes" in txt_file:
        continue
        
    with open(txt_file, "r") as f:
        lines = f.readlines()
        
    # Оставляем только строки, которые начинаются с цифры (0 или 1)
    valid_lines = []
    for line in lines:
        clean_line = line.strip()
        if clean_line and (clean_line.startswith("0") or clean_line.startswith("1")):
            valid_lines.append(clean_line)
            
    # Перезаписываем файл только валидными строками
    with open(txt_file, "w") as f:
        f.write("\n".join(valid_lines))

print("Все txt-файлы очищены от пустых строк и мусора.")