import os
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
labels_dir = os.path.join(BASE_DIR, "labels_raw")

# Проходим по всем txt-файлам
for file_path in glob.glob(os.path.join(labels_dir, "*.txt")):
    filename = os.path.basename(file_path)
    
    # Если файл начинается с mixed_ или single_
    if filename.startswith("mixed_") or filename.startswith("single_"):
        # Отрезаем префикс (например, "mixed_21.txt" -> "21.txt")
        new_filename = filename.split("_", 1)[1]
        new_path = os.path.join(labels_dir, new_filename)
        
        # Переименовываем
        os.rename(file_path, new_path)
        print(f"Переименован: {filename} -> {new_filename}")

print("Готово! Теперь имена txt совпадают с фото.")