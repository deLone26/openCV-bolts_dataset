import os
import shutil
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Папки с исходниками
RAW_DIRS = [
    os.path.join(BASE_DIR, "raw_photos", "single"),
    os.path.join(BASE_DIR, "raw_photos", "mixed")
]

# Куда будем копировать
DST_IMG_TRAIN = os.path.join(BASE_DIR, "dataset", "images", "train")
DST_IMG_VAL = os.path.join(BASE_DIR, "dataset", "images", "val")
DST_LBL_TRAIN = os.path.join(BASE_DIR, "dataset", "labels", "train")
DST_LBL_VAL = os.path.join(BASE_DIR, "dataset", "labels", "val")

os.makedirs(DST_IMG_TRAIN, exist_ok=True)
os.makedirs(DST_IMG_VAL, exist_ok=True)
os.makedirs(DST_LBL_TRAIN, exist_ok=True)
os.makedirs(DST_LBL_VAL, exist_ok=True)

pairs = []

for raw_dir in RAW_DIRS:
    if not os.path.exists(raw_dir):
        continue
        
    prefix = os.path.basename(raw_dir)
    
    # Создаем словарь всех txt-файлов в папке и её подпапках (имя без расширения -> полный путь)
    txt_files = {}
    for root, dirs, files in os.walk(raw_dir):
        for file in files:
            if file.endswith('.txt') and file != 'classes.txt':
                name_no_ext = os.path.splitext(file)[0]
                txt_files[name_no_ext] = os.path.join(root, file)
                
    # Теперь ищем фото и сопоставляем их с txt
    for root, dirs, files in os.walk(raw_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                name_no_ext = os.path.splitext(file)[0]
                img_path = os.path.join(root, file)
                
                # Если для этого фото есть txt-файл
                if name_no_ext in txt_files:
                    lbl_path = txt_files[name_no_ext]
                    out_name = f"{prefix}_{file}"
                    pairs.append((img_path, lbl_path, out_name))

# Убираем дубликаты, если скрипт случайно нашел одно и то же фото дважды
pairs = list(set(pairs))

if len(pairs) == 0:
    print("ВНИМАНИЕ: Не найдено ни одной пары 'фото + txt'.")
    print("Проверьте, что AlchemyAnnotate сохранил разметку в папки raw_photos/mixed и raw_photos/single")
else:
    # Перемешиваем и делим 80/20
    random.seed(42)
    random.shuffle(pairs)
    split_idx = int(len(pairs) * 0.8)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    def copy_pairs(pairs_list, dst_img, dst_lbl):
        for img_p, lbl_p, out_name in pairs_list:
            lbl_out_name = os.path.splitext(out_name)[0] + ".txt"
            shutil.copy(img_p, os.path.join(dst_img, out_name))
            shutil.copy(lbl_path, os.path.join(dst_lbl, lbl_out_name))

    copy_pairs(train_pairs, DST_IMG_TRAIN, DST_LBL_TRAIN)
    copy_pairs(val_pairs, DST_IMG_VAL, DST_LBL_VAL)

    print(f"Готово! Найдено фото с разметкой: {len(pairs)}")
    print(f"Train: {len(train_pairs)} фото, Val: {len(val_pairs)} фото.")