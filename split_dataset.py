import os
import shutil
import random
from collections import defaultdict

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

# Очищаем старые папки перед сборкой, чтобы не осталось мусора
for folder in [DST_IMG_TRAIN, DST_IMG_VAL, DST_LBL_TRAIN, DST_LBL_VAL]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

pairs = []

# Собираем все пары фото + txt
for raw_dir in RAW_DIRS:
    if not os.path.exists(raw_dir):
        continue
        
    prefix = os.path.basename(raw_dir)
    
    txt_files = {}
    for root, dirs, files in os.walk(raw_dir):
        for file in files:
            if file.endswith('.txt') and file != 'classes.txt':
                name_no_ext = os.path.splitext(file)[0]
                txt_files[name_no_ext] = os.path.join(root, file)
                
    for root, dirs, files in os.walk(raw_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                name_no_ext = os.path.splitext(file)[0]
                img_path = os.path.join(root, file)
                
                if name_no_ext in txt_files:
                    lbl_path = txt_files[name_no_ext]
                    
                    # Читаем txt, чтобы понять, какие классы там есть
                    with open(lbl_path, 'r') as f:
                        lines = [line.strip() for line in f if line.strip()]
                    
                    if len(lines) == 0:
                        continue # Пропускаем пустые txt (фото без деталей)
                    
                    # Собираем уникальные классы на этом фото
                    classes_in_img = set()
                    for line in lines:
                        class_id = line.split()[0]
                        classes_in_img.add(class_id)
                        
                    out_name = f"{prefix}_{file}"
                    pairs.append({
                        'img': img_path,
                        'lbl': lbl_path,
                        'name': out_name,
                        'classes': classes_in_img,
                        'num_objects': len(lines)
                    })

# Убираем дубликаты
unique_pairs = []
seen = set()
for p in pairs:
    if p['name'] not in seen:
        unique_pairs.append(p)
        seen.add(p['name'])
pairs = unique_pairs

if len(pairs) == 0:
    print("ВНИМАНИЕ: Не найдено ни одной пары 'фото + txt'.")
else:
    # Стратифицированное разделение
    random.seed(42)
    
    # Группируем фото по их содержимому
    only_bolt = [p for p in pairs if p['classes'] == {'0'}]
    only_nut = [p for p in pairs if p['classes'] == {'1'}]
    both = [p for p in pairs if len(p['classes']) > 1]
    
    random.shuffle(only_bolt)
    random.shuffle(only_nut)
    random.shuffle(both)
    
    # Функция для деления списка 80/20
    def split_list(lst, ratio=0.8):
        split_idx = int(len(lst) * ratio)
        return lst[:split_idx], lst[split_idx:]
    
    train_pairs, val_pairs = [], []
    
    # Делим каждую группу отдельно
    for group in [only_bolt, only_nut, both]:
        tr, va = split_list(group)
        train_pairs.extend(tr)
        val_pairs.extend(va)
    
    # Перемешиваем итоговые списки
    random.shuffle(train_pairs)
    random.shuffle(val_pairs)

    def copy_pairs(pairs_list, dst_img, dst_lbl):
        for p in pairs_list:
            lbl_out_name = os.path.splitext(p['name'])[0] + ".txt"
            shutil.copy(p['img'], os.path.join(dst_img, p['name']))
            shutil.copy(p['lbl'], os.path.join(dst_lbl, lbl_out_name))

    copy_pairs(train_pairs, DST_IMG_TRAIN, DST_LBL_TRAIN)
    copy_pairs(val_pairs, DST_IMG_VAL, DST_LBL_VAL)

    print(f"Готово! Всего фото с разметкой: {len(pairs)}")
    print(f"Из них: только болты - {len(only_bolt)}, только гайки - {len(only_nut)}, смешанные - {len(both)}")
    print(f"Train: {len(train_pairs)} фото, Val: {len(val_pairs)} фото.")