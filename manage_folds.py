import os
import glob
import json
import shutil
import argparse

def extract_meta(folds_dir, out_meta_json):
    """
    Обходит все папки fold_X/train/images и fold_X/val/images,
    записывая в JSON словарь: какие файлы изображений лежат в каждом сплите.
    """
    meta = {}
    folds = glob.glob(os.path.join(folds_dir, "fold_*"))
    
    for fold_path in sorted(folds):
        fold_name = os.path.basename(fold_path)
        meta[fold_name] = {"train": [], "val": []}
        
        for split in ["train", "val"]:
            img_dir = os.path.join(fold_path, split, "images")
            if os.path.exists(img_dir):
                images = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                meta[fold_name][split] = sorted(images)
                
    with open(out_meta_json, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)
        
    print(f"✅ Метаданные собраны и сохранены в {out_meta_json}")
    for fold_name, splits in meta.items():
        print(f"  {fold_name}: {len(splits['train'])} train, {len(splits['val'])} val")

def distribute_annotations(meta_json, folds_dir, src_root):
    """
    Читает JSON-файл распределения и раскидывает сырые маски/точки по фолдам
    в папки masks и points соответственно. Учитывает, что новые аннотации лежат в 
    подпапках train и val внутри src_root.
    """
    if not os.path.exists(meta_json):
        print(f"❌ Ошибка: Файл метаданных {meta_json} не найден.")
        return

    with open(meta_json, 'r', encoding='utf-8') as f:
        meta = json.load(f)
        
    copied_masks = 0
    copied_points = 0
    missing = 0

    print("🚀 Распределяем аннотации по фолдам...")
    for fold_name, splits in meta.items():
        for split, images in splits.items():
            masks_dest = os.path.join(folds_dir, fold_name, split, "masks")
            points_dest = os.path.join(folds_dir, fold_name, split, "points")
            
            os.makedirs(masks_dest, exist_ok=True)
            os.makedirs(points_dest, exist_ok=True)
            
            for img_name in images:
                stem = os.path.splitext(img_name)[0]
                
                # Ищем маску (.png) и точки (.json) в центральной папке, проверяя и train, и val
                src_mask = None
                src_point = None
                
                for src_split in ["train", "val"]:
                    cand_mask = os.path.join(src_root, src_split, "masks", f"{stem}.png")
                    cand_point = os.path.join(src_root, src_split, "points", f"{stem}.json")
                    if os.path.exists(cand_mask):
                        src_mask = cand_mask
                    if os.path.exists(cand_point):
                        src_point = cand_point
                
                # Копируем маску
                if src_mask and os.path.exists(src_mask):
                    shutil.copy2(src_mask, os.path.join(masks_dest, f"{stem}.png"))
                    copied_masks += 1
                else:
                    missing += 1
                    print(f"  [WARN] Маска не найдена для {img_name}")
                    
                # Копируем точки
                if src_point and os.path.exists(src_point):
                    shutil.copy2(src_point, os.path.join(points_dest, f"{stem}.json"))
                    copied_points += 1
                else:
                    print(f"  [WARN] Точки не найдены для {img_name}")
                    
    print(f"✅ Успешно распределено: {copied_masks} масок и {copied_points} точек.")
    if missing > 0:
        print(f"⚠️ Отсутствует масок/точек для {missing} изображений.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Управление фолдами: сбор меты и распределение аннотаций")
    parser.add_argument("--action", type=str, choices=["extract_meta", "distribute"], required=True, 
                        help="extract_meta - собрать мету из структуры, distribute - раскидать маски и точки по мете")
    parser.add_argument("--folds_dir", type=str, default="segpoint_folds", 
                        help="Путь к корневой папке с фолдами")
    parser.add_argument("--meta_file", type=str, default="folds_meta.json", 
                        help="Файл для сохранения/чтения распределения (JSON)")
    
    # Эти параметры нужны только для distribute
    parser.add_argument("--src_root", type=str, default="MANet_dataset", 
                        help="Путь к корневой папке MANet_dataset (где лежат подпапки train/val с масками и точками)")
    
    args = parser.parse_args()
    
    if args.action == "extract_meta":
        extract_meta(args.folds_dir, args.meta_file)
    elif args.action == "distribute":
        distribute_annotations(args.meta_file, args.folds_dir, args.src_root)
