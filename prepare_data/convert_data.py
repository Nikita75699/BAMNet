#!/usr/bin/env python3
import argparse
import base64
import json
import zlib
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from bamnet_paths import get_data_path

# Значения по умолчанию можно переопределить через CLI.
DEFAULT_SRC_ROOT = get_data_path("segmentation_point(v2)")
DEFAULT_DST_ROOT = get_data_path("export_project", "segpoint")
DEFAULT_TRAIN_PATIENTS = 83
DEFAULT_VAL_PATIENTS = 0

POINT_NAMES = ["AA1","AA2","STJ1","STJ2"]

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def collect_by_patient(src_root: Path):
    """
    Собираем (img, ann) по пациентам. Возвращаем dict: {patient_id: [(img_path, ann_path), ...], ...}
    Учёт кейса ann 'xxx.png.json' ↔ img 'xxx.png'.
    """
    IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    data = {}

    def ann_candidates(ann: Path):
        s = ann.stem  # '0001_01_003.png'
        cands = [s]
        for ext in IMG_EXTS:
            if s.lower().endswith(ext):
                cands.append(s[: -len(ext)])  # '0001_01_003'
                break
        return cands

    for patient_dir in sorted([d for d in src_root.iterdir() if d.is_dir()]):
        patient_id = patient_dir.name  # '001', '002', ...
        img_dir = patient_dir / "img"
        ann_dir = patient_dir / "ann"
        if not img_dir.exists() or not ann_dir.exists():
            continue

        pairs = []
        for ann_file in ann_dir.glob("*.json"):
            stems = ann_candidates(ann_file)

            # 1) точное совпадение
            found = None
            for s in stems:
                exact = img_dir / s
                if exact.exists() and exact.is_file():
                    found = exact
                    break
            # 2) подбор расширения
            if found is None:
                for s in stems:
                    for ext in IMG_EXTS:
                        cand = img_dir / f"{s}{ext}"
                        if cand.exists():
                            found = cand
                            break
                    if found is not None:
                        break
            if found is not None:
                pairs.append((found, ann_file))

        if pairs:
            data[patient_id] = pairs

    return data

def load_sly_annotation(ann_path: Path):
    with open(ann_path, "r", encoding="utf-8") as f:
        ann = json.load(f)
    size = ann.get("size", {})
    H = int(size.get("height", 0))
    W = int(size.get("width", 0))
    objects = ann.get("objects", [])
    return H, W, objects

def _paste_patch(dst01: np.ndarray, patch01: np.ndarray, y0: int, x0: int):
    H, W = dst01.shape[:2]
    h, w = patch01.shape[:2]
    y1, x1 = min(H, y0 + h), min(W, x0 + w)
    if y0 >= H or x0 >= W or y1 <= 0 or x1 <= 0:
        return
    dy0 = max(0, -y0); dx0 = max(0, -x0)
    y0c = max(0, y0);  x0c = max(0, x0)
    sub = patch01[dy0:dy0 + (y1 - y0c), dx0:dx0 + (x1 - x0c)]
    np.maximum(dst01[y0c:y1, x0c:x1], sub, out=dst01[y0c:y1, x0c:x1])

def decode_bitmap(bitmap_payload: dict):
    org = bitmap_payload.get("origin")
    if isinstance(org, dict):
        ext = org.get("points", {}).get("exterior", [])
        if ext and len(ext[0]) == 2:
            x0, y0 = int(ext[0][0]), int(ext[0][1])
        else:
            x0 = y0 = 0
    elif isinstance(org, (list, tuple)) and len(org) >= 2:
        x0, y0 = int(org[0]), int(org[1])
    else:
        x0 = y0 = 0

    raw = base64.b64decode(bitmap_payload["data"])
    try:
        raw = zlib.decompress(raw)
    except zlib.error:
        pass

    buf = np.frombuffer(raw, dtype=np.uint8)
    patch = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if patch is None:
        raise ValueError("cv2.imdecode returned None")

    if patch.ndim == 3:
        if patch.shape[2] == 4:
            patch = patch[:, :, 3]
        else:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    patch01 = (patch > 0).astype(np.uint8)
    return patch01, y0, x0

def build_mask_and_points(objects, H, W):
    """
    Маска: объединяем ВСЕ bitmap-фигуры.
    Точки: любые point-фигуры; отсутствующие из POINT_NAMES добиваем visible=0.
    Маска возвращается как 0/1.
    """
    mask01 = np.zeros((H, W), dtype=np.uint8)
    points = {}

    for obj in objects:
        gtype = (obj.get("geometryType") or obj.get("shape") or "").lower()
        cls = (obj.get("classTitle") or "").strip()

        if gtype == "bitmap" and "bitmap" in obj:
            try:
                patch01, y0, x0 = decode_bitmap(obj["bitmap"])
                _paste_patch(mask01, patch01, int(y0), int(x0))
            except Exception as e:
                print(f"[WARN] Bitmap decode failed (id={obj.get('id')} class={cls}): {e}")

        elif gtype == "point" and "points" in obj:
            ext = obj["points"].get("exterior") or []
            if len(ext) >= 1:
                x, y = float(ext[0][0]), float(ext[0][1])
                name = (cls or "P").strip().upper()
                points[name] = {"x": x, "y": y, "visible": 1}

    # добиваем обязательные имена
    for nm in POINT_NAMES:
        if nm not in points:
            points[nm] = {"x": 0.0, "y": 0.0, "visible": 0}

    mask01[mask01 != 0] = 1
    return mask01, points

def save_pair(split_root: Path, img_path: Path, ann_path: Path, patient_id: str):
    H, W, objects = load_sly_annotation(ann_path)

    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"[WARN] Cannot read {img_path}")
        return
    if H == 0 or W == 0:
        H, W = img_bgr.shape[:2]

    mask01, pts_raw = build_mask_and_points(objects, H, W)

    # пути
    img_out = split_root / "images" / f"{img_path.stem}.png"
    mask_out = split_root / "masks"  / f"{img_path.stem}.png"
    pts_out  = split_root / "points" / f"{img_path.stem}.json"
    ensure_dir(img_out.parent); ensure_dir(mask_out.parent); ensure_dir(pts_out.parent)

    # сохраняем
    cv2.imwrite(str(img_out), img_bgr)
    cv2.imwrite(str(mask_out), (mask01.astype(np.uint8) * 255))

    # точки с нормировкой
    points = {}
    for nm in POINT_NAMES:
        r = pts_raw.get(nm, {"x": 0.0, "y": 0.0, "visible": 0})
        x = float(r["x"]); y = float(r["y"]); vis = int(r["visible"])
        x_norm = (x / float(W)) if W > 0 else 0.0
        y_norm = (y / float(H)) if H > 0 else 0.0
        points[nm] = {
            "x": int(round(x)),
            "y": int(round(y)),
            "x_norm": float(x_norm),
            "y_norm": float(y_norm),
            "visible": int(1 if vis else 0),
        }

    payload = {
        "image_filename": img_out.name,
        "image_path": str(img_out.resolve()),
        "width": int(W),
        "height": int(H),
        "patient_id": patient_id,               # ← папка 001/002/...
        "points": points,
    }
    with open(pts_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(
        description="Конвертация Supervisely-проекта в формат BAMNet images/masks/points."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_SRC_ROOT),
        help="Путь к Supervisely-проекту с папками <patient>/img и <patient>/ann",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_DST_ROOT),
        help="Куда сохранить BAMNet-датасет",
    )
    parser.add_argument(
        "--train-patients",
        type=int,
        default=DEFAULT_TRAIN_PATIENTS,
        help="Сколько первых пациентов отправить в train",
    )
    parser.add_argument(
        "--val-patients",
        type=int,
        default=DEFAULT_VAL_PATIENTS,
        help="Сколько следующих пациентов отправить в val",
    )
    args = parser.parse_args()

    src_root = Path(args.input)
    dst_root = Path(args.output)

    if not src_root.exists():
        print(f"[ERR] входная папка не найдена: {src_root}")
        return

    by_patient = collect_by_patient(src_root)
    patients = sorted(by_patient.keys())
    if not patients:
        print("[ERR] не найдены пациенты/пары ann+img")
        return

    train_patients = patients[:args.train_patients]
    val_patients = patients[args.train_patients:args.train_patients + args.val_patients]

    if len(train_patients) < args.train_patients or len(val_patients) < args.val_patients:
        print(
            f"[WARN] доступно пациентов: {len(patients)}; "
            f"train={len(train_patients)}, val={len(val_patients)}"
        )

    print(f"[INFO] patients total={len(patients)} | train={len(train_patients)} | val={len(val_patients)}")
    print(f"[INFO] input={src_root.resolve()}")
    print(f"[INFO] output={dst_root.resolve()}")

    for split_name, plist in [("train", train_patients), ("val", val_patients)]:
        if not plist:
            continue
        split_root = dst_root / split_name
        ensure_dir(split_root)
        items = [(img, ann, pid) for pid in plist for (img, ann) in by_patient[pid]]
        print(f"[INFO] Обработка {split_name}: {len(items)} изображений, пациентов: {len(plist)}")
        for img_path, ann_path, pid in tqdm(items, desc=split_name, unit="img"):
            save_pair(split_root, img_path, ann_path, pid)

    print(f"[OK] Готово → {dst_root.resolve()}")

if __name__ == "__main__":
    main()
