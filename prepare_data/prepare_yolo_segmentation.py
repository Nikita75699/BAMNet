#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Подготовка данных для YOLO Segmentation из export_project/segpoint"""

import json
from pathlib import Path
import argparse
import cv2
import shutil
from bamnet_paths import get_data_path

POINT_NAMES = ["AA1", "AA2", "STJ1", "STJ2"]
SPLIT_NAMES = ["train", "val"]


def calculate_bbox(
    points: dict, pad: float = 0.25
) -> tuple[float, float, float, float] | None:
    """Вычисляет нормированный bbox по 4 точкам с отступом."""
    xs, ys = [], []

    for name in POINT_NAMES:
        p = points.get(name)
        if p and p.get("x_norm") is not None and p.get("y_norm") is not None:
            xs.append(p["x_norm"])
            ys.append(p["y_norm"])

    if not xs:
        return None

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    w = (x_max - x_min) * (1 + pad)
    h = (y_max - y_min) * (1 + pad)

    xc = (x_min + x_max) / 2
    yc = (y_min + y_max) / 2

    w = min(w, 1.0)
    h = min(h, 1.0)

    return xc, yc, w, h


def convert_points_to_yolo_keypoints(
    points: dict,
) -> tuple[str, str] | tuple[None, None]:
    """Преобразует точки в YOLO keypoint format (xc, yc, v) для одного класса."""
    keypoints = []

    for name in POINT_NAMES:
        point = points.get(name)
        if point:
            x_norm = point.get("x_norm")
            y_norm = point.get("y_norm")
            visible = point.get("visible", 0)
            if x_norm is not None and y_norm is not None:
                v = 2 if visible else 0
                keypoints.append(f"{x_norm:.6f} {y_norm:.6f} {v}")

    if not keypoints:
        return None, None

    bbox = calculate_bbox(points)
    if bbox is None:
        return None, None

    xc, yc, w, h = bbox
    bbox_line = f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
    keypoints_output = " ".join(keypoints)

    return bbox_line, keypoints_output


def convert_mask_to_yolo_polygon(
    mask_path: Path, image_width: int, image_height: int
) -> list[str]:
    """Конвертирует бинарную маску в YOLO segmentation format (polygon points)."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    yolo_polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue

        points = []
        for pt in contour:
            x_norm = pt[0][0] / image_width
            y_norm = pt[0][1] / image_height
            points.append(f"{x_norm:.6f} {y_norm:.6f}")

        if points:
            yolo_polygons.append(" ".join(points))

    return yolo_polygons


def process_split(split_path: Path, output_dir: Path) -> int:
    """Обрабатывает train или val split с масками для YOLO Segmentation."""
    points_dir = split_path / "points"
    images_dir = split_path / "images"
    masks_dir = split_path / "masks"

    if not points_dir.exists():
        print(f"[WARN] Нет папки points: {points_dir}")
        return 0

    txt_output_dir = output_dir / "labels" / split_path.name
    img_output_dir = output_dir / "images" / split_path.name

    # Очищаем split-папки перед генерацией, чтобы не оставались старые файлы.
    if txt_output_dir.exists():
        shutil.rmtree(txt_output_dir)
    if img_output_dir.exists():
        shutil.rmtree(img_output_dir)

    txt_output_dir.mkdir(parents=True, exist_ok=True)
    img_output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0

    for json_file in points_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                point_data = json.load(f)

            image_filename = point_data.get("image_filename", json_file.stem + ".png")
            image_width = point_data.get("width", 1024)
            image_height = point_data.get("height", 1024)

            image_src = images_dir / image_filename
            if not image_src.exists():
                for ext in [".png", ".jpg", ".jpeg"]:
                    if (images_dir / (image_filename.rsplit(".", 1)[0] + ext)).exists():
                        image_src = images_dir / (
                            image_filename.rsplit(".", 1)[0] + ext
                        )
                        break

            mask_src = masks_dir / image_filename
            if not mask_src.exists():
                for ext in [".png", ".jpg", ".jpeg"]:
                    if (masks_dir / (image_filename.rsplit(".", 1)[0] + ext)).exists():
                        mask_src = masks_dir / (image_filename.rsplit(".", 1)[0] + ext)
                        break

            if image_src.exists():
                image = cv2.imread(str(image_src), cv2.IMREAD_UNCHANGED)
                if image is not None:
                    image_height, image_width = image.shape[:2]

            if not mask_src.exists():
                print(f"[WARN] Пропуск {json_file}: не найдена маска {mask_src}")
                continue

            polygons = convert_mask_to_yolo_polygon(mask_src, image_width, image_height)
            if not polygons:
                print(f"[WARN] Пропуск {json_file}: пустая/некорректная маска")
                continue

            txt_file = txt_output_dir / f"{json_file.stem}.txt"
            with open(txt_file, "w", encoding="utf-8") as f:
                for polygon in polygons:
                    f.write(f"0 {polygon}\n")

            if image_src.exists() and image_src.is_file():
                img_dst = img_output_dir / image_src.name
                img_dst.write_bytes(image_src.read_bytes())

            processed += 1

        except Exception as e:
            print(f"[ERR] Ошибка при обработке {json_file}: {e}")
            continue

    return processed


def discover_dataset_roots(input_path: Path) -> list[tuple[str, Path]]:
    """Автоопределяет одиночный dataset root или папку с fold_*."""
    has_direct_splits = any((input_path / split_name).exists() for split_name in SPLIT_NAMES)

    fold_dirs = sorted(
        [
            d
            for d in input_path.iterdir()
            if d.is_dir()
            and d.name.startswith("fold_")
            and any((d / split_name).exists() for split_name in SPLIT_NAMES)
        ]
    )

    if fold_dirs:
        if has_direct_splits:
            print("[WARN] Найдены и train/val, и fold_*; используется режим fold_*")
        return [(fold_dir.name, fold_dir) for fold_dir in fold_dirs]

    if has_direct_splits:
        return [("", input_path)]

    return []


def process_dataset_root(input_path: Path, output_path: Path) -> int:
    """Обрабатывает один dataset root c подпапками train/val."""
    total_processed = 0

    for split_name in SPLIT_NAMES:
        split_path = input_path / split_name
        if not split_path.exists():
            continue

        masks_exist = (split_path / "masks").exists()
        if not masks_exist:
            print(f"[WARN] Нет папки masks в {split_path.name}, пропуск")
            continue

        print(f"[INFO] Обработка {split_name}:")
        count = process_split(split_path, output_path)
        total_processed += count
        print(f"[INFO]   Обработано файлов: {count}")

    return total_processed


def main():
    parser = argparse.ArgumentParser(
        description="Подготовка данных для YOLO Segmentation"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(get_data_path("export_project", "segpoint_folds")),
        help="Путь к export_project/segpoint или export_project/segpoint_folds во внешнем data root",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(get_data_path("export_project", "yolo_segmentation")),
        help="Путь к выходной папке",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"[ERR] Входная папка не найдена: {input_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    print("[INFO] Подготовка данных для YOLO Segmentation")
    print(f"[INFO] Входная папка: {input_path}")
    print(f"[INFO] Выходная папка: {output_path}")
    print("[INFO] Формат разметки: YOLO Segmentation polygons")

    datasets = discover_dataset_roots(input_path)
    if not datasets:
        print(f"[ERR] Не найден dataset root (ожидаются train/val или fold_*/train,val): {input_path}")
        return

    total_processed = 0

    for dataset_name, dataset_root in datasets:
        dataset_output = output_path / dataset_name if dataset_name else output_path
        dataset_output.mkdir(parents=True, exist_ok=True)
        if dataset_name:
            print(f"\n[INFO] === Обработка {dataset_name} ===")
            print(f"[INFO] Fold input: {dataset_root}")
            print(f"[INFO] Fold output: {dataset_output}")

        total_processed += process_dataset_root(dataset_root, dataset_output)

    print(f"\n[INFO] Всего обработано файлов: {total_processed}")
    print(f"[INFO] Структура готова для YOLO Segmentation:")
    if len(datasets) == 1 and not datasets[0][0]:
        print(f"      {output_path}/images/  - изображения")
        print(f"      {output_path}/labels/  - метки (polygons)")
    else:
        print(f"      {output_path}/fold_XX/images/  - изображения")
        print(f"      {output_path}/fold_XX/labels/  - метки (polygons)")


if __name__ == "__main__":
    main()
