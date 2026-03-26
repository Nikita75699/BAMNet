#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Подготовка данных для YOLO Object Detection из export_project/segpoint"""

import json
from pathlib import Path
import argparse
import shutil
from bamnet_paths import get_data_path

BOX_SIZE = 64
POINT_NAMES = ["AA1", "AA2", "STJ1", "STJ2"]
POINT_TO_CLASS = {name: idx for idx, name in enumerate(POINT_NAMES)}
SPLIT_NAMES = ["train", "val"]


def convert_point_to_yolo_bbox(
    class_id: int,
    x_norm: float,
    y_norm: float,
    image_width: int = 1024,
    image_height: int = 1024,
) -> str:
    """Преобразует нормализованные координаты точки в YOLO bbox (class xc yc w h).
    BOX_SIZE - размер бокса в пикселях, нормализуем по размеру изображения.
    """
    safe_w = max(1, int(image_width))
    safe_h = max(1, int(image_height))
    half_w = BOX_SIZE / 2 / safe_w
    half_h = BOX_SIZE / 2 / safe_h

    x1 = max(0.0, float(x_norm) - half_w)
    y1 = max(0.0, float(y_norm) - half_h)
    x2 = min(1.0, float(x_norm) + half_w)
    y2 = min(1.0, float(y_norm) + half_h)

    bbox_w = x2 - x1
    bbox_h = y2 - y1
    bbox_xc = x1 + bbox_w / 2
    bbox_yc = y1 + bbox_h / 2

    return f"{class_id} {bbox_xc:.6f} {bbox_yc:.6f} {bbox_w:.6f} {bbox_h:.6f}"


def process_split(split_path: Path, output_dir: Path) -> tuple[int, int]:
    """Обрабатывает train или val split."""

    points_dir = split_path / "points"
    images_dir = split_path / "images"

    if not points_dir.exists():
        print(f"[WARN] Нет папки points: {points_dir}")
        return 0, 0

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
    duplicates_removed_total = 0

    for json_file in points_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                point_data = json.load(f)

            image_filename = point_data.get("image_filename", json_file.stem + ".png")
            points = point_data.get("points", {})
            image_width = int(point_data.get("width", 1024))
            image_height = int(point_data.get("height", 1024))

            lines = []
            for name in POINT_NAMES:
                point = points.get(name)
                if point:
                    x_norm = point.get("x_norm")
                    y_norm = point.get("y_norm")
                    visible = point.get("visible", 0)
                    if x_norm is not None and y_norm is not None and visible:
                        class_id = POINT_TO_CLASS[name]
                        line = convert_point_to_yolo_bbox(
                            class_id=class_id,
                            x_norm=x_norm,
                            y_norm=y_norm,
                            image_width=image_width,
                            image_height=image_height,
                        )
                        lines.append(line)

            if lines:
                unique_lines = list(dict.fromkeys(lines))
                duplicates_removed_total += len(lines) - len(unique_lines)

                txt_file = txt_output_dir / f"{json_file.stem}.txt"
                with open(txt_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(unique_lines) + "\n")

                image_src = images_dir / image_filename
                if not image_src.exists():
                    for ext in [".png", ".jpg", ".jpeg"]:
                        if (
                            images_dir / (image_filename.rsplit(".", 1)[0] + ext)
                        ).exists():
                            image_src = images_dir / (
                                image_filename.rsplit(".", 1)[0] + ext
                            )
                            break

                if image_src.exists() and image_src.is_file():
                    img_dst = img_output_dir / image_src.name
                    img_dst.write_bytes(image_src.read_bytes())

                processed += 1

        except Exception as e:
            print(f"[ERR] Ошибка при обработке {json_file}: {e}")
            continue

    return processed, duplicates_removed_total


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


def process_dataset_root(input_path: Path, output_path: Path) -> tuple[int, int]:
    """Обрабатывает один dataset root c подпапками train/val."""
    total_processed = 0
    total_duplicates_removed = 0

    for split_name in SPLIT_NAMES:
        split_path = input_path / split_name
        if not split_path.exists():
            continue

        print(f"[INFO] Обработка {split_name}:")
        count, duplicates_removed = process_split(split_path, output_path)
        total_processed += count
        total_duplicates_removed += duplicates_removed
        print(f"[INFO]   Обработано файлов: {count}")
        print(f"[INFO]   Удалено дубликатов: {duplicates_removed}")

    return total_processed, total_duplicates_removed


def main():
    parser = argparse.ArgumentParser(description="Подготовка данных для YOLO OD")
    parser.add_argument(
        "--input",
        type=str,
        default=str(get_data_path("export_project", "segpoint")),
        help="Путь к export_project/segpoint или export_project/segpoint_folds во внешнем data root",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(get_data_path("export_project", "yolo_data")),
        help="Путь к выходной папке",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"[ERR] Входная папка не найдена: {input_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    print("[INFO] Подготовка данных для YOLO OD")
    print(f"[INFO] Входная папка: {input_path}")
    print(f"[INFO] Выходная папка: {output_path}")
    print(f"[INFO] Размер бокса: {BOX_SIZE}x{BOX_SIZE} px")

    datasets = discover_dataset_roots(input_path)
    if not datasets:
        print(f"[ERR] Не найден dataset root (ожидаются train/val или fold_*/train,val): {input_path}")
        return

    total_processed = 0
    total_duplicates_removed = 0

    for dataset_name, dataset_root in datasets:
        dataset_output = output_path / dataset_name if dataset_name else output_path
        dataset_output.mkdir(parents=True, exist_ok=True)
        if dataset_name:
            print(f"\n[INFO] === Обработка {dataset_name} ===")
            print(f"[INFO] Fold input: {dataset_root}")
            print(f"[INFO] Fold output: {dataset_output}")

        count, duplicates_removed = process_dataset_root(dataset_root, dataset_output)
        total_processed += count
        total_duplicates_removed += duplicates_removed

    print(f"\n[INFO] Всего обработано файлов: {total_processed}")
    print(f"[INFO] Всего удалено дубликатов: {total_duplicates_removed}")
    print(f"[INFO] Структура готова для YOLO:")
    if len(datasets) == 1 and not datasets[0][0]:
        print(f"      {output_path}/images/  - изображения")
        print(f"      {output_path}/labels/  - метки YOLO")
    else:
        print(f"      {output_path}/fold_XX/images/  - изображения")
        print(f"      {output_path}/fold_XX/labels/  - метки YOLO")


if __name__ == "__main__":
    main()
