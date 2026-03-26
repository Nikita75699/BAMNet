#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Подготовка данных для YOLO Keypoint из export_project/segpoint"""

import json
from pathlib import Path
import argparse
import math
import shutil
from bamnet_paths import get_data_path

POINT_NAMES = ["AA1", "AA2", "STJ1", "STJ2"]
SPLIT_NAMES = ["train", "val"]
DEFAULT_POINT_TEMPLATE = {
    "AA1": (0.35, 0.60),
    "AA2": (0.65, 0.60),
    "STJ1": (0.40, 0.40),
    "STJ2": (0.60, 0.40),
}


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


def clamp_01(value: float) -> float:
    return max(0.0, min(1.0, value))


def build_point_template(input_path: Path) -> dict[str, tuple[float, float]]:
    """Строит усредненный шаблон точек по train/val для восстановления пропусков."""
    sums = {name: [0.0, 0.0, 0] for name in POINT_NAMES}

    for split_name in ["train", "val"]:
        points_dir = input_path / split_name / "points"
        if not points_dir.exists():
            continue

        for json_file in points_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    point_data = json.load(f)
                points = point_data.get("points", {})
            except Exception:
                continue

            for name in POINT_NAMES:
                point = points.get(name)
                if point and point.get("x_norm") is not None and point.get("y_norm") is not None:
                    x_norm = float(point["x_norm"])
                    y_norm = float(point["y_norm"])
                    sums[name][0] += x_norm
                    sums[name][1] += y_norm
                    sums[name][2] += 1

    template = dict(DEFAULT_POINT_TEMPLATE)
    for name in POINT_NAMES:
        count = sums[name][2]
        if count > 0:
            template[name] = (sums[name][0] / count, sums[name][1] / count)

    return template


def estimate_similarity_transform(
    src_points: list[tuple[float, float]],
    dst_points: list[tuple[float, float]],
) -> tuple[float, float, float, float, float] | None:
    """Оценивает similarity transform (scale, cos, sin, tx, ty) из src -> dst."""
    if len(src_points) != len(dst_points) or len(src_points) < 2:
        return None

    src_cx = sum(p[0] for p in src_points) / len(src_points)
    src_cy = sum(p[1] for p in src_points) / len(src_points)
    dst_cx = sum(p[0] for p in dst_points) / len(dst_points)
    dst_cy = sum(p[1] for p in dst_points) / len(dst_points)

    src_centered = [(x - src_cx, y - src_cy) for x, y in src_points]
    dst_centered = [(x - dst_cx, y - dst_cy) for x, y in dst_points]

    a = 0.0
    b = 0.0
    denom = 0.0
    for (sx, sy), (dx, dy) in zip(src_centered, dst_centered):
        a += sx * dx + sy * dy
        b += sx * dy - sy * dx
        denom += sx * sx + sy * sy

    if denom <= 1e-12:
        return None

    theta = math.atan2(b, a)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    scale_num = 0.0
    for (sx, sy), (dx, dy) in zip(src_centered, dst_centered):
        rx = cos_t * sx - sin_t * sy
        ry = sin_t * sx + cos_t * sy
        scale_num += dx * rx + dy * ry

    scale = scale_num / denom
    if not math.isfinite(scale) or scale <= 1e-8:
        return None

    tx = dst_cx - scale * (cos_t * src_cx - sin_t * src_cy)
    ty = dst_cy - scale * (sin_t * src_cx + cos_t * src_cy)
    return scale, cos_t, sin_t, tx, ty


def recover_missing_points(
    points: dict,
    point_template: dict[str, tuple[float, float]],
) -> int:
    """Восстанавливает недостающие точки по минимум 2 известным с помощью шаблона."""
    known_names = []
    src_points = []
    dst_points = []
    missing_names = []

    for name in POINT_NAMES:
        point = points.get(name)
        if point and point.get("x_norm") is not None and point.get("y_norm") is not None:
            known_names.append(name)
            src_points.append(point_template[name])
            dst_points.append((float(point["x_norm"]), float(point["y_norm"])))
        else:
            missing_names.append(name)

    if len(missing_names) == 0 or len(known_names) < 2:
        return 0

    transform = estimate_similarity_transform(src_points, dst_points)
    if transform is None:
        return 0

    scale, cos_t, sin_t, tx, ty = transform
    recovered = 0

    for name in missing_names:
        sx, sy = point_template[name]
        x_est = scale * (cos_t * sx - sin_t * sy) + tx
        y_est = scale * (sin_t * sx + cos_t * sy) + ty
        points[name] = {
            "x_norm": clamp_01(x_est),
            "y_norm": clamp_01(y_est),
            "visible": 1,
            "imputed": True,
        }
        recovered += 1

    return recovered


def convert_points_to_yolo_pose_line(
    points: dict,
    point_template: dict[str, tuple[float, float]],
) -> tuple[str | None, int]:
    """Преобразует точки в одну строку YOLO Pose: cls + bbox + keypoints."""
    points_local = {name: dict(points.get(name) or {}) for name in POINT_NAMES}
    recovered_points = recover_missing_points(points_local, point_template)

    bbox = calculate_bbox(points_local)
    if bbox is None:
        return None, 0

    xc, yc, w, h = bbox
    keypoints_tokens: list[str] = []
    has_any_point = False

    for name in POINT_NAMES:
        point = points_local.get(name)
        if point and point.get("x_norm") is not None and point.get("y_norm") is not None:
            x_norm = float(point["x_norm"])
            y_norm = float(point["y_norm"])
            if point.get("imputed"):
                v = 1
            else:
                visible = point.get("visible", 0)
                v = 2 if visible else 0
            keypoints_tokens.extend([f"{x_norm:.6f}", f"{y_norm:.6f}", str(v)])
            has_any_point = True
        else:
            # YOLO ожидает фиксированное число keypoints: заполняем отсутствующие нулями.
            keypoints_tokens.extend(["0.000000", "0.000000", "0"])

    if not has_any_point:
        return None, 0

    line_tokens = ["0", f"{xc:.6f}", f"{yc:.6f}", f"{w:.6f}", f"{h:.6f}"] + keypoints_tokens
    return " ".join(line_tokens), recovered_points


def process_split(
    split_path: Path,
    output_dir: Path,
    point_template: dict[str, tuple[float, float]],
) -> tuple[int, int]:
    """Обрабатывает train или val split (только keypoints)."""

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
    recovered_points_total = 0

    for json_file in points_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                point_data = json.load(f)

            image_filename = point_data.get("image_filename", json_file.stem + ".png")
            points = point_data.get("points", {})

            pose_line, recovered_points = convert_points_to_yolo_pose_line(points, point_template)
            if pose_line is None:
                print(f"[WARN] Пропуск {json_file}: нет ключевых точек")
                continue
            recovered_points_total += recovered_points

            image_src = images_dir / image_filename
            if not image_src.exists():
                for ext in [".png", ".jpg", ".jpeg"]:
                    if (images_dir / (image_filename.rsplit(".", 1)[0] + ext)).exists():
                        image_src = images_dir / (
                            image_filename.rsplit(".", 1)[0] + ext
                        )
                        break

            txt_file = txt_output_dir / f"{json_file.stem}.txt"
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(pose_line + "\n")

            if image_src.exists() and image_src.is_file():
                img_dst = img_output_dir / image_src.name
                img_dst.write_bytes(image_src.read_bytes())

            processed += 1

        except Exception as e:
            print(f"[ERR] Ошибка при обработке {json_file}: {e}")
            continue

    return processed, recovered_points_total


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
    point_template = build_point_template(input_path)

    total_processed = 0
    total_recovered = 0

    for split_name in SPLIT_NAMES:
        split_path = input_path / split_name
        if not split_path.exists():
            continue

        print(f"[INFO] Обработка {split_name}:")
        count, recovered = process_split(split_path, output_path, point_template)
        total_processed += count
        total_recovered += recovered
        print(f"[INFO]   Обработано файлов: {count}")
        print(f"[INFO]   Восстановлено точек: {recovered}")

    return total_processed, total_recovered


def main():
    parser = argparse.ArgumentParser(description="Подготовка данных для YOLO Keypoint")
    parser.add_argument(
        "--input",
        type=str,
        default=str(get_data_path("export_project", "segpoint")),
        help="Путь к export_project/segpoint или export_project/segpoint_folds во внешнем data root",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(get_data_path("export_project", "yolo_keypoints")),
        help="Путь к выходной папке",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"[ERR] Входная папка не найдена: {input_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    print("[INFO] Подготовка данных для YOLO Keypoint")
    print(f"[INFO] Входная папка: {input_path}")
    print(f"[INFO] Выходная папка: {output_path}")
    print(f"[INFO] Ключевые точки: {POINT_NAMES}")
    print(f"[INFO] Восстановление пропусков: включено (минимум 2 известных точки)")

    datasets = discover_dataset_roots(input_path)
    if not datasets:
        print(f"[ERR] Не найден dataset root (ожидаются train/val или fold_*/train,val): {input_path}")
        return

    total_processed = 0
    total_recovered = 0

    for dataset_name, dataset_root in datasets:
        dataset_output = output_path / dataset_name if dataset_name else output_path
        dataset_output.mkdir(parents=True, exist_ok=True)
        if dataset_name:
            print(f"\n[INFO] === Обработка {dataset_name} ===")
            print(f"[INFO] Fold input: {dataset_root}")
            print(f"[INFO] Fold output: {dataset_output}")

        count, recovered = process_dataset_root(dataset_root, dataset_output)
        total_processed += count
        total_recovered += recovered

    print(f"\n[INFO] Всего обработано файлов: {total_processed}")
    print(f"[INFO] Всего восстановлено точек: {total_recovered}")
    print(f"[INFO] Структура готова для YOLO Keypoint:")
    if len(datasets) == 1 and not datasets[0][0]:
        print(f"      {output_path}/images/  - изображения")
        print(f"      {output_path}/labels/  - метки ключевых точек")
    else:
        print(f"      {output_path}/fold_XX/images/  - изображения")
        print(f"      {output_path}/fold_XX/labels/  - метки ключевых точек")


if __name__ == "__main__":
    main()
