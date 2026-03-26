#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Восстанавливает папки img/ в Supervisely-экспорте из внешнего источника изображений."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from bamnet_paths import get_data_path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def detect_source_layout(source_root: Path) -> str:
    patient_dirs = [d for d in source_root.iterdir() if d.is_dir() and (d / "img").exists()]
    if patient_dirs:
        return "patient"

    direct_splits = [
        source_root / split / "images"
        for split in ("train", "val", "test")
        if (source_root / split / "images").exists()
    ]
    if direct_splits:
        return "split"

    fold_splits = list(source_root.glob("fold_*/*/images")) + list(
        source_root.glob("test_fold_*/*/images")
    )
    if fold_splits:
        return "fold"

    raise FileNotFoundError(
        f"Не удалось определить структуру источника изображений: {source_root}"
    )


def build_global_image_index(source_root: Path, layout: str) -> tuple[dict[str, list[Path]], int]:
    index: dict[str, list[Path]] = defaultdict(list)
    source_count = 0

    if layout == "patient":
        image_dirs = [d / "img" for d in source_root.iterdir() if d.is_dir() and (d / "img").exists()]
    elif layout == "split":
        image_dirs = [
            source_root / split / "images"
            for split in ("train", "val", "test")
            if (source_root / split / "images").exists()
        ]
    elif layout == "fold":
        image_dirs = sorted(source_root.glob("fold_*/*/images")) + sorted(
            source_root.glob("test_fold_*/*/images")
        )
    else:
        raise ValueError(f"Неизвестная структура источника: {layout}")

    for image_dir in image_dirs:
        for path in image_dir.iterdir():
            if not is_image_file(path):
                continue
            index[path.name].append(path)
            source_count += 1

    return dict(index), source_count


def resolve_source_image(
    patient_id: str,
    image_name: str,
    source_root: Path,
    layout: str,
    global_index: dict[str, list[Path]],
) -> tuple[Path | None, bool]:
    if layout == "patient":
        candidate = source_root / patient_id / "img" / image_name
        if candidate.exists():
            return candidate, False

    matches = global_index.get(image_name, [])
    if not matches:
        return None, False

    return matches[0], len(matches) > 1


def restore_images(
    supervisely_root: Path,
    source_root: Path,
    overwrite: bool,
    dry_run: bool,
) -> dict:
    layout = detect_source_layout(source_root)
    global_index, source_count = build_global_image_index(source_root, layout)

    patient_dirs = sorted(d for d in supervisely_root.iterdir() if d.is_dir())
    summary = Counter()
    missing_images: list[str] = []
    ambiguous_images: dict[str, list[str]] = {}
    extra_source_images: list[str] = []
    per_patient: dict[str, dict] = {}

    for patient_dir in patient_dirs:
        ann_dir = patient_dir / "ann"
        if not ann_dir.exists():
            continue

        patient_id = patient_dir.name
        dst_img_dir = patient_dir / "img"
        if not dry_run:
            dst_img_dir.mkdir(parents=True, exist_ok=True)

        patient_report = Counter()
        expected_names = set()

        ann_files = sorted(ann_dir.glob("*.json"))
        for ann_path in ann_files:
            image_name = ann_path.stem
            expected_names.add(image_name)
            summary["annotations_total"] += 1
            patient_report["annotations_total"] += 1

            source_image, ambiguous = resolve_source_image(
                patient_id=patient_id,
                image_name=image_name,
                source_root=source_root,
                layout=layout,
                global_index=global_index,
            )

            if source_image is None:
                summary["missing"] += 1
                patient_report["missing"] += 1
                missing_images.append(f"{patient_id}/{image_name}")
                continue

            if ambiguous:
                ambiguous_images[f"{patient_id}/{image_name}"] = [
                    str(path) for path in global_index.get(image_name, [])
                ]
                summary["ambiguous_matches"] += 1
                patient_report["ambiguous_matches"] += 1

            dst_image = dst_img_dir / image_name
            if dst_image.exists():
                if overwrite and not dry_run:
                    shutil.copy2(source_image, dst_image)
                    summary["overwritten"] += 1
                    patient_report["overwritten"] += 1
                else:
                    summary["already_present"] += 1
                    patient_report["already_present"] += 1
                continue

            if not dry_run:
                shutil.copy2(source_image, dst_image)

            summary["copied"] += 1
            patient_report["copied"] += 1

        if layout == "patient":
            src_img_dir = source_root / patient_id / "img"
            if src_img_dir.exists():
                for image_path in sorted(src_img_dir.iterdir()):
                    if not is_image_file(image_path):
                        continue
                    if image_path.name not in expected_names:
                        extra_source_images.append(f"{patient_id}/{image_path.name}")
                        patient_report["extra_source_images"] += 1

        per_patient[patient_id] = dict(patient_report)

    summary["patients_total"] = len(per_patient)
    summary["source_images_total"] = source_count
    summary["unique_source_names"] = len(global_index)

    return {
        "supervisely_root": str(supervisely_root.resolve()),
        "source_root": str(source_root.resolve()),
        "source_layout": layout,
        "dry_run": dry_run,
        "overwrite": overwrite,
        "summary": dict(summary),
        "missing_images": missing_images,
        "ambiguous_images": ambiguous_images,
        "extra_source_images": extra_source_images,
        "per_patient": per_patient,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Восстановление папок img/ в Supervisely-экспорте."
    )
    parser.add_argument(
        "--supervisely-root",
        type=str,
        default=str(get_data_path("export_project", "segmentation_point")),
        help="Путь к корню Supervisely-экспорта без изображений",
    )
    parser.add_argument(
        "--source-root",
        type=str,
        default=str(get_data_path("segmentation_point(v2)")),
        help="Путь к источнику изображений",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="",
        help="Куда сохранить JSON-отчёт. По умолчанию рядом с Supervisely root",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Перезаписывать уже существующие изображения",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Только собрать отчёт, без копирования файлов",
    )
    args = parser.parse_args()

    supervisely_root = Path(args.supervisely_root)
    source_root = Path(args.source_root)

    if not supervisely_root.exists():
        raise FileNotFoundError(f"Не найден Supervisely root: {supervisely_root}")
    if not source_root.exists():
        raise FileNotFoundError(f"Не найден source root: {source_root}")

    report_path = (
        Path(args.report)
        if args.report
        else supervisely_root / "image_restore_report.json"
    )

    report = restore_images(
        supervisely_root=supervisely_root,
        source_root=source_root,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    summary = report["summary"]
    print(f"[INFO] source layout: {report['source_layout']}")
    print(f"[INFO] annotations total: {summary.get('annotations_total', 0)}")
    print(f"[INFO] copied: {summary.get('copied', 0)}")
    print(f"[INFO] already present: {summary.get('already_present', 0)}")
    print(f"[INFO] overwritten: {summary.get('overwritten', 0)}")
    print(f"[INFO] missing: {summary.get('missing', 0)}")
    print(f"[INFO] ambiguous matches: {summary.get('ambiguous_matches', 0)}")
    print(f"[INFO] source images total: {summary.get('source_images_total', 0)}")
    print(f"[INFO] unique source names: {summary.get('unique_source_names', 0)}")
    print(f"[INFO] report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
