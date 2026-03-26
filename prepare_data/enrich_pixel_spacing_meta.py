#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Переносит pixel_spacing_row_mm из image_mapping.csv в img_info/*.json."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from bamnet_paths import get_data_path

DEFAULT_DATASET_ROOT = get_data_path("export_project", "segmentation_point")
DEFAULT_MAPPING_CSV = Path("dataset_metadata/segmentation_point/image_mapping.csv")
DEFAULT_REPORT_NAME = "pixel_spacing_row_mm_report.json"


@dataclass(frozen=True, order=True)
class MappingKey:
    seg_patient: int
    image_name: str


def key_to_json(key: MappingKey) -> dict:
    return {"seg_patient": key.seg_patient, "image_name": key.image_name}


def parse_pixel_spacing(raw_value: str, *, context: str) -> float:
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Некорректный pixel_spacing_row_mm ({context}): {raw_value!r}") from exc

    if not math.isfinite(value):
        raise ValueError(f"pixel_spacing_row_mm должен быть конечным числом ({context}): {raw_value!r}")

    return value


def load_mapping(mapping_csv: Path):
    mapping: dict[MappingKey, float] = {}
    row_count = 0
    row_count_by_key: Counter[MappingKey] = Counter()
    duplicate_same_value_keys: list[dict] = []
    conflicting_duplicate_keys: list[dict] = []
    values_by_key: defaultdict[MappingKey, set[float]] = defaultdict(set)

    with open(mapping_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required_columns = {"seg_patient", "image_name", "pixel_spacing_row_mm"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"В {mapping_csv} отсутствуют обязательные колонки: {sorted(missing_columns)}"
            )

        for row_number, row in enumerate(reader, start=2):
            row_count += 1
            try:
                seg_patient = int(row["seg_patient"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Некорректный seg_patient в строке {row_number}: {row['seg_patient']!r}"
                ) from exc

            image_name = (row.get("image_name") or "").strip()
            if not image_name:
                raise ValueError(f"Пустой image_name в строке {row_number}")

            key = MappingKey(seg_patient=seg_patient, image_name=image_name)
            value = parse_pixel_spacing(
                row.get("pixel_spacing_row_mm", ""),
                context=f"CSV row {row_number}, key={key}",
            )

            row_count_by_key[key] += 1
            values_by_key[key].add(value)

            if key not in mapping:
                mapping[key] = value

    for key, count in row_count_by_key.items():
        values = sorted(values_by_key[key])
        if count > 1 and len(values) == 1:
            duplicate_same_value_keys.append(
                {
                    **key_to_json(key),
                    "occurrences": count,
                    "pixel_spacing_row_mm": values[0],
                }
            )
        elif len(values) > 1:
            conflicting_duplicate_keys.append(
                {
                    **key_to_json(key),
                    "occurrences": count,
                    "pixel_spacing_row_mm_values": values,
                }
            )

    duplicate_same_value_keys.sort(key=lambda item: (item["seg_patient"], item["image_name"]))
    conflicting_duplicate_keys.sort(key=lambda item: (item["seg_patient"], item["image_name"]))
    return mapping, row_count, duplicate_same_value_keys, conflicting_duplicate_keys


def iter_img_info_files(dataset_root: Path):
    for patient_dir in sorted(d for d in dataset_root.iterdir() if d.is_dir()):
        img_info_dir = patient_dir / "img_info"
        if not img_info_dir.exists():
            continue
        patient_id = int(patient_dir.name)
        for img_info_path in sorted(img_info_dir.glob("*.json")):
            yield patient_id, img_info_path


def build_report(
    *,
    dataset_root: Path,
    mapping_csv: Path,
    dry_run: bool,
    mapping: dict[MappingKey, float],
    csv_row_count: int,
    duplicate_same_value_keys: list[dict],
    conflicting_duplicate_keys: list[dict],
    summary: Counter,
    missing_mappings: list[dict],
    unused_csv_rows: list[dict],
) -> dict:
    return {
        "dataset_root": str(dataset_root.resolve()),
        "mapping_csv": str(mapping_csv.resolve()),
        "dry_run": dry_run,
        "summary": {
            "csv_rows_total": csv_row_count,
            "csv_unique_keys_total": len(mapping),
            "dataset_files_scanned": summary["dataset_files_scanned"],
            "matched_files": summary["matched_files"],
            "updated_files": summary["updated_files"],
            "already_matching_files": summary["already_matching_files"],
            "missing_mappings_count": len(missing_mappings),
            "unused_csv_rows_count": len(unused_csv_rows),
            "duplicate_same_value_keys_count": len(duplicate_same_value_keys),
            "conflicting_duplicate_keys_count": len(conflicting_duplicate_keys),
        },
        "missing_mappings": missing_mappings,
        "unused_csv_rows": unused_csv_rows,
        "duplicate_same_value_keys": duplicate_same_value_keys,
        "conflicting_duplicate_keys": conflicting_duplicate_keys,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Добавляет pixel_spacing_row_mm в meta каждого img_info/*.json."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(DEFAULT_DATASET_ROOT),
        help="Путь к публикационному Supervisely-проекту",
    )
    parser.add_argument(
        "--mapping-csv",
        type=str,
        default=str(DEFAULT_MAPPING_CSV),
        help="Путь к image_mapping.csv",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Только проверить покрытие и сформировать отчёт, без записи файлов",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="",
        help="Путь к JSON-отчёту. По умолчанию <dataset-root>/pixel_spacing_row_mm_report.json",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    mapping_csv = Path(args.mapping_csv)
    report_path = (
        Path(args.report)
        if args.report
        else dataset_root / DEFAULT_REPORT_NAME
    )

    if not dataset_root.exists():
        raise FileNotFoundError(f"Не найден dataset root: {dataset_root}")
    if not mapping_csv.exists():
        raise FileNotFoundError(f"Не найден CSV mapping: {mapping_csv}")

    mapping, csv_row_count, duplicate_same_value_keys, conflicting_duplicate_keys = load_mapping(
        mapping_csv
    )

    summary: Counter = Counter()
    missing_mappings: list[dict] = []
    used_keys: set[MappingKey] = set()

    if conflicting_duplicate_keys:
        report = build_report(
            dataset_root=dataset_root,
            mapping_csv=mapping_csv,
            dry_run=args.dry_run,
            mapping=mapping,
            csv_row_count=csv_row_count,
            duplicate_same_value_keys=duplicate_same_value_keys,
            conflicting_duplicate_keys=conflicting_duplicate_keys,
            summary=summary,
            missing_mappings=[],
            unused_csv_rows=[],
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[ERR] Найдены конфликтующие дубликаты в CSV: {len(conflicting_duplicate_keys)}")
        print(f"[INFO] report: {report_path.resolve()}")
        return 1

    for patient_id, img_info_path in iter_img_info_files(dataset_root):
        summary["dataset_files_scanned"] += 1
        key = MappingKey(seg_patient=patient_id, image_name=img_info_path.stem)
        pixel_spacing_row_mm = mapping.get(key)

        if pixel_spacing_row_mm is None:
            missing_mappings.append(key_to_json(key))
            continue

        summary["matched_files"] += 1
        used_keys.add(key)

        with open(img_info_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        meta = payload.get("meta")
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise ValueError(f"Поле meta должно быть объектом: {img_info_path}")

        existing_value = meta.get("pixel_spacing_row_mm")
        if existing_value is None:
            needs_update = True
        else:
            try:
                existing_value = parse_pixel_spacing(
                    existing_value,
                    context=f"img_info {img_info_path}",
                )
            except ValueError:
                needs_update = True
            else:
                needs_update = not math.isclose(
                    existing_value, pixel_spacing_row_mm, rel_tol=0.0, abs_tol=1e-12
                )

        if needs_update:
            meta["pixel_spacing_row_mm"] = pixel_spacing_row_mm
            payload["meta"] = meta
            summary["updated_files"] += 1
            if not args.dry_run:
                with open(img_info_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
        else:
            summary["already_matching_files"] += 1

    unused_csv_rows = [
        key_to_json(key) | {"pixel_spacing_row_mm": mapping[key]}
        for key in sorted(set(mapping.keys()) - used_keys)
    ]
    missing_mappings.sort(key=lambda item: (item["seg_patient"], item["image_name"]))

    report = build_report(
        dataset_root=dataset_root,
        mapping_csv=mapping_csv,
        dry_run=args.dry_run,
        mapping=mapping,
        csv_row_count=csv_row_count,
        duplicate_same_value_keys=duplicate_same_value_keys,
        conflicting_duplicate_keys=conflicting_duplicate_keys,
        summary=summary,
        missing_mappings=missing_mappings,
        unused_csv_rows=unused_csv_rows,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[INFO] dataset files scanned: {summary['dataset_files_scanned']}")
    print(f"[INFO] matched files: {summary['matched_files']}")
    print(f"[INFO] updated files: {summary['updated_files']}")
    print(f"[INFO] already matching files: {summary['already_matching_files']}")
    print(f"[INFO] missing mappings: {len(missing_mappings)}")
    print(f"[INFO] unused csv rows: {len(unused_csv_rows)}")
    print(f"[INFO] duplicate same-value keys: {len(duplicate_same_value_keys)}")
    print(f"[INFO] conflicting duplicate keys: {len(conflicting_duplicate_keys)}")
    print(f"[INFO] report: {report_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
