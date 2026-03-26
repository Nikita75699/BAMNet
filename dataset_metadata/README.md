# Dataset Metadata

Этот каталог содержит небольшие tracked-метаданные, нужные для воспроизводимости, но не сами датасеты.

Содержимое:

- `segmentation_point/image_mapping.csv` — соответствие кадров и DICOM-источников, включая `pixel_spacing_row_mm`.
- `segpoint_folds/folds_meta.json` — сводная metadata по разбиению на фолды.
- `segpoint_folds_manifests/` — patient/sample manifests для patient-level split.

Тяжёлые данные, raw export и производные датасеты должны жить во внешнем `BAMNET_DATA_ROOT`, а не в git.
