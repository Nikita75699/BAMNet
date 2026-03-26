# Dataset Preparation

Этот каталог содержит утилиты для подготовки датасета под публикацию на Zenodo и под обучение моделей BAMNet/YOLO.

## 1. Что публиковать на Zenodo

Рекомендуемый публикуемый артефакт: raw Supervisely-export с восстановленными изображениями.

Все тяжёлые данные и производные выгрузки предполагаются во внешнем корне данных:

```bash
export BAMNET_DATA_ROOT=/mnt/ssd4tb/data/BAMNet-data
```

Текущий raw export:

- `$BAMNET_DATA_ROOT/export_project/segmentation_point`
- внутри каждой папки пациента лежат `ann/` и `img_info/`, но изначально отсутствует `img/`

Полный источник изображений:

- `$BAMNET_DATA_ROOT/segmentation_point(v2)`
- у него совпадает структура `<patient>/img`, `<patient>/ann`, `<patient>/img_info`

Команда для восстановления `img/` в raw Supervisely-проекте:

```bash
python prepare_data/restore_supervisely_images.py \
  --supervisely-root "$BAMNET_DATA_ROOT/export_project/segmentation_point" \
  --source-root "$BAMNET_DATA_ROOT/segmentation_point(v2)"
```

Что делает скрипт:

- проходит по всем annotation-файлам в Supervisely-проекте;
- ищет одноимённое изображение во внешнем источнике `segmentation_point(v2)`;
- создаёт папки `img/` и копирует туда исходные `.png`;
- сохраняет отчёт в `image_restore_report.json`.

Перед архивированием для Zenodo стоит проверить:

```bash
python prepare_data/restore_supervisely_images.py \
  --supervisely-root "$BAMNET_DATA_ROOT/export_project/segmentation_point" \
  --source-root "$BAMNET_DATA_ROOT/segmentation_point(v2)" \
  --dry-run
```

Следующий шаг перед архивированием: перенести `pixel_spacing_row_mm` из `image_mapping.csv` в per-image metadata.

```bash
python prepare_data/enrich_pixel_spacing_meta.py \
  --dataset-root "$BAMNET_DATA_ROOT/export_project/segmentation_point" \
  --mapping-csv "dataset_metadata/segmentation_point/image_mapping.csv"
```

Скрипт записывает `pixel_spacing_row_mm` в `img_info/<image>.json -> meta.pixel_spacing_row_mm` и сохраняет отчёт в `pixel_spacing_row_mm_report.json`.

Для проверки без изменения файлов:

```bash
python prepare_data/enrich_pixel_spacing_meta.py \
  --dataset-root "$BAMNET_DATA_ROOT/export_project/segmentation_point" \
  --mapping-csv "dataset_metadata/segmentation_point/image_mapping.csv" \
  --dry-run
```

После этого можно архивировать саму папку `$BAMNET_DATA_ROOT/export_project/segmentation_point`.

Ожидаемая структура публикации:

```text
segmentation_point/
├── meta.json
├── README.md
├── pixel_spacing_row_mm_report.json
├── 001/
│   ├── ann/
│   ├── img/
│   └── img_info/
├── 002/
│   ├── ann/
│   ├── img/
│   └── img_info/
└── ...
```

## 2. Формат для BAMNet

BAMNet обучается не на raw Supervisely-структуре, а на промежуточном формате:

```text
<dataset_root>/
├── train/
│   ├── images/
│   ├── masks/
│   └── points/
└── val/
    ├── images/
    ├── masks/
    └── points/
```

Где:

- `images/` содержит исходные кадры;
- `masks/` содержит бинарную маску аортального корня;
- `points/` содержит JSON с четырьмя точками `AA1`, `AA2`, `STJ1`, `STJ2`;
- имена файлов совпадают между `images`, `masks`, `points`.

Конвертация из Supervisely-проекта:

```bash
python prepare_data/convert_data.py \
  --input "$BAMNET_DATA_ROOT/export_project/segmentation_point" \
  --output "$BAMNET_DATA_ROOT/export_project/segpoint" \
  --train-patients 83 \
  --val-patients 0
```

Если нужен отдельный holdout:

```bash
python prepare_data/convert_data.py \
  --input "$BAMNET_DATA_ROOT/export_project/segmentation_point" \
  --output "$BAMNET_DATA_ROOT/export_project/segpoint_holdout" \
  --train-patients 70 \
  --val-patients 13
```

Как работает `convert_data.py`:

- не требует установленного `supervisely` Python SDK;
- декодирует Supervisely bitmap в бинарную маску;
- извлекает landmark-точки;
- нормализует координаты точек;
- сохраняет один JSON на изображение в `points/`.

## 3. Конвертация в YOLO

Все YOLO-скрипты берут на вход уже подготовленный BAMNet-формат (`images/masks/points`).

### YOLO Object Detection

```bash
python prepare_data/prepare_yolo_data.py \
  --input "$BAMNET_DATA_ROOT/export_project/segpoint" \
  --output "$BAMNET_DATA_ROOT/export_project/yolo_data"
```

Что получается:

- 4 класса: `AA1`, `AA2`, `STJ1`, `STJ2`;
- каждая точка превращается в bbox размером `64x64 px`;
- выход: `images/train|val` и `labels/train|val`.

### YOLO Segmentation

```bash
python prepare_data/prepare_yolo_segmentation.py \
  --input "$BAMNET_DATA_ROOT/export_project/segpoint" \
  --output "$BAMNET_DATA_ROOT/export_project/yolo_segmentation"
```

Что получается:

- один класс сегментации;
- маска переводится в polygon-разметку YOLO;
- выход: `images/train|val` и `labels/train|val`.

### YOLO Keypoints / Pose

```bash
python prepare_data/prepare_yolo_keypoints.py \
  --input "$BAMNET_DATA_ROOT/export_project/segpoint" \
  --output "$BAMNET_DATA_ROOT/export_project/yolo_keypoints"
```

Что получается:

- один pose-класс;
- 4 keypoints в порядке `AA1`, `AA2`, `STJ1`, `STJ2`;
- если отсутствуют не все точки, скрипт пытается восстановить пропуски по similarity transform.

## 4. Что уже есть в репозитории

В репозитории уже присутствуют готовые варианты:

- `$BAMNET_DATA_ROOT/export_project/segpoint` — BAMNet-формат `train/val`;
- `$BAMNET_DATA_ROOT/export_project/segpoint_folds` — patient-level 5-fold splits;
- `dataset_metadata/segpoint_folds_manifests` — tracked-манифесты распределения пациентов и сэмплов;
- `$BAMNET_DATA_ROOT/export_project/yolo_data_folds` — detection folds;
- `$BAMNET_DATA_ROOT/export_project/yolo_segmentation_folds` — segmentation folds;
- `$BAMNET_DATA_ROOT/export_project/yolo_keypoints_folds` — pose/keypoints folds.

## 5. Краткое описание датасета для Zenodo

### English short description

This dataset contains fluoroscopic TAVI frames annotated for two tasks: binary segmentation of the aortic root and localization of four anatomical landmarks (`AA1`, `AA2`, `STJ1`, `STJ2`). The annotations are provided in Supervisely JSON format and are organized on a per-patient basis. The dataset can be used for multitask training of BAMNet as well as for derived YOLO object detection, segmentation, and keypoint benchmarks.

### Русское краткое описание

Датасет содержит флюороскопические кадры TAVI с разметкой двух задач: бинарная сегментация корня аорты и локализация четырёх анатомических ориентиров (`AA1`, `AA2`, `STJ1`, `STJ2`). Разметка сохранена в формате Supervisely JSON и организована по пациентам. Датасет может использоваться как для многозадачного обучения BAMNet, так и для производных постановок YOLO object detection, segmentation и keypoints.

### Suggested keywords

`TAVI`, `fluoroscopy`, `aortic root segmentation`, `landmark localization`, `medical image analysis`, `Supervisely`, `YOLO`, `BAMNet`
