# BAMNet

`BAMNet` is a multitask deep learning project for fluoroscopic guidance during transcatheter aortic valve implantation (TAVI). The model jointly solves two tasks in one forward pass:

- aortic root segmentation;
- localization of four anatomical landmarks: `AA1`, `AA2`, `STJ1`, `STJ2`.

The current training pipeline is implemented in PyTorch Lightning and is focused on supervised training, validation, checkpointing, and qualitative visualization of predictions.

## Project Overview

The core model is implemented in [model_backbone_coords.py](./model_backbone_coords.py). In the current repository version, BAMNet combines:

- an `EfficientNet-V2` encoder;
- a segmentation-oriented decoder inspired by `MA-Net` / `U-Net`;
- decoder-side global attention;
- a landmark head with `Coordinate Attention`;
- boundary-aware supervision;
- offset-based landmark refinement;
- joint optimization of segmentation and landmark losses.

The main training entry point is [train.py](./train.py). Dataset loading and augmentation are implemented in [data.py](./data.py). The default experiment configuration is stored in [config.yaml](./config.yaml).

## Repository Structure

- [train.py](./train.py) — main training script.
- [data.py](./data.py) — dataset, augmentations, datamodule, debug visualizations.
- [model_backbone_coords.py](./model_backbone_coords.py) — BAMNet architecture and Lightning module.
- [config.yaml](./config.yaml) — default configuration.
- [dataset_metadata/](./dataset_metadata/) — small tracked metadata required for dataset reproducibility.
- [prepare_data/](./prepare_data/) — dataset preparation, publication, and YOLO conversion utilities.
- [publication/](./publication/) — manuscript, figures, and LaTeX sources for the paper.
- [Swin-Unet/](./Swin-Unet/) — separate baseline code kept in the repository.

## Environment Setup

The repository does not currently include a pinned root `requirements.txt`, so installation is done from the imports used by the training pipeline.

Recommended environment:

- Python `3.10+`
- PyTorch with CUDA support if training on GPU

Example setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install the PyTorch build appropriate for your system first.
# Example for CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install pytorch-lightning albumentations opencv-python matplotlib pyyaml tensorboard numpy
```

If you train on CPU only, install the CPU build of PyTorch instead of the CUDA wheel.

## Repository Hygiene

This repository is intended to stay source-only by default.

- Commit code, configs, docs, and small metadata needed for reproducibility.
- Keep large datasets, prepared exports, YOLO outputs, and other generated artifacts outside the repo.
- Set `BAMNET_DATA_ROOT` to the external data root. If it is unset, the code falls back to `/mnt/ssd4tb/data/BAMNet-data`.

Example:

```bash
export BAMNET_DATA_ROOT=/mnt/ssd4tb/data/BAMNet-data
```

Expected external layout:

```text
$BAMNET_DATA_ROOT/
├── export_project/
├── segmentation_point(v2)/
├── runs/
└── ablation_runs/
```

## Dataset Layout

The training code expects a fold directory pointed to by `data_path` in [config.yaml](./config.yaml). Inside that directory, the following structure is expected:

```text
<data_path>/
├── train/
│   ├── images/
│   ├── masks/
│   └── points/
└── val/
    ├── images/
    ├── masks/
    └── points/
```

Expected file conventions:

- images: `*.png`, `*.jpg`, `*.jpeg`
- masks: `*.png`
- points: `*.json`
- filenames must share the same basename across `images/`, `masks/`, and `points/`

Example:

```text
train/images/frame_001.png
train/masks/frame_001.png
train/points/frame_001.json
```

Point annotations support either absolute or normalized coordinates. A typical JSON file can look like this:

```json
{
  "image_size": { "width": 1000, "height": 1000 },
  "points": {
    "AA1":  { "x": 120, "y": 540, "visible": 1 },
    "AA2":  { "x": 410, "y": 548, "visible": 1 },
    "STJ1": { "x": 170, "y": 310, "visible": 1 },
    "STJ2": { "x": 365, "y": 300, "visible": 1 }
  }
}
```

The loader also accepts normalized coordinates via `x_norm` / `y_norm`.

## Configuration

The default configuration is in [config.yaml](./config.yaml). The most important fields are:

- `data_path` — path to the current fold directory;
- `encoder_name` — e.g. `efficientnet_v2_m`;
- `img_size` — input image size;
- `batch_size` — batch size;
- `num_points` and `point_names` — landmark definition;
- `logging.save_dir` and `logging.experiment_name` — output location;
- `trainer.epochs`, `trainer.devices`, `trainer.precision` — Lightning trainer settings.

Default landmark order:

```yaml
point_names: ["AA1", "AA2", "STJ1", "STJ2"]
```

## How to Run Training

1. Export `BAMNET_DATA_ROOT` if you use a custom external data root.
2. Edit [config.yaml](./config.yaml) if you want a fold different from the default one.
3. Activate the environment.
4. Start training:

```bash
python train.py --config config.yaml
```

Example with parameter overrides:

```bash
python train.py \
  --config config.yaml \
  --lr 1e-4 \
  --w_pts 1.0 \
  --w_bnd 0.1 \
  --exp_name bamnet_ablation
```

The trainer automatically uses GPU when `torch.cuda.is_available()` is true; otherwise it falls back to CPU.

## Training Outputs

By default, training artifacts are written to:

```text
$BAMNET_DATA_ROOT/runs/<experiment_name>/<architecture>/<version>/
```

Typical contents:

- TensorBoard logs
- CSV logs
- checkpoints
- `best.pt` symlink or copy to the best checkpoint
- `val_vis/` — qualitative validation predictions
- `debug_inspect/` — saved sample previews from train/val splits

To inspect logs:

```bash
tensorboard --logdir runs
```

## Current Scope

This repository is currently centered on training and validation. There is no separate polished inference CLI yet. If you need deployment or standalone prediction scripts, they should be added on top of the existing Lightning checkpoint workflow.

## Results Reported in the Publication

The manuscript source is available in [publication/sn-article-template/bamnet-article.tex](./publication/sn-article-template/bamnet-article.tex).

Briefly, the paper reports the following BAMNet results:

- independent validation set:
  - Dice: `0.887`
  - IoU: `0.805`
  - mean landmark localization error: `3.16 mm` (`11.95 px`)
  - real-time throughput: `43.9 FPS`
- patient-level 5-fold cross-validation:
  - Dice: `0.897 ± 0.024`
  - IoU: `0.826 ± 0.033`
  - Surface Dice@4 mm: `0.807 ± 0.058`
  - median landmark error: `8.06 ± 0.75 px`
  - mean landmark error: `11.13 ± 0.76 px`

These results support the use of the model as a foundation for contrast-sparing visual guidance and future robot-assisted TAVI workflows.

## Notes

- The repository contains code and publication materials under active development.
- Paths in [config.yaml](./config.yaml) and helper scripts resolve through `BAMNET_DATA_ROOT`.
- The `Swin-Unet/` subdirectory is a separate baseline implementation and is not required for BAMNet training.
