#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python3 train_v4.py --config configs/ablation_v4/full_model.yaml --exp_name ablation_v4_full_model
