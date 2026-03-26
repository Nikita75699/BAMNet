#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python3 ../train.py --config configs/ablation_v4/full_model.yaml --exp_name ablation_v4_full_model
python3 ../train.py --config configs/ablation_v4/no_position_attention.yaml --exp_name ablation_v4_no_position_attention
python3 ../train.py --config configs/ablation_v4/no_coordinate_attention.yaml --exp_name ablation_v4_no_coordinate_attention
python3 ../train.py --config configs/ablation_v4/no_fusion.yaml --exp_name ablation_v4_no_fusion
python3 ../train.py --config configs/ablation_v4/no_boundary_guidance.yaml --exp_name ablation_v4_no_boundary_guidance
python3 ../train.py --config configs/ablation_v4/no_boundary_loss.yaml --exp_name ablation_v4_no_boundary_loss
python3 ../train.py --config configs/ablation_v4/beta_fixed_8.yaml --exp_name ablation_v4_beta_fixed_8
python3 ../train.py --config configs/ablation_v4/beta_schedule_4_8.yaml --exp_name ablation_v4_beta_schedule_4_8
