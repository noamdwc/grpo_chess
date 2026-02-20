#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-$HOME/miniconda3/envs/grpo_chess/bin/python}"

echo "[1/3] Converting DeepMind bag shard to distillation shards"
"$PYTHON" -m src.distill.convert_deepmind_data \
  --config distill_9m.yaml \
  --num_shards 1

echo "[2/3] Distilling student checkpoint"
"$PYTHON" -m src.distill.distill \
  --config distill_9m.yaml \
  --no_wandb

echo "[3/3] GRPO post-training from distilled 9M warm-start"
"$PYTHON" -m src.train_self_play \
  --config grpo_9m_posttrain.yaml
