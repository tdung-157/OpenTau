#!/bin/bash
#SBATCH --job-name=opentau_pi05_serving_coffee
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=worker-1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=60
#SBATCH --mem-per-cpu=8192
#SBATCH --time=8-00:00:00
#SBATCH --output=logs/train_serving_coffee_%j.out
#SBATCH --error=logs/train_serving_coffee_%j.err
#
# Fine-tune pi05 on the Astribot serving-brewed-coffee dataset
# ($HF_OPENTAU_HOME/serving_brewed_coffee_lrb_opentau, LeRobot v3.0, 374 episodes) with the
# per-frame Flora subtask caption as the prompt and no response prediction.
#
# This file only sets the Slurm resources, job name and log names, then hands off to
# sbatch_train.sh. The batch split, HF token handling, env vars and the accelerate launch all
# live there, so a fix to one applies to both runs. The #SBATCH lines in sbatch_train.sh are
# ignored here: run by bash, they are plain comments.
#
# Usage (from the repo root on the cluster):
#   mkdir -p logs
#   sbatch configs/astribot/sbatch_train_serving_coffee.sh
#
# Same env knobs as sbatch_train.sh (CONFIG, GLOBAL_BATCH, MICRO_BATCH, HF_OPENTAU_HOME,
# HF_HOME), and extra args reach train.py:
#   MICRO_BATCH=2 sbatch configs/astribot/sbatch_train_serving_coffee.sh
#   sbatch configs/astribot/sbatch_train_serving_coffee.sh --resume=true
#   sbatch --nodelist= configs/astribot/sbatch_train_serving_coffee.sh   # unpin worker-1
#
# 4 GPUs: GLOBAL_BATCH=32 splits to 8/rank = MICRO_BATCH 4 x 2 accum steps (same effective
# batch as the 2-GPU runs). Workers/rank = (cpus - 2*gpus) / gpus = (60 - 8) / 4 = 13.
# Memory: --mem-per-cpu=8192 x 60 CPUs = 480 GB total, which worker-1 must have free.

set -euo pipefail

# Slurm runs a spool copy of this script, so locate the checkout via the submit dir.
cd "${SLURM_SUBMIT_DIR:-/mnt/data/dungnt232_1/repos/OpenTau}"

export CONFIG="${CONFIG:-configs/astribot/pi05_astribot_serving_coffee_subtask_prompt.json}"

exec bash configs/astribot/sbatch_train.sh "$@"
