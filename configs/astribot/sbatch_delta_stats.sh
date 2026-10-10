#!/bin/bash
#SBATCH --job-name=opentau_delta_stats
#SBATCH --nodelist=worker-0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-cpu=8192
#SBATCH --time=04:00:00
#SBATCH --output=logs/delta_stats_%j.out
#SBATCH --error=logs/delta_stats_%j.err
#
# Precompute the delta-action normalization statistics (the OpenTau analogue of openpi's
# sbatch_norm_stats.sh) and report the dataloader ceiling.
#
# NOTE: no `#SBATCH --gpus` line, on purpose. This job is CPU-only -- profile_dataloader.py
# builds no model and falls back to torch.device("cpu") when CUDA is absent -- so it queues
# without waiting behind GPU jobs.
#
# WHAT NEEDS COMPUTING, AND WHAT DOES NOT
#   Absolute state/action stats: already on disk in meta/episodes_stats.jsonl (v2.1), so there
#   is no openpi-style mandatory norm_stats.json step. Image stats are the ImageNet constants
#   (use_imagenet_stats, default true). But `use_delta_joint_actions: true` changes the action
#   distribution the policy trains against -- a joint's displacement over a 50-step chunk is far
#   smaller than its absolute range -- so the absolute stats would misscale every mapped dim by
#   2-5x. Those delta stats are computed on the fly and cached; this job just pays that cost up
#   front instead of inside the training allocation.
#
#   The pass is O(frames x chunk_size) = 264,874 x 50 = 13.2M rows, but reads only the numeric
#   state/action parquet columns -- never video. MEASURED: 4.5 s for the exact, uncapped pass on
#   10 workers. So this job is NOT here to save compute -- the real reason to run it is that it
#   exercises the entire data path (config parse, delta map, video decode, collate) on the
#   cluster in under a minute, before a 3-day allocation is committed to it.
#
#   Cache: $HF_OPENTAU_HOME/<repo_id>/meta/delta_action_stats/<key>.json, keyed by everything
#   that changes the numbers (delta map, chunk length, column indices, episode selection,
#   resample strategy). Editing any of those correctly recomputes rather than serving stale
#   stats. The dataset root must be WRITABLE by this job.
#
#   mkdir -p logs
#   sbatch configs/astribot/sbatch_delta_stats.sh
#
# Env knobs:
#   CONFIG           training JSON to read the mixture from
#   PROFILE_BATCHES  measured batches after a 20-batch warmup (default 300).
#                    Set PROFILE_BATCHES=1 to warm the cache and stop -- the stats are written
#                    while the mixture is built, before a single batch is fetched.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/mnt/data/dungnt232_1/repos/OpenTau}"

CONFIG="${CONFIG:-configs/astribot/pi05_astribot_subtask.json}"

export HF_OPENTAU_HOME="${HF_OPENTAU_HOME:-/mnt/data/dungnt232_1/data}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PROFILE_BATCHES="${PROFILE_BATCHES:-300}"

# `num_workers` is doing double duty here, which is why it is set this high: the delta-stats
# ProcessPoolExecutor is sized from it (factory.py) and it is also the DataLoader worker count.
# The stats pool fans out over the 186 per-episode parquet files, so it scales to ~the core
# count; leave 2 cores for the parent process.
CPUS="${SLURM_CPUS_PER_TASK:-8}"
NUM_WORKERS=$(( CPUS - 2 ))
(( NUM_WORKERS < 1 )) && NUM_WORKERS=1

echo "[delta_stats] $(hostname) $(date -Is) cwd=$PWD"
echo "[delta_stats] config=$CONFIG workers=$NUM_WORKERS batches=$PROFILE_BATCHES"
echo "[delta_stats] HF_OPENTAU_HOME=$HF_OPENTAU_HOME"

# Plain python, not `accelerate launch`: the compute is single-process by design (only the main
# rank computes; other ranks would just poll the filesystem for the file it writes), and
# profile_dataloader.py creates its own optional Accelerator. `val_freq=0` skips building the
# validation split this job does not use.
exec uv run python src/opentau/scripts/profile_dataloader.py \
  --config_path="$CONFIG" \
  --num_workers="$NUM_WORKERS" \
  --val_freq=0 \
  "$@"
