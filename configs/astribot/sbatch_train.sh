#!/bin/bash
#SBATCH --job-name=opentau_pi05_astribot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-cpu=8192
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#
# Resource asks are deliberately modest so the job can BACKFILL on a busy queue. The openpi
# script this was modelled on pins --nodelist=worker-0 and asks for 30 CPUs / 240 GB / 3 days;
# that combination sits in PD (Priority) behind every higher-priority job because the scheduler
# can never prove it fits in a gap. Notes on each:
#   no --nodelist : let Slurm place the job. The pin only narrows what can schedule you.
#   --gpus=2      : the script derives the batch split from SLURM_GPUS_ON_NODE, so 1 or 2 both
#                   give the same GLOBAL_BATCH=32 (2 GPUs: 16/rank = 8 micro x 2 accum, so
#                   each step is ~2x faster, same LR schedule). Override with `sbatch --gpus=1 ...`.
#                   Resuming the 1-GPU checkpoint on 2 is supported: train.py re-seeds the new
#                   rank (reseed_new_ranks_on_resume); its "global batch changed by 2.000x"
#                   warning assumes a fixed per-rank batch and does not apply here.
#   --cpus-per-task=16 : dataloader workers only. Some nodes report just 16 CPUs total, so 30
#                   is unsatisfiable there. 16 -> 6 workers/rank on 2 GPUs.
#   --mem=128G    : a total, not per-CPU. --mem-per-cpu=8192 x 30 silently asked for 240 GB.
#                   Two trainer processes each stage pi05 on CPU at load, hence above 96G.
#   --time=1-00:00:00 : save_freq=5000 writes checkpoints throughout, so a shorter window is
#                   safe -- the script auto-resumes from the newest checkpoint (see below), so
#                   just resubmit it. A 3-day ask cannot backfill.
#   --dependency=singleton : at most one job with this name runs at a time, so a chain of
#                   identical submissions runs back to back, each resuming where the last
#                   stopped (up to save_freq steps are redone after a timeout).
#
# Fine-tune pi05 on the Astribot iced-coffee dataset (prompt = "Make an iced coffee",
# per-frame subtask supervised as `response`).
#
# Run `mkdir -p logs` once before the first submit -- Slurm opens the --output file BEFORE
# this script runs, so a missing logs/ dir fails the job at launch with no log to explain it.
#
# Warm the delta-action stats cache first (minutes, not hours -- see sbatch_delta_stats.sh),
# then submit:
#
#   mkdir -p logs
#   sbatch configs/astribot/sbatch_delta_stats.sh
#   sbatch configs/astribot/sbatch_train.sh
#
# Continuing past the first 30k-step run: the script resumes from the newest complete
# checkpoint under OUTPUT_DIR and trains until TOTAL_STEPS (an absolute step, not "N more", so
# every requeue targets the same end). 500k steps does not fit in one --time window, so queue a
# chain; jobs left over after the target is reached exit immediately:
#
#   for i in $(seq 15); do sbatch configs/astribot/sbatch_train.sh; done
#
# Env knobs:
#   CONFIG        path to the training JSON        (default: configs/astribot/pi05_astribot_subtask.json)
#                 used only for a fresh run; a resume reads the checkpoint's train_config.json
#   OUTPUT_DIR    run dir to resume from           (default: the config's output_dir)
#   TOTAL_STEPS   absolute step to train to        (default: 530000 = 30k first run + 500k)
#   GLOBAL_BATCH  batch across ALL ranks           (default: 32, matching openpi's batch_size=32)
#   MICRO_BATCH   per-rank per-step batch          (default: 8; lower this first on OOM)
# Extra args pass through to train.py, e.g. --log_freq=100

set -euo pipefail

# Slurm copies this script to the node's spool dir, so BASH_SOURCE cannot locate the checkout.
cd "${SLURM_SUBMIT_DIR:-/mnt/data/dungnt232_1/repos/OpenTau}"

CONFIG="${CONFIG:-configs/astribot/pi05_astribot_subtask.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["output_dir"])' "$CONFIG")}"
TOTAL_STEPS="${TOTAL_STEPS:-530000}"

# Find the newest COMPLETE checkpoint. train.py keeps no `last` symlink, and step dirs are
# zero-padded numbers. training_step.json is written after accelerate's save_state, so its
# presence means the save finished -- a job killed mid-save leaves a dir without it.
RESUME_DIR=""
RESUME_STEP=0
if [[ -d "$OUTPUT_DIR/checkpoints" ]]; then
  for d in "$OUTPUT_DIR"/checkpoints/*/; do
    d="${d%/}"
    name="${d##*/}"
    [[ "$name" =~ ^[0-9]+$ ]] || continue
    [[ -f "$d/training_step.json" && -f "$d/train_config.json" ]] || continue
    if (( 10#$name > RESUME_STEP )); then
      RESUME_STEP=$(( 10#$name ))
      RESUME_DIR="$d"
    fi
  done
fi

if (( RESUME_STEP >= TOTAL_STEPS )); then
  echo "[train] $RESUME_DIR is already at step $RESUME_STEP >= TOTAL_STEPS=$TOTAL_STEPS; nothing to do."
  exit 0
fi

if [[ -n "$RESUME_DIR" ]]; then
  # last_checkpoint_only=true deletes every other step dir at the next save, which would take
  # the finished first-run model with it. Hard-link a copy outside checkpoints/ once: instant,
  # no extra disk while both exist, and the files survive the prune.
  KEEP="$OUTPUT_DIR/kept/$(basename "$RESUME_DIR")"
  if (( RESUME_STEP == 30000 )) && [[ ! -e "$KEEP" ]]; then
    mkdir -p "$OUTPUT_DIR/kept"
    cp -al "$RESUME_DIR" "$KEEP" || cp -a "$RESUME_DIR" "$KEEP"
    echo "[train] preserved first-run checkpoint at $KEEP"
  fi
  # Resuming reads the checkpoint's own train_config.json (validate() takes the checkpoint dir
  # from config_path's parent); pointing --resume at the original JSON would not find it.
  RUN_ARGS=(--config_path="$RESUME_DIR/train_config.json" --resume=true)
else
  RUN_ARGS=(--config_path="$CONFIG")
fi

# `root` is deliberately absent from the dataset entry, so the repo_id resolves to
# $HF_OPENTAU_HOME/<repo_id> -- openpi's HF_LEROBOT_HOME pattern, so the same JSON works on
# the cluster and on a workstation by changing one env var.
export HF_OPENTAU_HOME="${HF_OPENTAU_HOME:-/mnt/data/dungnt232_1/data}"

# pi05_base is ~3B of weights fetched from huggingface.co at startup. Keep the cache on the
# shared volume rather than the default ~/.cache/huggingface, so it survives and is visible to
# whichever node the job lands on. If the compute nodes have no egress (they cannot reach
# api.wandb.ai, hence wandb.enable=false), pre-warm it from a node that does:
#     HF_HOME=/mnt/data/dungnt232_1/hf_cache hf download TensorAuto/pi05_base
# A private repo also needs HF_TOKEN exported, or the token file copied into $HF_HOME.
export HF_HOME="${HF_HOME:-/mnt/data/dungnt232_1/hf_cache}"

# Redirecting HF_HOME also moves where huggingface_hub looks for the auth token
# ($HF_HOME/token), orphaning the one `hf auth login` wrote to ~/.cache/huggingface. That
# matters because pi05 pulls the PaliGemma tokenizer from the GATED repo
# google/paligemma-3b-pt-224 (modeling_pi05.py), so an unauthenticated job dies with a 401
# before step 0. Carry the token across explicitly -- HF_TOKEN takes precedence over the file,
# and nothing secret is stored in this script.
if [[ -z "${HF_TOKEN:-}" && -r "$HOME/.cache/huggingface/token" ]]; then
  HF_TOKEN="$(< "$HOME/.cache/huggingface/token")"
  export HF_TOKEN
fi
[[ -n "${HF_TOKEN:-}" ]] || echo "[train] WARNING: no HF_TOKEN; gated google/paligemma-3b-pt-224 will 401 on a cache miss" >&2

# train.py logs metrics through logging/tqdm to stdout. Slurm redirects stdout to a file, so
# Python block-buffers it at 8 KB and the loss lines would only appear in large bursts. W&B is
# off in the config (the compute nodes cannot reach api.wandb.ai), so this log is the only
# place metrics land.
export PYTHONUNBUFFERED=1

# One thread per worker. Torch defaults to one thread PER CORE in every worker process, so the
# dataloader workers would otherwise fight over hundreds of threads. The GPUs do the compute;
# the workers only decode video and collate.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# `accelerate launch` is invoked directly rather than through the `opentau-train` console
# script, because that wrapper routes unknown flags to train.py -- so --num_processes could not
# reach accelerate. It goes through `uv run` because a Slurm batch job gets no shell profile and
# so no activated venv: a bare `accelerate` exits 127, command-not-found, before the job logs
# anything useful. Direct invocation also skips the allocator default that scripts/launch.py
# would have set; set it here. pi05 on an 80 GB card sits at the allocator's
# edge and OOMs on fragmentation without expandable segments (issue #264).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

NGPU="${SLURM_GPUS_ON_NODE:-1}"
CPUS="${SLURM_CPUS_PER_TASK:-8}"

# OpenTau runs with accelerate's `split_batches=False`, so `batch_size` in the config is
# PER RANK and the true global batch is batch_size x num_processes. openpi's batch_size=32 is
# a global batch sharded by FSDP, so divide here to land on the same effective batch.
GLOBAL_BATCH="${GLOBAL_BATCH:-32}"
if (( GLOBAL_BATCH % NGPU != 0 )); then
  echo "[train] ERROR: GLOBAL_BATCH=$GLOBAL_BATCH is not divisible by NGPU=$NGPU." >&2
  exit 1
fi
RANK_BATCH=$(( GLOBAL_BATCH / NGPU ))

MICRO_BATCH="${MICRO_BATCH:-8}"
(( MICRO_BATCH > RANK_BATCH )) && MICRO_BATCH="$RANK_BATCH"
if (( RANK_BATCH % MICRO_BATCH != 0 )); then
  echo "[train] ERROR: per-rank batch $RANK_BATCH is not divisible by MICRO_BATCH=$MICRO_BATCH." >&2
  exit 1
fi
ACCUM=$(( RANK_BATCH / MICRO_BATCH ))

# Workers are per rank, so budget the cores across ranks and leave ~2 per rank for the
# trainer process itself.
NUM_WORKERS=$(( (CPUS - 2 * NGPU) / NGPU ))
(( NUM_WORKERS < 1 )) && NUM_WORKERS=1

# Keep concurrent jobs on one node from colliding on the rendezvous port.
PORT=$(( 29500 + ${SLURM_JOB_ID:-0} % 1000 ))

echo "[train] $(hostname) $(date -Is) cwd=$PWD"
if [[ -n "$RESUME_DIR" ]]; then
  echo "[train] resuming from $RESUME_DIR (step $RESUME_STEP) -> $TOTAL_STEPS"
else
  echo "[train] fresh run from $CONFIG -> $TOTAL_STEPS"
fi
echo "[train] config=$CONFIG gpus=$NGPU workers/rank=$NUM_WORKERS port=$PORT"
echo "[train] global_batch=$GLOBAL_BATCH = ${NGPU} rank(s) x ${MICRO_BATCH} micro x ${ACCUM} accum"
echo "[train] HF_OPENTAU_HOME=$HF_OPENTAU_HOME HF_HOME=$HF_HOME"

# Explicit rather than relying on accelerate's "more than one GPU visible" auto-detect.
MULTI_GPU=()
(( NGPU > 1 )) && MULTI_GPU=(--multi_gpu)

exec uv run -- accelerate launch \
  --num_machines 1 \
  --num_processes "$NGPU" \
  "${MULTI_GPU[@]}" \
  --mixed_precision bf16 \
  --dynamo_backend no \
  --main_process_port "$PORT" \
  src/opentau/scripts/train.py \
  "${RUN_ARGS[@]}" \
  --steps="$TOTAL_STEPS" \
  --batch_size="$RANK_BATCH" \
  --dataloader_batch_size="$MICRO_BATCH" \
  --gradient_accumulation_steps="$ACCUM" \
  --num_workers="$NUM_WORKERS" \
  "$@"
