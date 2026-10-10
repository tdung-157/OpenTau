#!/bin/bash
#SBATCH --job-name=opentau_pi05_astribot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#
# Resource asks are deliberately modest so the job can BACKFILL on a busy queue. The openpi
# script this was modelled on pins --nodelist=worker-0 and asks for 30 CPUs / 240 GB / 3 days;
# that combination sits in PD (Priority) behind every higher-priority job because the scheduler
# can never prove it fits in a gap. Notes on each:
#   no --nodelist : let Slurm place the job. The pin only narrows what can schedule you.
#   --gpus=1      : the script derives the batch split from SLURM_GPUS_ON_NODE, so 1 or 2 both
#                   give the same GLOBAL_BATCH=32. Override with `sbatch --gpus=2 ...`.
#   --cpus-per-task=12 : dataloader workers only. Some nodes report just 16 CPUs total, so 30
#                   is unsatisfiable there. 12 -> 10 workers/rank, enough for video decode.
#   --mem=96G     : a total, not per-CPU. --mem-per-cpu=8192 x 30 silently asked for 240 GB.
#   --time=1-00:00:00 : save_freq=5000 writes checkpoints throughout, so a shorter window is
#                   safe -- requeue with `sbatch configs/astribot/sbatch_train.sh --resume=true`
#                   and it picks up from the last checkpoint. A 3-day ask cannot backfill.
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
# Env knobs:
#   CONFIG        path to the training JSON        (default: configs/astribot/pi05_astribot_subtask.json)
#   GLOBAL_BATCH  batch across ALL ranks           (default: 32, matching openpi's batch_size=32)
#   MICRO_BATCH   per-rank per-step batch          (default: 8; lower this first on OOM)
# Extra args pass through to train.py, e.g. --steps=60000 --resume=true

set -euo pipefail

# Slurm copies this script to the node's spool dir, so BASH_SOURCE cannot locate the checkout.
cd "${SLURM_SUBMIT_DIR:-/mnt/data/dungnt232_1/repos/OpenTau}"

CONFIG="${CONFIG:-configs/astribot/pi05_astribot_subtask.json}"

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
echo "[train] config=$CONFIG gpus=$NGPU workers/rank=$NUM_WORKERS port=$PORT"
echo "[train] global_batch=$GLOBAL_BATCH = ${NGPU} rank(s) x ${MICRO_BATCH} micro x ${ACCUM} accum"
echo "[train] HF_OPENTAU_HOME=$HF_OPENTAU_HOME HF_HOME=$HF_HOME"

exec uv run -- accelerate launch \
  --num_machines 1 \
  --num_processes "$NGPU" \
  --mixed_precision bf16 \
  --dynamo_backend no \
  --main_process_port "$PORT" \
  src/opentau/scripts/train.py \
  --config_path="$CONFIG" \
  --batch_size="$RANK_BATCH" \
  --dataloader_batch_size="$MICRO_BATCH" \
  --gradient_accumulation_steps="$ACCUM" \
  --num_workers="$NUM_WORKERS" \
  "$@"
