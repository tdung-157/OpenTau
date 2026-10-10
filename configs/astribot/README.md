# Fine-tuning π₀.₅ on the Astribot iced-coffee dataset (`configs/astribot`)

Per-embodiment config folder, mirroring the layout of `custom_astribot/` in the sibling
`openpi` checkout (one folder per embodiment: `custom_astribot`, `custom_motion2`, ...). Same
robot, same 186-episode dataset, same delta-joint/absolute-gripper action space — but OpenTau is
config-driven (draccus JSON + `accelerate launch`), so there are no Python wrapper modules to
register: the JSON *is* the config.

| Property | Value |
| --- | --- |
| Robot | `Astribot`, bimanual, 30 fps |
| Cameras | `cam_head` → `camera0`, `cam_left_wrist` → `camera1`, `cam_right_wrist` → `camera2` (480×640 RGB, letterboxed to 224×224) |
| State / action | 16-dim `[7 left-arm joints, 7 right-arm joints, left_gripper, right_gripper]` — grippers **trail** the arms |
| Action space | delta joints (dims 0–13) w.r.t. the chunk-start state, **absolute** grippers (dims 14–15) |
| Episodes / frames | 186 / 264,874 |
| Prompt | `Make an iced coffee` (constant) |
| Response | the per-frame subtask string, CE-supervised |

## What differs from the openpi setup

openpi ships two variants of this run and makes you pick one: `pi05_astribot_full` conditions
every frame on the fixed `"Make an iced coffee"` prompt and ignores the subtask labels, while
`pi05_astribot_anno` puts the per-frame subtask *in the prompt* and drops the episode-level
goal. OpenTau's pi05 carries a separate `response` channel, so this config does both at once —
the fixed goal as the prompt, the subtask as a CE-supervised response the model learns to emit
itself. That is π₀.₅'s own knowledge-insulation split: the action MSE trains only the action
expert, and the VLM backbone is trained by the subtask + discrete-action CE losses.

Consequence for deployment: the served policy takes the **constant** prompt and predicts its
own subtask, so `tasks.yaml`-style live prompt stepping is not needed. The prompt string must
still match what `deploy_ros2.py` sends (`"Make an iced coffee"`) — it is baked into
`meta/tasks.jsonl` at dataset-prep time, so changing one means regenerating the other.

## Dataset

Built from the labelled source by [`prepare_astri_for_pi05.py`](prepare_astri_for_pi05.py),
which transposes the per-frame `task_index` subtask labels into pi05's prompt/response split
(the analogue of openpi's `convert_v3_to_v21.py`, which lives in its embodiment folder too):

```bash
uv run python configs/astribot/prepare_astri_for_pi05.py <source_root> <dest_root> \
    --task "Make an iced coffee"
```

`pyarrow` is its only import — it does not touch OpenTau, so on a login node with no training
env you can run it standalone with
`uv run --no-project --with pyarrow python configs/astribot/prepare_astri_for_pi05.py ...`.

It writes a `response` column (the subtask active at that frame), collapses `task_index` to 0
with a single-entry `tasks.jsonl`, and records each episode's subtask start frames as
`segments` in `episodes.jsonl`. State, action and video bytes are untouched (videos are
hardlinked).

### Getting it onto the cluster

Prefer **prepping in place on the cluster** over copying: the script rewrites only the ~44 MB
of parquets and *hardlinks* the 2.3 GB of videos, so the whole thing costs seconds and no
transfer. The cluster already holds `astr_coffee_making_v21`, which is the per-frame-subtask
dataset openpi's `pi05_astribot_anno` trains on (its `pi05_astribot_full` sibling just
overrides the prompt) — i.e. the same shape this script expects as input:

```bash
uv run python configs/astribot/prepare_astri_for_pi05.py \
    /mnt/data/dungnt232_1/data/astr_coffee_making_v21 \
    /mnt/data/dungnt232_1/data/astri_cf_old_opentau \
    --task "Make an iced coffee"
```

Verify before trusting it — the script is strict about its input but says nothing about
whether the cluster copy matches the local one:

```bash
python3 -c "
import json; r='/mnt/data/dungnt232_1/data/astri_cf_old_opentau'
i=json.load(open(f'{r}/meta/info.json')); e=[json.loads(l) for l in open(f'{r}/meta/episodes.jsonl')]
print('episodes',i['total_episodes'],'frames',i['total_frames'],'tasks',i['total_tasks'])
print('segments/ep', min(len(x['segments']) for x in e), max(len(x['segments']) for x in e))
print('prompt', json.loads(open(f'{r}/meta/tasks.jsonl').readline())['task'])"
```

Expect `episodes 186  frames 264874  tasks 1`, `segments/ep 5 6` (episode 75 is missing one
subtask) and the prompt string above. Otherwise `rsync -a` the locally prepared copy instead.

The dataset entry in the JSON deliberately has **no `root`**, so `repo_id` resolves to
`$HF_OPENTAU_HOME/<repo_id>` — OpenTau's equivalent of openpi's `HF_LEROBOT_HOME`. The same
config therefore runs on the cluster and on a workstation by changing one env var:

```bash
export HF_OPENTAU_HOME=/mnt/data/dungnt232_1/data        # cluster (the sbatch default)
export HF_OPENTAU_HOME=/home/kronze/Documents/Work/data  # workstation
```

## Submitting

```bash
# 1. Prepare the dataset on the cluster (REQUIRED -- see Dataset above). Seconds; hardlinks video.
uv run python configs/astribot/prepare_astri_for_pi05.py \
    /mnt/data/dungnt232_1/data/astr_coffee_making_v21 \
    /mnt/data/dungnt232_1/data/astri_cf_old_opentau \
    --task "Make an iced coffee"

# 2. Pre-warm the weights cache if the compute nodes have no egress (REQUIRED in that case).
HF_HOME=/mnt/data/dungnt232_1/hf_cache hf download TensorAuto/pi05_base

# 3. Slurm opens --output before the job script runs, so this is required too.
mkdir -p logs

# 4. Optional: validate the data path + measure dataloader throughput (~1 min).
sbatch configs/astribot/sbatch_delta_stats.sh

# 5. Train.
sbatch configs/astribot/sbatch_train.sh
```

Steps 1-3 are prerequisites, not optimizations. Step 4 is genuinely optional: the delta stats
compute themselves on first use and take only 4.5 s (measured), so its value is catching a
broken data path cheaply, not saving compute.

Skipping step 1 fails in a confusing way rather than cleanly: with no `root` set,
`LeRobotDatasetMetadata` treats a missing directory as "not downloaded yet" and tries to pull
`astri_cf_old_opentau` from the Hub, where it does not exist — so you get a
repo-not-found error rather than a path error. It also `mkdir`s an empty
`$HF_OPENTAU_HOME/astri_cf_old_opentau/meta/` on the way out, which then makes the prep
script refuse the destination as already existing; pass `--overwrite` or remove the stub.

Overrides, all on the `sbatch` line:

```bash
GLOBAL_BATCH=64 sbatch configs/astribot/sbatch_train.sh         # effective batch across ranks
MICRO_BATCH=4   sbatch configs/astribot/sbatch_train.sh         # first thing to lower on OOM
CONFIG=configs/astribot/other.json sbatch configs/astribot/sbatch_train.sh
sbatch configs/astribot/sbatch_train.sh --steps=60000           # extra args reach train.py
```

**Batch semantics differ from openpi.** OpenTau runs accelerate with `split_batches=False`, so
`batch_size` in the JSON is **per rank** and the true global batch is
`batch_size × num_processes`. openpi's `batch_size=32` is a global batch sharded by FSDP. The
sbatch therefore takes `GLOBAL_BATCH` (default 32, matching openpi) and derives the per-rank
split from `SLURM_GPUS_ON_NODE`, passing all three of `--batch_size`,
`--dataloader_batch_size` and `--gradient_accumulation_steps` — the validator rejects an
inconsistent triple, so they must always move together.

## Normalization statistics

Shorter answer than openpi's: there is **no mandatory `compute_norm_stats.py` equivalent**.

- **Absolute state/action stats** — nothing to compute. OpenTau reads them from the dataset's
  `meta/episodes_stats.jsonl`, which this v2.1 dataset already carries for all 186 episodes
  (min/max/mean/std/count per feature). openpi needs its own pass because it writes
  `assets/<config>/<repo_id>/norm_stats.json`; OpenTau just reads the LeRobot metadata.
- **Image stats** — nothing to compute. `use_imagenet_stats` (default true) overwrites them
  with the ImageNet constants.
- **Delta-action stats** — *these do need computing*, because `use_delta_joint_actions: true`.
  Subtracting the chunk-start state changes the distribution the policy trains against: a
  joint's displacement over a 50-step chunk is far smaller than its absolute range, so reusing
  the absolute stats misscales every mapped dim by 2-5x. The pass is
  `O(frames × chunk_size)` = 264,874 × 50 = 13.2M rows, but touches only the numeric
  state/action parquet columns — never video — which is what keeps it affordable.

The delta pass runs **automatically** on first use and caches to
`$HF_OPENTAU_HOME/<repo_id>/meta/delta_action_stats/<key>.json`.

**Measured cost on this dataset: 4.5 s** (all 13,243,700 = 264,874 x 50 anchor rows, exact, no
cap, 10 worker processes on a 12-core box). So precomputing it saves essentially nothing — the
`O(frames x chunk_size)` warning in `delta_action_stats.py` is calibrated for million-episode
sources, not a 186-episode one. Run `sbatch_delta_stats.sh` for what it is actually worth:
validating the whole data path on the cluster in under a minute before committing a 3-day
allocation, and measuring dataloader throughput. Not because the stats are expensive.

### Running it

The pool width comes from `num_workers` — `factory.py` sizes the `ProcessPoolExecutor` from it
("only the main process computes, so sizing the pool from the full `num_workers` neither
oversubscribes the node nor leaves it idle"), and it fans out over the 186 per-episode parquet
files. So parallelism is just a high `--num_workers`.

It is **CPU-only and offline**: no model is built, `profile_dataloader.py` falls back to
`torch.device("cpu")` without CUDA, and `cfg.validate()` only touches the Hub when
`--policy.path` is passed on the CLI (not for a `pretrained_path` in the JSON). So it runs on a
login node:

```bash
export HF_OPENTAU_HOME=/mnt/data/dungnt232_1/data
PROFILE_BATCHES=1 uv run python src/opentau/scripts/profile_dataloader.py \
    --config_path=configs/astribot/pi05_astribot_subtask.json \
    --num_workers=28 --val_freq=0
```

`PROFILE_BATCHES=1` warms the cache and stops — the stats are written while the mixture is
built, before a single batch is fetched. Drop it (default 300) to also measure dataloader
throughput, which is what should actually set `num_workers` for training: with three 480×640
video streams per sample, video decode is the likely bottleneck, and `batches/s` here versus
`steps/s` from `profile_step.py` tells you whether the GPUs are ever waiting.

Or as a batch job, which is the same thing with the node's core count plugged in:

```bash
sbatch configs/astribot/sbatch_delta_stats.sh
```

Re-run it whenever you change `chunk_size`, the delta map, `episodes`/`excluded_episodes`,
`state_index`/`action_index`, or the resample strategy — all of those are in the cache key, so a
change recomputes rather than silently serving stats from a different configuration.

## If it OOMs

In order of preference:

1. `MICRO_BATCH=4` (then 2). Keeps the effective batch via more accumulation steps.
2. `"gradient_checkpointing": true` in the policy block — ~25-33% more compute for ~30-40 GB of
   activation memory. Legal under the DDP / single-process launch the sbatch uses (and under
   FSDP or ZeRO-1/2); only DeepSpeed ZeRO-3 rejects it. Note the field's own docstring is stale
   on this — `train.py` is the authority.
3. `"num_cams": 2` (drop `camera2`) — 512 instead of 768 image tokens, but the right-hand
   subtasks lose their wrist view.

## Notes on a few non-obvious settings

- `"train_expert_only": false` is **required**, not a tuning choice. The response CE loss runs
  through `paligemma.lm_head`, and `train_expert_only` freezes all of PaliGemma — the CE term
  would be logged but could not move a weight. `knowledge_insulation: true` still keeps the
  action MSE out of the backbone.
- `"prompt_max_length": 256` — with `state_type: "discrete"` the state is rendered into the
  prompt as 32 space-separated bin indices, so the real prompt is 149–157 PaliGemma tokens, not
  the ~22 the task string suggests. Anything near 48 silently truncates the state away.
- `"subgoal_drop_prob": 1.0` — pi05 never reads `subgoal*` (only the pi07 low-level policies
  do), but the dataloader decodes 3 extra video frames on 25% of samples by default. The drop
  roll sits before the decode, so this is free throughput. Side effect: `response_drop_prob`
  only fires when subgoals are present, so this also gives 100% response supervision instead of
  70% — which is what you want when the point is that the model predicts the subtask.
- `"discrete_action_max_length": 192`, not the 100 the example configs use. FAST-tokenizing
  this robot's 50-step delta-action chunks (measured over 30k sampled anchors) gives p50 71,
  p99 115, max 168 tokens — a cap of 100 silently truncates the action CE target on **6.9%** of
  chunks, 128 on 0.16%, 192 on none. `pad_discrete_tokens` does log a warning when it
  truncates, so watch the first hundred steps if you change the chunk size or the delta map.
- `"attention_implementation": "sdpa"` — `"fa2"` is accepted but silently falls back to eager.
- `"wandb": {"enable": false}` — the compute nodes cannot reach `api.wandb.ai`, the same reason
  openpi's configs set `wandb_enabled=False`. Metrics go to `logs/train_<jobid>.out`.
- `"STATE": "MIN_MAX"` is load-bearing, not a preference: discrete state binning assumes
  values in `[-1, 1]`. `QUANTILE` would silently fall back to min/max anyway — this dataset's
  `episodes_stats.jsonl` carries no q01/q99.
- Validation loss reads optimistically. `make_dataset` splits on **frames**, not episodes, so
  frame *t* can train while *t+1* validates. For a clean held-out signal, exclude whole
  episodes with `excluded_episodes` on the dataset entry instead of relying on
  `val_split_ratio`.

## Hyperparameters vs openpi

Matched: `batch_size` 32 global, `warmup_steps` 1000, `peak_lr` 5e-5, `clip_gradient_norm` 1.0,
`action_horizon`/`chunk_size` 50, delta mask `make_bool_mask(14, -2)` ≡
`delta_action_state_map {0..13}`.

Deviations, both deliberate: openpi sets `decay_lr == peak_lr` (constant LR after warmup); this
config decays to 5e-6 over the 30k steps. And `steps` is 30,000 here against openpi's 300,000
for `pi05_astribot_anno` — 30k is ~3.6 epochs at batch 32 and is where the LR schedule lands;
raise it with `--steps=` if you want the longer run.

## Sequence budget

The prefix every sample pays for, at these settings:

| block | tokens |
| --- | --- |
| 3 cameras x 256 | 768 |
| prompt (task + 32 discrete state dims) | 256 |
| response indicator + response | 3 + 52 |
| discrete-action indicator + action tokens | 3 + 192 |
| **total prefix** | **1274** |

Two places with slack, if throughput ever matters more than headroom. The prompt pads to 256
but measures 149-157, so ~100 tokens are padding. And `max_state_dim: 32` on a 16-dim robot
renders 16 constant `128` bin indices into that prompt, ~65 tokens of nothing. Both are kept as
they are on purpose: 32/32 is what `pi05_base` was pretrained with, and a narrower state would
shift the prompt distribution the checkpoint expects.
