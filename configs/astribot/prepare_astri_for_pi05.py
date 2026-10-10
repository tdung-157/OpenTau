#!/usr/bin/env python

# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Restructure a per-frame-subtask LeRobot v2.1 dataset into pi0.5's prompt/response form.

The source dataset encodes the active subtask in each frame's ``task_index``, so the
prompt the policy sees changes six times per episode and there is no episode-level goal.
pi0.5 wants the opposite split: a constant ``prompt`` naming the whole job, and a
per-frame ``response`` carrying the current subtask as the CE target
(``predict_response: true``). This script performs that transposition on a copy:

  - ``data/.../episode_*.parquet`` gains a ``response`` column (the subtask string that
    the frame's original ``task_index`` pointed at) and has ``task_index`` set to 0.
  - ``meta/tasks.jsonl`` collapses to the single overall task.
  - ``meta/episodes.jsonl`` gets ``tasks: [<overall>]`` plus ``segments`` — the subtask
    start frames, read by ``LeRobotDataset._sample_subgoal_frame`` to keep sampled
    subgoal frames inside the current subtask instead of a blind 4 s lookahead.
  - ``meta/info.json`` declares ``response`` and drops to ``total_tasks: 1``.
  - ``meta/episodes_stats.jsonl`` has its ``task_index`` stats zeroed to match the data.

Videos are hardlinked (same filesystem) rather than copied; pass --copy-videos to copy.
Normalization stats for state/action/images are untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_TASK = (
    "make a cup of coffee, add a drink from the beverage dispenser, "
    "and place the cup on the tray"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", type=Path, help="Source LeRobot v2.1 dataset root.")
    p.add_argument("dst", type=Path, help="Destination root (must not exist unless --overwrite).")
    p.add_argument("--task", default=DEFAULT_TASK, help="The single overall task used as the prompt.")
    p.add_argument("--copy-videos", action="store_true", help="Copy videos instead of hardlinking them.")
    p.add_argument("--overwrite", action="store_true", help="Delete dst first if it exists.")
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def link_tree(src: Path, dst: Path, copy: bool) -> None:
    """Mirror `src` into `dst`, hardlinking leaf files unless `copy`."""
    for root, _dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for name in files:
            s, d = Path(root) / name, dst / rel / name
            if copy:
                shutil.copy2(s, d)
                continue
            try:
                os.link(s, d)
            except OSError:
                shutil.copy2(s, d)


def segments_from_task_index(task_index: list[int]) -> list[int]:
    """Start frame of each contiguous run of equal `task_index`.

    The source labels are already monotonically non-decreasing per episode (verified
    across all 186 episodes), so a run boundary is exactly a subtask boundary.
    """
    starts = [0]
    for i in range(1, len(task_index)):
        if task_index[i] != task_index[i - 1]:
            starts.append(i)
    return starts


def rewrite_parquet(path: Path, tasks: dict[int, str]) -> list[int]:
    """Add `response`, zero `task_index`, and return this episode's segment starts."""
    table = pq.read_table(path)
    task_index = table.column("task_index").to_pylist()
    if "response" in table.column_names:
        raise ValueError(f"{path} already has a 'response' column; refusing to overwrite.")

    missing = sorted({t for t in task_index if t not in tasks})
    if missing:
        raise ValueError(f"{path} references task_index {missing} absent from tasks.jsonl.")

    response = pa.array([tasks[t] for t in task_index], type=pa.string())
    table = table.append_column("response", response)
    col = table.schema.get_field_index("task_index")
    zeros = pa.array([0] * table.num_rows, type=table.column("task_index").type)
    table = table.set_column(col, table.schema.field(col), zeros)

    # The stale `pandas` schema metadata would still describe the pre-rewrite column
    # set. `load_dataset("parquet", ...)` reads the Arrow schema, not this blob, but a
    # mismatched copy is a trap for anything that does read it (e.g. pandas round-trips).
    meta = dict(table.schema.metadata or {})
    meta.pop(b"pandas", None)
    table = table.replace_schema_metadata(meta or None)

    pq.write_table(table, path)
    return segments_from_task_index(task_index)


def main() -> None:
    args = parse_args()
    src, dst = args.src.resolve(), args.dst.resolve()
    if not (src / "meta" / "info.json").is_file():
        raise SystemExit(f"Not a LeRobot dataset root: {src}")
    if dst.exists():
        if not args.overwrite:
            raise SystemExit(f"Destination exists: {dst} (pass --overwrite)")
        shutil.rmtree(dst)

    info = json.loads((src / "meta" / "info.json").read_text(encoding="utf-8"))
    tasks = {rec["task_index"]: rec["task"] for rec in load_jsonl(src / "meta" / "tasks.jsonl")}
    episodes = load_jsonl(src / "meta" / "episodes.jsonl")
    print(f"source: {len(episodes)} episodes, {len(tasks)} subtasks, fps={info['fps']}")

    dst.mkdir(parents=True)
    print("copying data/ ...")
    link_tree(src / "data", dst / "data", copy=True)  # rewritten in place, never link
    if (src / "videos").is_dir():
        print(f"{'copying' if args.copy_videos else 'hardlinking'} videos/ ...")
        link_tree(src / "videos", dst / "videos", copy=args.copy_videos)
    (dst / "meta").mkdir(exist_ok=True)

    chunks_size = info.get("chunks_size", 1000)
    segments_by_ep: dict[int, list[int]] = {}
    for rec in episodes:
        ep = rec["episode_index"]
        rel = info["data_path"].format(episode_chunk=ep // chunks_size, episode_index=ep)
        segments_by_ep[ep] = rewrite_parquet(dst / rel, tasks)
    n_seg = [len(s) for s in segments_by_ep.values()]
    print(f"rewrote {len(segments_by_ep)} parquets; segments/episode min={min(n_seg)} max={max(n_seg)}")

    write_jsonl(dst / "meta" / "tasks.jsonl", [{"task_index": 0, "task": args.task}])

    for rec in episodes:
        rec["tasks"] = [args.task]
        rec["segments"] = segments_by_ep[rec["episode_index"]]
    write_jsonl(dst / "meta" / "episodes.jsonl", episodes)

    info["total_tasks"] = 1
    info["features"]["response"] = {"dtype": "string", "shape": [1], "names": None}
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")

    # `task_index` is now constant 0, so its per-episode stats must say so. Nothing in
    # the training path normalizes `task_index`, but leaving the old 0..5 spread behind
    # is a landmine for any consumer that trusts episodes_stats.
    stats = load_jsonl(src / "meta" / "episodes_stats.jsonl")
    for rec in stats:
        ti = rec.get("stats", {}).get("task_index")
        if ti is not None:
            count = ti.get("count", [1])
            rec["stats"]["task_index"] = {
                "min": [0.0], "max": [0.0], "mean": [0.0], "std": [0.0], "count": count,
            }
    write_jsonl(dst / "meta" / "episodes_stats.jsonl", stats)

    for extra in ("stats.json", "lerobot_annotations.json"):
        if (src / "meta" / extra).is_file():
            shutil.copy2(src / "meta" / extra, dst / "meta" / extra)
    if (src / ".gitattributes").is_file():
        shutil.copy2(src / ".gitattributes", dst / ".gitattributes")

    print(f"\ndone -> {dst}")
    print(f'prompt   = "{args.task}"')
    print("response = per-frame subtask (6 per episode), task_index = 0 everywhere")


if __name__ == "__main__":
    main()

