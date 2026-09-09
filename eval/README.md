# `/eval` — evaluation harness and Intel benchmark

**Branch:** `feature/eval-benchmark` · **Config:** `configs/eval.yaml` (+ `sim.yaml`, `randomization.yaml`, `inference.yaml`)

Two **separate entry points** that share this directory and little else.

## 1. Episode harness — deliverable #2 and #4

Drives the full pipeline: reset `/sim` at a given seed, step physics, query
`/inference` for actions, detect subtask completion, stop on success or timeout.
Runs across the 10 seeds in `configs/randomization.yaml` and writes one record per
episode — instruction, seed, per-subtask completion, outcome, timing — in the
format `configs/eval.yaml:logging` specifies.

That log is not incidental: it is the source of both the 10-seed success-rate
summary in the submission and the on-screen state for the demonstration video.

### Run it

```bash
python eval/run_episodes.py --config configs/eval.yaml
```

All defaults come from config; every flag below is an override:

| Flag | Default | Effect |
|---|---|---|
| `--policy {dummy,openvino}` | `dummy` | Which policy drives episodes. `openvino` raises until `/inference` exposes a `Policy`-compatible runtime. |
| `--seeds 0,3,7` | `configs/randomization.yaml:seeds` | Debug subset. A non-reserved seed prints a warning — real results are only over the 10. |
| `--max-steps N` | `configs/eval.yaml:episode.max_steps` | Physics-step budget per episode (one policy action per `sim.yaml:physics.control_decimation` steps). |
| `--output-dir DIR` | `configs/eval.yaml:logging.output_dir` | Where records/videos land. |
| `--video` / `--no-video` | `configs/eval.yaml:logging.record_video` | Per-seed MP4 of the `policy.primary_camera` view. |
| `--quiet` | off | Suppress per-episode progress lines. |

### Output (under `logging.output_dir`, default `outputs/eval/`)

- `episodes.jsonl` — one JSON record per seed: `subtasks` (per-subtask status),
  `subtasks_completed` / `_total` / `_implemented`, `first_completion_step`,
  `outcome`, `success`, `policy_steps`, `physics_steps`, `wall_time_s`,
  `in_drawer_items`, `video_path`.
- `seed_<n>.mp4` — the demo view for that seed, when video is on.
- `summary.json` — aggregate: `full_task_success_ratio` (a raw `"k/10"` string,
  never a percentage — 10 fixed seeds do not support that precision),
  `per_subtask_completed`, `not_implemented_subtasks`.

### Subtask detection

Each subtask in `configs/eval.yaml:subtasks` is verified from MJCF body/joint
state (`eval/subtasks.py`), never from a signal the policy emits. Status is one
of `completed`, `incomplete`, or `not_implemented`. `pour_completed` is
`not_implemented` — there is no simulated liquid in `/sim`, so no state can
confirm a pour; it is logged as such and never as success. Thresholds live in
`configs/eval.yaml:detection`.

### Module layout

| File | Role |
|---|---|
| `run_episodes.py` | CLI entry point, episode loop, JSONL + summary writing |
| `policy_interface.py` | `Policy` Protocol + `build_policy` — the only seam to `/policy` / `/inference` |
| `scene.py` | `EpisodeScene` — reuses `/sim` model compile + `DomainRandomizer`, exposes reset/observe/act/step |
| `subtasks.py` | `SubtaskTracker` + per-subtask geometric detectors |
| `common.py` | repo-root path + YAML helpers |
| `test_harness.py` | pytest: the loop runs one seed with the dummy policy without crashing |

```bash
pytest eval/test_harness.py
```

## 2. Intel benchmark — deliverable #3

Runs the OpenVINO pipeline **standalone, without stepping MuJoCo**, and reports
latency, throughput, device selection, and precision. Keeping physics out is the
whole point: it isolates model performance, which is what the 20-point OpenVINO
criterion is scored on.

It calls into `/inference` rather than re-implementing model loading, so the
benchmarked path is the path the harness actually runs. **Not landed yet.**

## What does not go here

Scene construction (`/sim`), model internals (`/inference`), training (`/policy`).
The harness orchestrates; it does not reimplement.

## Definition of done

One command runs 10 seeds and prints a success rate; a second, independent
command produces the benchmark report — both on the Intel Core Ultra machine,
both reproducible from a clean clone.
