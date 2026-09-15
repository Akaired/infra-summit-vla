# Bimanual VLA Manipulation — Setting Up a Dinner Table

Submission for the **Intel Physical AI Online Challenge**: *Bimanual VLA
Manipulation with Multi-Modal Reasoning*, challenge option **Setting Up a Dinner
Table**.

Two simulated **SO-101 arms in MuJoCo** take a natural-language instruction,
perceive the scene through simulated cameras, and execute a multi-step
table-setting sequence — open the drawer, retrieve cutlery, pick up the plate and
cup, hand off between arms, pour. The inference pipeline runs **locally on an
Intel Core Ultra Series 2/3 system**, with the policy converted to **OpenVINO IR**
and targeted at CPU / iGPU / NPU.

**Reference documents**
- [Official challenge brief](./Online_Physical_AI_Challenge_Online.pdf) (Intel)
- [Project PRD v2](./PRD_bimanual_vla_robot_simulation_v2.md) — the authoritative build spec
- [CONTRIBUTING.md](./CONTRIBUTING.md) — git workflow (EN/RU)

> **Status: early build.** `/sim` has a placeholder dual-arm scene with full
> domain randomization and a seed viewer; `/policy` has a SmolVLA-via-LeRobot
> training/eval-smoke pipeline. `/eval` now has **two** harnesses: the original
> 10-seed one driven by a dummy policy (`eval/run_episodes.py`) and a native
> SmolVLA one (`eval/eval_smolvla.py`) running a trained 48k checkpoint at
> **7/10** on the drawer + plate subtask, bit-for-bit reproducible — see
> [Running the trained SmolVLA](#running-the-trained-smolvla-48k) and
> [`EVAL_DIAGNOSIS.md`](./EVAL_DIAGNOSIS.md). `/inference` has a runnable
> stub — the `(instruction, obs, state) -> action` runtime and the standalone
> Intel benchmark (`inference/benchmark.py`), dummy-policy-backed until `/policy`
> exports an OpenVINO IR. The real SO-101 arm rig is not in the scene yet, and
> no OpenVINO IR exists yet.

---

## The constraint that shapes everything

The simulation *and* the inference pipeline must run on Intel hardware for the
final demonstration and benchmark. PRD v2 explicitly rejected cloud GPU inference
(HF ZeroGPU and similar): it cannot produce the required Intel CPU/iGPU/NPU
measurement, which is **20 of 100 rubric points**. A cloud or local GPU is still
fine for the *offline training step* — training hardware is unconstrained by the
brief.

So: no cloud hosting on the graded path. Everything below runs locally.

## Repository layout

```
configs/      Single source of truth for paths, seeds, device, precision, checkpoints
sim/          MuJoCo scene: dual SO-101 + dinner table, domain randomization
policy/       LeRobot fine-tuning / distillation, export to ONNX -> OpenVINO IR
inference/    OpenVINO runtime: IR loading, device selection, (instruction, obs, state) -> action
eval/         Two entry points: 10-seed episode harness, and the standalone Intel benchmark
apps/web/     Optional local viewer for demo polish. Not graded, cut first.
```

Each module is independently runnable and testable — that separation is itself a
rubric line (*Technical Quality & Reproducibility*, 10 pts).

## Getting started

```bash
git clone https://github.com/Akaired/infra-summit-vla.git
cd infra-summit-vla

uv venv --python 3.12
source .venv/bin/activate

# /sim: mujoco is not in pyproject's `sim` extra yet — install explicitly
uv pip install mujoco mediapy numpy pyyaml pytest
python sim/smoke_test.py            # scene compiles, name contract holds, cameras render
python sim/test_randomization.py    # domain randomization across all 10 eval seeds
python sim/viewer_seeds.py          # interactive; [ / ] cycle seeds live

# /policy (training only, heavy): set configs/policy.yaml:experiment.device to cpu first
uv pip install "lerobot[smolvla,dataset]==0.6.1"
python policy/dummy_policy.py --config configs/policy.yaml \
  --instruction "open drawer" --smoke-test

# /eval: 10-seed episode harness (reuses the /sim deps above)
python eval/run_episodes.py --config configs/eval.yaml   # writes outputs/eval/
pytest eval/test_harness.py                              # loop runs without crashing

# /inference: runs on any CPU, no OpenVINO or Intel hardware needed
python -m inference.runtime --config configs/inference.yaml --instruction "open the top drawer"
python -m inference.benchmark --inference-config configs/inference.yaml --eval-config configs/eval.yaml
python -m pytest inference/
```

Dependencies are only partially pinned: `pyproject.toml` still carries empty
`sim` / `inference` / `eval` extras that get populated as each module stabilizes.
The full command list lives in [`CLAUDE.md`](./CLAUDE.md#toolchain-and-commands).

## Running the trained SmolVLA (48k)

This is the native-SmolVLA path (`eval/eval_smolvla.py`), separate from the
dummy-policy harness (`eval/run_episodes.py`). It evaluates the drawer + plate
subtask only — cutlery, cup, hand-off and pour are not part of this criterion.

### Prerequisite: the checkpoint

Use **`checkpoints/048000-native-geometry/`**, not the raw 48k output folder.
The weights are bit-identical; the difference is the declared input shapes in
`config.json`, `policy_preprocessor.json` and `train_config.json`. The raw
folder declares `[3, 256, 256]` for all three cameras, which training never
used but the eval harness acted on — that mismatch alone cost 5 of 10 episodes.
See `EVAL_DIAGNOSIS.md` §B11.

`checkpoints/` and `*.safetensors` are gitignored (906 MB), so the folder comes
from the shared drive or HF, not from a clone. `checkpoints/048000-native-geometry/MANIFEST.md`
identifies exactly which artefact is expected.

### 10-seed evaluation

```bash
uv run python eval/eval_smolvla.py --config configs/smolvla_rollout.yaml
```

Writes a fresh `outputs/eval/<name>/runs/<timestamp>-seeds<N>-<nnn>/` per run —
`episodes.jsonl`, `summary.json`, one MP4 per seed. Previous runs are never
overwritten.

Check the config and the checkpoint agree without stepping physics:

```bash
uv run python eval/eval_smolvla.py --config configs/smolvla_rollout.yaml --check-only
```

### Single episode with a spoken / arbitrary instruction

For ASR integration (Speechmatics) and for one-off runs:

```bash
uv run python scratch/run_vla.py --instruction "Open the drawer, ..." --seed 0
```

From Python — load the model once, run many episodes:

```python
from scratch.run_vla import VLASession

session = VLASession("configs/smolvla_rollout.yaml")   # model loads once
record = session.run(asr_text, seed=0)
print(record["success"], record["event_steps"])
```

The integration point is `SmolVLARunner.predict(instruction, observation)` —
the instruction is already a per-call string, so recognised text goes straight
in with no change to the model path. `run_vla.py` reuses the same
`run_episode()` the eval harness uses, so behaviour matches eval step for step.

> **The instruction does not currently steer the robot.** `demo-100`'s
> `meta/tasks.parquet` contains exactly one task string (`task_index: [0]`)
> across all 100 episodes. The language encoder was trained on a single
> sentence, so the text carries no discriminative signal: a different phrase
> will not change behaviour — the robot runs the same sequence. The ASR
> pipeline is still worth building and debugging now, but "the robot obeys
> spoken commands" cannot be demonstrated on this checkpoint. Real language
> conditioning needs a multi-task dataset and a retrain. `run_vla.py` prints a
> warning when the instruction differs from the training one.

### Reproducibility

Both determinism switches are on by default in `configs/smolvla_rollout.yaml`:

| switch | what it removes | measured |
|---|---|---|
| `deterministic_render` | MuJoCo offscreen render differs by ±1 LSB per frame, which diverges an episode over 600 closed-loop steps | required — without it, three runs gave different per-seed timings |
| `deterministic_kernels` | cuDNN autotune picks algorithms by timing, so the first run on a cold GPU differs | +5.0% runtime (134.8 → 141.5 s per 10 episodes) |

With both on, repeated runs are bit-identical including the first. To verify,
run twice and diff `event_steps` in `episodes.jsonl`. To reproduce a pre-fix
run, pass `--no-deterministic-render`.

Decompose determinism by component (scene / render / policy) in seconds:

```bash
uv run python eval/check_determinism.py --seeds 0,3,6
```

### Diagnostics

```bash
# full per-step trace of the plate phase, CSV + verdict per seed
uv run python eval/diagnose_plate.py --config configs/smolvla_rollout.yaml

# action chunk on the pre-weld observation, all 50 steps, both thresholds
uv run python scratch/preweld_chunk_probe.py --config configs/smolvla_rollout.yaml --seeds 0,2,4,9
```

Three single-variable diagnostic configs, each writing to its own output
directory (differences from the baseline are verified programmatically):

| config | changes | question it answers |
|---|---|---|
| `smolvla_rollout_strict.yaml` | three tolerances from `configs/eval.yaml` | how much of 7/10 survives strict grading |
| `smolvla_rollout_dataset_res.yaml` | render resolutions → dataset's 320×240 / 160×120 | does removing the residual resampling stabilise the borderline seeds |
| `smolvla_rollout_nas10.yaml` | `n_action_steps` 50 → 10 | open-loop horizon (**rejected**: 3/5 vs 5/5 on `drawer_opened`) |

### What 7/10 actually means

Read the breakdown, not just the ratio. Across five runs in two numeric
regimes the count is always 7/10, but its composition is not:

- **6 stable successes** (seeds 0, 2, 3, 5, 6, 7) — plate displacement
  0.069–0.188 m against a 0.05 m threshold, 1.4–3.7× margin.
- **2 stable failures** — seed 8 never touches the plate (spawn at the extreme
  +x edge); seed 4 nudges it 1.4–1.7 cm but never gets a commanded-closed
  contact, so the weld never fires.
- **2 unstable** — seeds 1 and 9 flip depending on numerics at the 1e-7 level,
  and when seed 9 passes it passes by 1.4 mm (0.0514 vs 0.0500).

95% Wilson CI for 7/10 on ten seeds is **40%–89%**, which cannot distinguish
6/10 from 8/10. Any checkpoint or config comparison at this sample size is
close to uninformative — see open question 5 below.

## No hardcoded values

The PRD's strictest rule (§5): scene and asset paths, seed lists, randomization
ranges, checkpoint names, inference device, precision, and thresholds all live in
`configs/*.yaml`, never as constants in code. A PR that inlines one of those is
rejected in review. See [`configs/README.md`](./configs/README.md).

## Required deliverables

| # | Deliverable | Where it comes from |
|---|---|---|
| 1 | Reproducible GitHub repository | this repo |
| 2 | Reproducible MuJoCo simulation package | `sim/` + `configs/sim.yaml`, `configs/randomization.yaml` |
| 3 | Intel inference benchmark script | `inference/benchmark.py` (standalone, no physics) + `configs/inference.yaml`, `configs/eval.yaml:benchmark` |
| 4 | Demonstration video across 10 randomized seeds | `eval/` harness output |
| 5 | Technical readme / architecture summary | this file, expanded before submission |

## Judging criteria (100 points)

| Points | Criterion |
|---:|---|
| 30 | End-to-end task completion & bimanual manipulation |
| 20 | VLA / multi-modal reasoning |
| 20 | OpenVINO & Intel Core Ultra optimization |
| 15 | Robustness & generalization (10 randomized seeds) |
| 10 | Technical quality & reproducibility |
| 5 | Innovation & technical demonstration |

## Open questions

Unresolved team decisions, tracked in PRD §7 — not to be settled inside a PR:

1. Base policy: SmolVLA / Pi0.5 / ACT.
2. Demonstration data: self-collected MuJoCo teleop vs. an existing LeRobot dataset.
3. **Who has access to an Intel Core Ultra Series 2/3 machine** — blocks deliverables #3 and #4.
4. Reasoning split between the VLA policy and an auxiliary LLM/VLM planning layer.
5. Target precision (FP16 vs INT8) and device (CPU/iGPU/NPU).

### Eval-harness decisions (new, from the 48k debugging pass)

Separate from PRD §7. Full context in [`EVAL_DIAGNOSIS.md`](./EVAL_DIAGNOSIS.md).

1. **May `configs/randomization.yaml` be changed?** `seeds_full_drawer`
   `[10000, 10025, 10050, 10075, 10100]` lies entirely inside
   `training.seed_range: [10000, 1000000]`, and `is_eval_seed()` only checks
   `seeds` (0–9), so `collect_demonstrations.py`'s assert would not catch an
   overlap. All 100 demo seeds were re-derived: there is **no actual
   contamination** — the guarantee is luck, not construction.
   `test_reserved_full_drawer_seeds_do_not_overlap_training` asserts otherwise
   and is the only failing test (7/8 pass). The file is on the "do not touch"
   list: fix the config, or change the test?
2. **Which harness is canonical on tolerances?** `configs/eval.yaml` and
   `configs/smolvla_rollout.yaml` disagree on three thresholds for the same
   quantities (resting speed 0.05 vs 0.08 m/s, height tolerance 0.02 vs
   0.035 m, lift 0.06 vs 0.05 m). `eval.yaml` describes the older task
   formulation, so it should not win by default.
   `smolvla_rollout_strict.yaml` exists to measure the sensitivity.
3. **A regression test pins a success criterion the harness no longer uses.**
   `plate_placed` used to require the plate within 0.12 m of (0,0) — not
   reachable for ~10% of what the demos actually do. It was replaced with a
   displacement criterion, which is what `configs/eval.yaml:88-89` already
   declared. `longest_successful_hold` and its test were left untouched and
   `longest_displacement_hold` added alongside; the diagnostic reports both.
   Rewrite the test, or mark it legacy?
4. **Can `sim/grasp_assist.py` be unfrozen for one investigation?** At
   `n_action_steps=10` the drawer release was requested while the drawer was
   still 43–81% open, versus 12–15% at the default, and `drawer_opened`
   dropped from 10/10 to 3/5. The horizon change is rejected regardless, but
   `_update_drawer` looks worth a look. File is frozen pending sign-off, so
   only the observation is recorded.
5. **Is a 10-seed eval enough for the deliverable?** 7/10 has a 95% CI of
   40%–89%, and the composition is 6 stable + 2 stable failures + 2 unstable.
   Reporting the bare ratio would overstate it. Widen the seed set, and to how
   many?
6. **Confirm a units fix that reaches `/policy`.**
   `configs/action_space.yaml` declared `units.gripper: meters`; the joint is
   a hinge (`range="-0.174533 1.74533"`, no `type`), so it is radians.
   `policy/create_bimanual_v0.py:181` reads `units` into `action_units`, so
   the wrong unit was being baked into generated policy configs. Changed to
   `radians`, and both action conventions are now documented explicitly
   (`sim/env.py` normalises from [−1,1]; the SmolVLA eval path writes raw
   actuator positions).

Two things deliberately **not** claimed: that the dataset does not need
re-collecting (current data neither requires nor rules it out), and that 7/10
is comparable to the earlier 8/10 (that one was measured with
non-deterministic rendering and was a favourable draw).

## Workflow

The team pushes freely to `main` — no required Pull Requests, no branch
restrictions. See [CONTRIBUTING.md](./CONTRIBUTING.md) for branch-naming
conventions.
