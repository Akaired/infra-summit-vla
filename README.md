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
> training/eval-smoke pipeline; `/eval` has the 10-seed episode harness, driven
> by a dummy policy until the real runtime lands. The standalone Intel benchmark
> (`/eval` deliverable #3) and `/inference` are still README-only.
> The real SO-101 arm rig is not in the scene yet.

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
```

Dependencies are only partially pinned: `pyproject.toml` still carries empty
`sim` / `inference` / `eval` extras that get populated as each module stabilizes.
The full command list lives in [`CLAUDE.md`](./CLAUDE.md#toolchain-and-commands).

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
| 3 | Intel inference benchmark script | `eval/` (standalone, no physics) + `configs/inference.yaml` |
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

## Workflow

The team pushes freely to `main` — no required Pull Requests, no branch
restrictions. See [CONTRIBUTING.md](./CONTRIBUTING.md) for branch-naming
conventions.
