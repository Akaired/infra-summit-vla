# `/inference` — OpenVINO runtime

**Branch:** `feature/openvino-inference` · **Config:** `configs/inference.yaml`

The only module that talks to OpenVINO. Worth 20 of 100 rubric points on its own,
and the reason the whole project is local-first: a cloud GPU cannot produce the
Intel Core Ultra benchmark the brief requires.

## Files

| File | Role |
|---|---|
| `runtime.py` | `InferenceRuntime` — the one public call. Device resolution, latency-hinted compile helpers, compile cache. |
| `benchmark.py` | Standalone Intel benchmark (deliverable #3). Measures **only** the inference path, never MuJoCo. |
| `test_runtime.py` | pytest — proves `predict()` runs end to end on CPU with the stub backend. |

## The one public API

```python
from inference.runtime import InferenceRuntime

runtime = InferenceRuntime("configs/inference.yaml")
action = runtime.predict(instruction, observation, robot_state)
```

`predict(instruction: str, observation, robot_state) -> np.ndarray` **never changes
signature**, whatever backend sits behind it. `/eval` and `/sim` know only this.

- `observation` — a `{camera_name: HxWx3 uint8}` mapping, a `{"images": {...},
  "robot_state": ...}` mapping (the shape `/eval` produces), or a bare RGB frame.
- `robot_state` — a mapping of robot state (reserved for the real model; the stub
  ignores it, same as `policy/dummy_policy.py`).
- returns `(action_dim,)` when `action_chunk_size == 1`, else
  `(action_chunk_size, action_dim)`.

The underlying model (SmolVLA / Pi0.5 / ACT / whatever) stays swappable behind
that call — an explicit PRD requirement, not a nicety.

## Current status: stub until an exported SmolVLA artifact is present

No OpenVINO IR is exported yet (`/policy` does that later). Until then
`InferenceRuntime` delegates to the model-free `DummyPolicy` in `/policy`, so
`/eval` and the benchmark integrate **today**. The swap-in point is a single
function — `_build_backend` in `runtime.py`, marked with a `TODO(inference)`:

```python
# when configs/inference.yaml:model.ir_xml is a real IR:
return _IRBackend(compile_ir_model(cfg))   # + delete the `stub:` block from the config
```

When `velocity_step.xml`, `velocity_step.bin`, and `manifest.json` exist, the runtime loads
the real `_IRBackend`. The manifest records the fixed tensor ABI and checkpoint
used for LeRobot preprocessing/postprocessing; install `.[policy,inference]`
for this path. `compile_ir_model`, `resolve_device`, and `compile_properties` are
already
written against the official device-agnostic OpenVINO pattern:

- **Device selection** reads `configs/inference.yaml:device` (`CPU` / `GPU` /
  `NPU` / `AUTO`), checks it against `openvino.Core().available_devices`, and
  **falls back to CPU with a logged warning** if the requested accelerator is
  absent. `AUTO` passes straight through to OpenVINO's AUTO plugin.
- **Compile** always passes `PERFORMANCE_HINT = LATENCY` (from
  `runtime.performance_hint`) — the correct choice for single-inference robotics.
  `THROUGHPUT` is for many parallel clients and would raise tail latency here.
  `num_streams` / `num_requests` are **never set by hand**: OpenVINO derives them
  from the hint (device-agnostic best practice; the config keeps them
  `null` / `1` as documentation of that intent).
- **Compile cache** — `CACHE_DIR` from `runtime.cache_dir` (on-disk, cross-run)
  plus an in-process dict keyed by `(ir, device, precision, hint)`. Matters most
  for NPU, where first-compile is slow.

## Local dev without Intel hardware

**This stub runs and is tested on any CPU — no Intel Core Ultra, no OpenVINO
install required.** The team does not have access to the Intel machine yet
(PRD §7.3, open logistics question), so **CPU is the safe default** until that is
resolved:

- `openvino` is an optional extra. If it is not importable, `runtime.py` still
  works — the stub backend never touches it, and `runtime_info()` reports
  `openvino_installed: false`.
- Ask for `GPU` or `NPU` in the config on a machine that has neither and the
  runtime logs the miss and runs on CPU. Nothing crashes; the benchmark records
  the resolved device so a CPU-fallback row is never mistaken for a real NPU
  number.
- `benchmark.py` produces a well-formed report today with `stub_run: true` in it.
  Re-run it unchanged once `/policy` exports the IR to get real numbers.

## Commands

```bash
# one-shot runtime check on synthetic input
python -m inference.runtime --config configs/inference.yaml --instruction "open the top drawer"

# standalone Intel benchmark -> writes configs/eval.yaml:benchmark.report_path
python -m inference.benchmark --inference-config configs/inference.yaml --eval-config configs/eval.yaml

# tests
python -m pytest inference/
```

## Known gaps

- `openvino==2025.3.0` is pinned in `pyproject.toml`'s `inference` extra but not
  yet in `uv.lock` (same situation as `mujoco` in the `sim` extra). Lock it when
  the module is exercised on the real Intel machine.
- `_IRBackend` / `compile_ir_model` are unverified — no IR exists to compile
  against. The device-agnostic compile path is written to OpenVINO docs, not yet
  run.

## Open team decision

**Precision and device** (PRD §7.5): FP16 vs INT8, and whether to benchmark
CPU/iGPU/NPU all three or pick one and justify it. `configs/eval.yaml:benchmark`
currently sweeps `[CPU, GPU, NPU] x [FP16, INT8]`. The rubric weighs quantization
choice and device utilization *and* requires that optimization not degrade task
success — so any precision drop needs a success-rate number beside it.

## Definition of done

The same `predict()` call works on CPU, iGPU, and NPU by changing one config key,
and the task success rate is measured on each configuration that gets reported.
