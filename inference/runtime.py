"""OpenVINO inference runtime for the bimanual VLA policy.

Public surface -- ONE method, stable forever regardless of the backend behind it::

    InferenceRuntime(config).predict(instruction, observation, robot_state) -> action

Status: STUB. No OpenVINO IR has been exported by ``/policy`` yet, so the runtime
delegates to the model-free ``DummyPolicy`` in ``/policy``. Every OpenVINO-specific
helper below (device resolution, latency-hinted compile, compile cache) is already
written against the official device-agnostic API, so swapping in the real model is
a single localized change -- see the TODO in :func:`_build_backend`.

Design rules honoured here (CONTRIBUTING.md 3, PRD 5):

* Nothing that ``configs/inference.yaml`` can express is hardcoded -- device,
  precision, performance hint, cache dir, chunk size and the stub policy config
  all come from the YAML.
* Module boundary: imports from ``/policy`` only. Never ``/sim`` or ``/eval``.
* The performance hint drives stream/request counts; we never set
  ``num_streams`` / ``num_requests`` by hand (OpenVINO's device-agnostic advice).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

REPO_ROOT = Path(__file__).resolve().parents[1]

# Accepted enum values, mirrored from the comments in configs/inference.yaml.
# These are validation domains, not tunables.
VALID_DEVICES = ("CPU", "GPU", "NPU", "AUTO")
VALID_PRECISIONS = ("FP32", "FP16", "INT8")
VALID_PERFORMANCE_HINTS = ("LATENCY", "THROUGHPUT")

# OpenVINO's always-present device. Per inference/README.md ("Local dev without
# Intel hardware") this is the safe fallback target until the team has an Intel
# Core Ultra machine -- it is a spec constant, not a magic number.
FALLBACK_DEVICE = "CPU"

# Process-wide cache of compiled models, keyed by (ir_path, device, precision,
# hint). Compilation -- NPU especially -- is slow; never do it twice for the
# same target in one process. OpenVINO's on-disk CACHE_DIR covers the
# cross-process case and is passed as a compile property (see compile_properties).
_COMPILED_MODEL_CACHE: dict[tuple, object] = {}


def clear_compiled_model_cache() -> None:
    """Drop the in-process compiled-model cache (test / long-run hygiene)."""
    _COMPILED_MODEL_CACHE.clear()


def _resolve_path(value: str | Path) -> Path:
    """Resolve ``value`` against the repository root unless already absolute."""
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RuntimeConfig:
    """Parsed, validated view of ``configs/inference.yaml``.

    Built with :meth:`from_config`. ``overrides`` exists so the benchmark can
    sweep device x precision without a second config file; it is a keyword map
    restricted to this dataclass's own fields.
    """

    ir_xml: Path
    ir_bin: Path
    device: str
    precision: str
    performance_hint: str
    cache_dir: Path
    action_chunk_size: int

    @classmethod
    def from_config(
        cls, config_path: str | Path, *, overrides: Mapping | None = None
    ) -> "RuntimeConfig":
        path = _resolve_path(config_path)
        try:
            with path.open(encoding="utf-8") as stream:
                raw = yaml.safe_load(stream)
        except OSError as exc:
            raise ValueError(f"cannot read inference config {path}: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path}: expected a top-level mapping")
        return cls.from_mapping(raw, overrides=overrides)

    @classmethod
    def from_mapping(
        cls, raw: Mapping, *, overrides: Mapping | None = None
    ) -> "RuntimeConfig":
        model = raw.get("model")
        runtime = raw.get("runtime")
        if not isinstance(model, Mapping):
            raise ValueError("inference config: 'model' mapping is required")
        if not isinstance(runtime, Mapping):
            raise ValueError("inference config: 'runtime' mapping is required")

        fields = {
            "ir_xml": model.get("ir_xml"),
            "ir_bin": model.get("ir_bin"),
            "device": raw.get("device"),
            "precision": raw.get("precision"),
            "performance_hint": runtime.get("performance_hint"),
            "cache_dir": runtime.get("cache_dir"),
            "action_chunk_size": raw.get("action_chunk_size"),
        }
        if overrides:
            unknown = set(overrides) - set(fields)
            if unknown:
                raise ValueError(f"unknown RuntimeConfig overrides: {sorted(unknown)}")
            fields.update(overrides)

        for key, value in fields.items():
            if value is None:
                raise ValueError(f"inference config: '{key}' is required (no default)")

        device = str(fields["device"]).upper()
        if device not in VALID_DEVICES:
            raise ValueError(f"device must be one of {VALID_DEVICES}, got {fields['device']!r}")
        precision = str(fields["precision"]).upper()
        if precision not in VALID_PRECISIONS:
            raise ValueError(
                f"precision must be one of {VALID_PRECISIONS}, got {fields['precision']!r}"
            )
        hint = str(fields["performance_hint"]).upper()
        if hint not in VALID_PERFORMANCE_HINTS:
            raise ValueError(
                f"performance_hint must be one of {VALID_PERFORMANCE_HINTS}, "
                f"got {fields['performance_hint']!r}"
            )
        chunk = fields["action_chunk_size"]
        if type(chunk) is not int or chunk <= 0:
            raise ValueError("action_chunk_size must be a positive integer")

        return cls(
            ir_xml=_resolve_path(fields["ir_xml"]),
            ir_bin=_resolve_path(fields["ir_bin"]),
            device=device,
            precision=precision,
            performance_hint=hint,
            cache_dir=_resolve_path(fields["cache_dir"]),
            action_chunk_size=chunk,
        )

    def with_overrides(self, **changes) -> "RuntimeConfig":
        return dataclasses.replace(self, **changes)


# ---------------------------------------------------------------------------
# Device selection  (official OpenVINO device-agnostic pattern)
# ---------------------------------------------------------------------------
def openvino_available_devices() -> list[str] | None:
    """Device ids OpenVINO can see, or ``None`` when OpenVINO is not installed.

    A ``None`` return is the normal case on a non-Intel dev box today: the stub
    does not need OpenVINO at all, so the package is an optional extra.
    """
    try:
        import openvino as ov
    except ImportError:
        return None
    try:
        return list(ov.Core().available_devices)
    except Exception as exc:  # pragma: no cover - plugin/driver failure
        logger.warning("openvino Core().available_devices failed: %s", exc)
        return []


def resolve_device(
    requested: str, available: list[str] | None, *, fallback: str = FALLBACK_DEVICE
) -> str:
    """Pick the device to compile on.

    * ``AUTO`` passes straight through to OpenVINO's AUTO plugin.
    * A concrete request (``CPU`` / ``GPU`` / ``NPU``) is checked against
      ``available`` -- both exact ids and family prefixes, since OpenVINO lists
      multi-adapter devices as ``GPU.0`` / ``GPU.1``.
    * If it is not present, log the miss and fall back to CPU so the pipeline
      still runs on a machine without that accelerator.
    * ``available is None`` (OpenVINO absent) -> return the request unverified;
      the caller is the stub and will not touch OpenVINO anyway.
    """
    requested = str(requested).upper()
    if requested == "AUTO":
        return "AUTO"
    if available is None:
        # CPU is always present, so an unverified CPU request is not worth a warning.
        if requested != fallback:
            logger.warning(
                "openvino not installed; cannot verify device %r is present -- "
                "using it as requested",
                requested,
            )
        return requested
    families = {device_id.split(".", 1)[0] for device_id in available}
    if requested in available or requested in families:
        return requested
    logger.warning(
        "device %r not available on this machine (have: %s); falling back to %r",
        requested,
        sorted(available) or "none",
        fallback,
    )
    return fallback


def compile_properties(cfg: RuntimeConfig) -> dict[str, str]:
    """Device-agnostic compile-time properties for ``core.compile_model``.

    Only two things are set:

    * ``PERFORMANCE_HINT`` -- ``ov::hint::PerformanceMode``. For a single
      robotic inference this is ``LATENCY``; ``THROUGHPUT`` is for serving many
      parallel clients and would raise tail latency here. OpenVINO derives
      ``num_streams`` / ``num_requests`` from this hint, which is why they are
      never set by hand (configs/inference.yaml keeps them ``null`` / ``1`` as
      documentation of that intent).
    * ``CACHE_DIR`` -- persist compiled blobs between runs. Matters most for
      NPU, where first-compile is slow.
    """
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    props = {
        "PERFORMANCE_HINT": cfg.performance_hint,
        "CACHE_DIR": str(cfg.cache_dir),
    }
    # TODO(inference): once /policy exports FP16/INT8 IR, decide whether to also
    # pass INFERENCE_PRECISION_HINT here or rely purely on the exported IR
    # precision. Precision is primarily an export-time concern, so it is only
    # recorded (cache key + report) for now, not forced at compile time.
    return props


def compile_ir_model(cfg: RuntimeConfig):
    """Read + compile the exported OpenVINO IR for the configured device.

    NOT WIRED IN YET. Kept ready for when
    ``configs/inference.yaml:model.ir_xml`` exists; :func:`_build_backend` will
    call this in place of the dummy backend. Raises a clear error if OpenVINO is
    missing or the IR has not been exported.
    """
    try:
        import openvino as ov
    except ImportError as exc:
        raise RuntimeError(
            "openvino is not installed; run `uv pip install -e '.[inference]'` "
            "on the Intel Core Ultra machine"
        ) from exc

    if not cfg.ir_xml.is_file():
        raise FileNotFoundError(
            f"OpenVINO IR not found at {cfg.ir_xml}; /policy has not exported the model yet"
        )

    core = ov.Core()
    device = resolve_device(cfg.device, list(core.available_devices))
    key = (str(cfg.ir_xml), device, cfg.precision, cfg.performance_hint)
    cached = _COMPILED_MODEL_CACHE.get(key)
    if cached is not None:
        logger.info("reusing cached compiled model for %s", key)
        return cached

    model = core.read_model(cfg.ir_xml, cfg.ir_bin)
    compiled = core.compile_model(model, device, compile_properties(cfg))
    _COMPILED_MODEL_CACHE[key] = compiled
    logger.info(
        "compiled %s for %s (precision=%s, hint=%s)",
        cfg.ir_xml.name,
        device,
        cfg.precision,
        cfg.performance_hint,
    )
    return compiled


# ---------------------------------------------------------------------------
# Preprocessing + backend
# ---------------------------------------------------------------------------
def extract_camera_image(observation) -> np.ndarray:
    """Reduce an observation to the single RGB frame the stub backend needs.

    Accepts, in order of preference:

    * a bare ``(H, W, 3)`` uint8 array;
    * a mapping with an ``"images"`` entry that is itself a ``{camera: frame}``
      mapping (the shape ``/eval`` hands to a policy);
    * a mapping of ``{camera: frame}`` directly (as ``/sim``'s cameras produce).

    With a camera mapping the lexicographically first camera is used -- a
    deterministic rule, not a configured camera name. The real IR preprocess
    (all cameras + ``robot_state`` -> input tensors) will live in this module
    too, replacing this function.
    """
    if isinstance(observation, np.ndarray):
        return observation
    if isinstance(observation, Mapping):
        source = observation
        if isinstance(observation.get("images"), Mapping):
            source = observation["images"]
        frames = {name: value for name, value in source.items() if isinstance(value, np.ndarray)}
        if not frames:
            raise ValueError("observation mapping contains no image arrays")
        return frames[sorted(frames)[0]]
    raise TypeError(
        "observation must be an ndarray or a mapping of camera name -> ndarray"
    )


class _DummyBackend:
    """Wraps ``/policy`` ``DummyPolicy`` behind the runtime's ``(dict, dict)`` contract."""

    def __init__(self, policy_config: str | Path):
        from policy.dummy_policy import DummyPolicy

        self._policy = DummyPolicy.from_config(policy_config)

    @property
    def action_dim(self) -> int:
        return 2 * self._policy.controls_per_arm

    @property
    def image_shape(self) -> tuple[int, int, int]:
        return self._policy.smoke_shape

    def predict(self, instruction: str, observation, robot_state) -> np.ndarray:
        frame = extract_camera_image(observation)
        return self._policy.predict(instruction, frame, robot_state)


def _stub_policy_config(inference_config_path: Path) -> Path:
    """Read ``stub.policy_config`` from ``configs/inference.yaml``.

    The stub delegates to ``DummyPolicy``, which needs its own YAML. Rather than
    name ``configs/policy.yaml`` in code, the inference config carries a
    temporary ``stub:`` block that is deleted once ``model.ir_xml`` exists.
    """
    with inference_config_path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    stub = raw.get("stub") if isinstance(raw, Mapping) else None
    if not isinstance(stub, Mapping) or not stub.get("policy_config"):
        raise ValueError(
            "configs/inference.yaml: 'stub.policy_config' is required while the "
            "runtime delegates to the dummy policy -- delete the 'stub:' block once "
            "'model.ir_xml' is a real exported IR"
        )
    return _resolve_path(stub["policy_config"])


def _build_backend(cfg: RuntimeConfig, inference_config_path: Path):
    """Choose the inference backend.

    TODO(inference): when ``/policy`` exports the OpenVINO IR
    (``configs/inference.yaml:model.ir_xml``), replace the body of this function
    with::

        return _IRBackend(compile_ir_model(cfg))

    and delete the ``stub:`` block from ``configs/inference.yaml``. The
    ``_DummyBackend`` path exists only so ``/eval`` and ``/inference`` integrate
    before that export lands. ``InferenceRuntime.predict`` does not change.
    """
    if cfg.ir_xml.is_file():
        logger.warning(
            "IR present at %s but _IRBackend is not wired yet -- still using the "
            "dummy backend (see TODO in inference.runtime._build_backend)",
            cfg.ir_xml,
        )
    return _DummyBackend(_stub_policy_config(inference_config_path))


# ---------------------------------------------------------------------------
# Public runtime
# ---------------------------------------------------------------------------
class InferenceRuntime:
    """Stable inference entry point for ``/eval`` and ``/sim``.

    The single public method is :meth:`predict`. Everything else is
    introspection and never part of the contract.
    """

    def __init__(self, config: str | Path, *, overrides: Mapping | None = None):
        self.config_path = _resolve_path(config)
        self.config = RuntimeConfig.from_config(self.config_path, overrides=overrides)
        self._backend = _build_backend(self.config, self.config_path)

    # -- the one and only public API ---------------------------------------
    def predict(self, instruction: str, observation, robot_state=None) -> np.ndarray:
        """``(instruction, observation, robot_state) -> action``.

        ``observation`` is a mapping (``{camera: frame}`` or ``{"images": {...}}``)
        or a bare RGB uint8 frame. ``robot_state`` is a mapping of robot state
        (ignored by the stub, reserved for the real model). Returns a float32
        array of shape ``(action_dim,)`` when ``action_chunk_size`` is 1, else
        ``(action_chunk_size, action_dim)``.
        """
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string")
        if observation is None:
            raise ValueError("observation is required")

        chunk = self.config.action_chunk_size
        if chunk == 1:
            return self._backend.predict(instruction, observation, robot_state)
        return np.stack(
            [self._backend.predict(instruction, observation, robot_state) for _ in range(chunk)]
        )

    # -- introspection (not the contract) --------------------------------
    @property
    def action_dim(self) -> int:
        return self._backend.action_dim

    @property
    def image_shape(self) -> tuple[int, int, int]:
        return self._backend.image_shape

    def runtime_info(self) -> dict:
        available = openvino_available_devices()
        return {
            "backend": type(self._backend).__name__,
            "openvino_installed": available is not None,
            "available_devices": available,
            "device_requested": self.config.device,
            "device_resolved": resolve_device(self.config.device, available),
            "precision": self.config.precision,
            "performance_hint": self.config.performance_hint,
            "action_chunk_size": self.config.action_chunk_size,
        }


# ---------------------------------------------------------------------------
# Manual one-shot check
# ---------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one InferenceRuntime.predict on synthetic input and print the result."
    )
    parser.add_argument(
        "--config", required=True, help="inference YAML, relative to the repository root"
    )
    parser.add_argument("--instruction", required=True, help="natural-language task instruction")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_arg_parser().parse_args()

    runtime = InferenceRuntime(args.config)
    observation = {"camera": np.zeros(runtime.image_shape, dtype=np.uint8)}
    robot_state = {"joint_positions": [0.0] * runtime.action_dim}
    action = runtime.predict(args.instruction, observation, robot_state)

    print(
        json.dumps(
            {
                "runtime": runtime.runtime_info(),
                "action_shape": list(np.asarray(action).shape),
                "action": np.asarray(action, dtype=float).round(4).tolist(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
