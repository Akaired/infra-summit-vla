"""ONNX -> OpenVINO IR conversion -- WBS 6.2, the export half of the Intel pipeline.

Turns an exported policy ONNX file into the OpenVINO IR pair (``.xml`` graph +
``.bin`` weights) that ``configs/inference.yaml:model`` points at, then proves
numerical parity against ONNX Runtime on synthetic input.

Model-agnostic on purpose: nothing here assumes what network sits inside the
ONNX file. The real export from ``/policy`` (WBS 6.1, SmolVLA / ACT) does not
exist yet, so ``--make-dummy`` builds a tiny two-input stand-in network
(image + robot state -> action) directly with ``onnx.helper`` -- no PyTorch
required -- and runs the full convert + verify path on it. Once
``configs/policy.yaml:export.onnx_path`` is a real file, the same command
converts it with zero code changes.

Usage::

    python -m inference.convert --make-dummy      # dummy onnx -> IR -> parity check
    python -m inference.convert                   # convert the real export, when it lands
    python -m inference.convert --onnx path/to/model.onnx --out-xml out/model.xml

Where it plugs in: the produced IR is exactly what
:func:`inference.runtime.compile_ir_model` reads, and what the
``TODO(inference)`` in :func:`inference.runtime._build_backend` swaps to. This
script never edits the runtime -- it manufactures the artifact the runtime is
waiting for.

Design rules honoured (CONTRIBUTING.md 3, PRD 5):

* No hardcoded tunables. Source/target paths, opset, and the dummy network's
  interface (controls per arm, smoke-image size, seed) all come from
  ``configs/policy.yaml`` and ``configs/inference.yaml``; every CLI flag is an
  override, not the source of truth. The dummy artifacts default to a
  ``dummy/`` directory *derived* from ``export.openvino_ir_dir`` so they can
  never shadow the real ``model.ir_xml``.
* Module boundary: imports from ``/inference`` only.
* ``openvino`` / ``onnx`` / ``onnxruntime`` are imported lazily with clear
  install hints, mirroring how ``runtime.py`` treats ``openvino`` as an
  optional extra -- the rest of the repo must keep working without them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import yaml

from inference.runtime import REPO_ROOT, _resolve_path

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_INSTALL_HINT = (
    "run `uv pip install -e '.[inference]' onnx onnxruntime` in the repo venv"
)

# Absolute-difference tolerances for the ONNX-vs-IR parity check. FP16
# compression rounds every weight to half precision, so bit-exact equality is
# impossible by construction; these are the conventional acceptance bounds,
# overridable with --atol.
DEFAULT_ATOL = {"FP16": 5e-3, "FP32": 1e-4}


def _import_or_die(module_name: str):
    try:
        return __import__(module_name)
    except ImportError as exc:  # pragma: no cover - exercised only when deps missing
        raise RuntimeError(f"{module_name} is not installed; {_INSTALL_HINT}") from exc


def _load_yaml(path: Path) -> Mapping:
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path}: expected a top-level mapping")
    return raw


# ---------------------------------------------------------------------------
# Config views (paths + dummy interface come from YAML, never from code)
# ---------------------------------------------------------------------------
def load_export_spec(policy_config: Path) -> dict:
    """``configs/policy.yaml:export`` -- source ONNX, target IR dir, opset."""
    raw = _load_yaml(policy_config)
    export = raw.get("export")
    if not isinstance(export, Mapping):
        raise ValueError(f"{policy_config}: 'export' mapping is required")
    for key in ("onnx_path", "openvino_ir_dir", "opset"):
        if export.get(key) is None:
            raise ValueError(f"{policy_config}: 'export.{key}' is required")
    opset = export["opset"]
    if type(opset) is not int or opset <= 0:
        raise ValueError("export.opset must be a positive integer")
    return {
        "onnx_path": _resolve_path(export["onnx_path"]),
        "openvino_ir_dir": _resolve_path(export["openvino_ir_dir"]),
        "opset": opset,
    }


def load_dummy_spec(policy_config: Path) -> dict:
    """``configs/policy.yaml:dummy`` -- the provisional policy interface.

    The dummy ONNX mirrors the exact contract the repo already smoke-tests
    (`2 * controls_per_arm` action dims, smoke-image size), so the conversion
    is exercised on the same shapes the real runtime expects.
    """
    raw = _load_yaml(policy_config)
    dummy = raw.get("dummy")
    if not isinstance(dummy, Mapping):
        raise ValueError(f"{policy_config}: 'dummy' mapping is required")
    controls = dummy.get("controls_per_arm")
    seed = dummy.get("seed")
    smoke = dummy.get("smoke_image")
    if type(controls) is not int or controls <= 0:
        raise ValueError("dummy.controls_per_arm must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("dummy.seed must be a non-negative integer")
    if not isinstance(smoke, Mapping):
        raise ValueError("dummy.smoke_image mapping is required")
    height, width = smoke.get("height"), smoke.get("width")
    if type(height) is not int or type(width) is not int or height <= 0 or width <= 0:
        raise ValueError("dummy.smoke_image height/width must be positive integers")
    return {
        "seed": seed,
        "action_dim": 2 * controls,
        "height": height,
        "width": width,
    }


def load_ir_target(inference_config: Path) -> Path:
    """``configs/inference.yaml:model.ir_xml`` -- where the real IR must land."""
    raw = _load_yaml(inference_config)
    model = raw.get("model")
    if not isinstance(model, Mapping) or model.get("ir_xml") is None:
        raise ValueError(f"{inference_config}: 'model.ir_xml' is required")
    return _resolve_path(model["ir_xml"])


# ---------------------------------------------------------------------------
# Dummy model (onnx.helper, no PyTorch)
# ---------------------------------------------------------------------------
def make_dummy_onnx(
    path: Path, *, seed: int, action_dim: int, height: int, width: int, opset: int
) -> Path:
    """Build a small two-input network and save it as ONNX.

    ``image [1,3,H,W]`` --Conv/ReLU/GlobalAveragePool/Flatten--> 8 features,
    concatenated with ``state [1,action_dim]``, through one Gemm to
    ``action [1,action_dim]``. Untrained seeded weights: the *numbers* are
    meaningless, but the *graph* exercises the multi-input convert path the
    real VLA will need (ResNet-style single input does not).
    """
    onnx = _import_or_die("onnx")
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(seed)
    channels = 8  # tiny on purpose; feature width is irrelevant to conversion

    def weight(shape: tuple[int, ...], fan_in: int, name: str):
        scale = np.float32(1.0 / np.sqrt(fan_in))
        return numpy_helper.from_array(
            (rng.standard_normal(shape).astype(np.float32) * scale), name
        )

    initializers = [
        weight((channels, 3, 3, 3), 27, "conv_w"),
        weight((channels,), channels, "conv_b"),
        weight((channels + action_dim, action_dim), channels + action_dim, "gemm_w"),
        weight((action_dim,), action_dim, "gemm_b"),
    ]
    nodes = [
        helper.make_node(
            "Conv",
            ["image", "conv_w", "conv_b"],
            ["conv_out"],
            kernel_shape=[3, 3],
            strides=[2, 2],
            pads=[1, 1, 1, 1],
        ),
        helper.make_node("Relu", ["conv_out"], ["relu_out"]),
        helper.make_node("GlobalAveragePool", ["relu_out"], ["pooled"]),
        helper.make_node("Flatten", ["pooled"], ["image_feat"], axis=1),
        helper.make_node("Concat", ["image_feat", "state"], ["features"], axis=1),
        helper.make_node("Gemm", ["features", "gemm_w", "gemm_b"], ["action"]),
    ]
    graph = helper.make_graph(
        nodes,
        "dummy_bimanual_policy",
        inputs=[
            helper.make_tensor_value_info(
                "image", TensorProto.FLOAT, [1, 3, height, width]
            ),
            helper.make_tensor_value_info("state", TensorProto.FLOAT, [1, action_dim]),
        ],
        outputs=[
            helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, action_dim])
        ],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset)],
        producer_name="infra-summit-vla/inference.convert",
    )
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    logger.info("wrote dummy ONNX %s (opset %d)", path, opset)
    return path


# ---------------------------------------------------------------------------
# Conversion (the actual WBS 6.2 step)
# ---------------------------------------------------------------------------
def convert_onnx_to_ir(
    onnx_path: Path, xml_path: Path, *, compress_to_fp16: bool = True
) -> tuple[Path, Path]:
    """ONNX file -> OpenVINO IR pair (``xml_path`` + sibling ``.bin``)."""
    ov = _import_or_die("openvino")
    if not onnx_path.is_file():
        raise FileNotFoundError(
            f"ONNX model not found at {onnx_path}; /policy (WBS 6.1) has not "
            "exported it yet -- use --make-dummy to exercise the pipeline, or "
            "--onnx to point at another file"
        )
    # Core().read_model hits the ONNX frontend directly. ov.convert_model on a
    # path probes other frontends first, and when torch is installed that probe
    # calls torch.export.load on the .onnx and prints a wall of harmless but
    # scary-looking warnings before falling back to ONNX. Same resulting model.
    ov_model = ov.Core().read_model(str(onnx_path))
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, str(xml_path), compress_to_fp16=compress_to_fp16)
    bin_path = xml_path.with_suffix(".bin")
    if not bin_path.is_file():
        raise RuntimeError(f"conversion wrote {xml_path} but no weights at {bin_path}")
    logger.info(
        "converted %s -> %s + %s (compress_to_fp16=%s)",
        onnx_path.name,
        xml_path.name,
        bin_path.name,
        compress_to_fp16,
    )
    return xml_path, bin_path


# ---------------------------------------------------------------------------
# Parity verification (ONNX Runtime reference vs compiled IR, both on CPU)
# ---------------------------------------------------------------------------
_ORT_TO_NUMPY = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
}


def _synthetic_feed(session, seed: int) -> dict[str, np.ndarray]:
    """One seeded random tensor per declared model input (dynamic dims -> 1)."""
    rng = np.random.default_rng(seed)
    feed: dict[str, np.ndarray] = {}
    for inp in session.get_inputs():
        dtype = _ORT_TO_NUMPY.get(inp.type)
        if dtype is None:
            raise ValueError(
                f"input {inp.name!r} has type {inp.type}; extend _ORT_TO_NUMPY "
                "when the real export needs non-float inputs"
            )
        shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in inp.shape]
        feed[inp.name] = rng.standard_normal(shape).astype(dtype)
    return feed


def verify_parity(onnx_path: Path, xml_path: Path, *, atol: float, seed: int) -> dict:
    """Same input through ONNX Runtime and the compiled IR; outputs must agree.

    Both engines compute the same math on the same numbers, so this validates
    the *conversion* without needing a trained model. Returns a report dict;
    ``passed`` is False when any output exceeds ``atol``.
    """
    ort = _import_or_die("onnxruntime")
    ov = _import_or_die("openvino")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feed = _synthetic_feed(session, seed)
    reference = session.run(None, feed)

    compiled = ov.Core().compile_model(str(xml_path), "CPU")
    result = compiled(feed)
    converted = [np.asarray(result[port]) for port in compiled.outputs]

    if len(reference) != len(converted):
        raise RuntimeError(
            f"output count mismatch: onnxruntime {len(reference)} vs IR {len(converted)}"
        )

    output_names = [out.name for out in session.get_outputs()]
    outputs = []
    for index, (ref, got) in enumerate(zip(reference, converted)):
        if ref.shape != got.shape:
            raise RuntimeError(
                f"output {output_names[index]!r} shape mismatch: "
                f"{ref.shape} vs {got.shape}"
            )
        outputs.append(
            {
                "name": output_names[index],
                "shape": list(ref.shape),
                "max_abs_diff": float(np.max(np.abs(ref - got))) if ref.size else 0.0,
            }
        )
    worst = max((entry["max_abs_diff"] for entry in outputs), default=0.0)
    report = {"outputs": outputs, "max_abs_diff": worst, "atol": atol, "passed": worst <= atol}
    log = logger.info if report["passed"] else logger.error
    log("parity %s: max |onnx - ir| = %.3e (atol %.1e)", "OK" if report["passed"] else "FAILED", worst, atol)
    return report


# ---------------------------------------------------------------------------
# Manifest (reproducibility breadcrumb next to the IR)
# ---------------------------------------------------------------------------
def _repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_manifest(
    xml_path: Path,
    bin_path: Path,
    onnx_path: Path,
    *,
    compress_to_fp16: bool,
    parity: dict | None,
) -> Path:
    versions = {}
    for module_name in ("onnx", "onnxruntime", "openvino"):
        try:
            versions[module_name] = __import__(module_name).__version__
        except ImportError:  # pragma: no cover
            versions[module_name] = None
    manifest = {
        "generated_by": "inference/convert.py",
        "source_onnx": _repo_relative(onnx_path),
        "source_onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "ir_xml": _repo_relative(xml_path),
        "ir_bin": _repo_relative(bin_path),
        "ir_bin_bytes": bin_path.stat().st_size,
        "precision": "FP16" if compress_to_fp16 else "FP32",
        "tool_versions": versions,
        "parity": parity,
    }
    manifest_path = xml_path.with_name(xml_path.stem + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %s", manifest_path)
    return manifest_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a policy ONNX export to OpenVINO IR and verify parity."
    )
    parser.add_argument(
        "--policy-config",
        default="configs/policy.yaml",
        help="policy YAML carrying the 'export' and 'dummy' blocks (default: %(default)s)",
    )
    parser.add_argument(
        "--inference-config",
        default="configs/inference.yaml",
        help="inference YAML carrying 'model.ir_xml' (default: %(default)s)",
    )
    parser.add_argument(
        "--make-dummy",
        action="store_true",
        help="build a small seeded two-input ONNX first and convert that; "
        "artifacts go to a dummy/ directory beside export.openvino_ir_dir",
    )
    parser.add_argument("--onnx", help="source ONNX (default: policy export.onnx_path)")
    parser.add_argument(
        "--out-xml", help="target IR .xml (default: inference model.ir_xml)"
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="keep FP32 weights instead of the default FP16 compression",
    )
    parser.add_argument(
        "--atol",
        type=float,
        help="parity tolerance override (default: %(FP16)s FP16 / %(FP32)s FP32)"
        % {k: f"{v:g}" for k, v in DEFAULT_ATOL.items()},
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="convert only; skip the onnxruntime parity check",
    )
    return parser


def run(args: argparse.Namespace) -> dict:
    policy_config = _resolve_path(args.policy_config)
    inference_config = _resolve_path(args.inference_config)
    export = load_export_spec(policy_config)
    compress_to_fp16 = not args.fp32
    precision = "FP16" if compress_to_fp16 else "FP32"
    atol = args.atol if args.atol is not None else DEFAULT_ATOL[precision]

    if args.make_dummy:
        dummy = load_dummy_spec(policy_config)
        seed = dummy["seed"]
        # Sibling of the configured IR dir, so the dummy can never be mistaken
        # for (or overwrite) the real model.ir_xml the runtime watches for.
        dummy_dir = export["openvino_ir_dir"].parent / "dummy"
        onnx_path = Path(args.onnx) if args.onnx else dummy_dir / "policy_dummy.onnx"
        onnx_path = _resolve_path(onnx_path)
        make_dummy_onnx(
            onnx_path,
            seed=seed,
            action_dim=dummy["action_dim"],
            height=dummy["height"],
            width=dummy["width"],
            opset=export["opset"],
        )
        xml_path = _resolve_path(args.out_xml) if args.out_xml else dummy_dir / "policy_dummy.xml"
    else:
        seed = 0  # feed seed for verification only; any fixed value works
        onnx_path = _resolve_path(args.onnx) if args.onnx else export["onnx_path"]
        xml_path = (
            _resolve_path(args.out_xml)
            if args.out_xml
            else load_ir_target(inference_config)
        )

    xml_path, bin_path = convert_onnx_to_ir(
        onnx_path, xml_path, compress_to_fp16=compress_to_fp16
    )
    parity = None
    if not args.skip_verify:
        parity = verify_parity(onnx_path, xml_path, atol=atol, seed=seed)
    manifest_path = write_manifest(
        xml_path, bin_path, onnx_path, compress_to_fp16=compress_to_fp16, parity=parity
    )
    return {
        "onnx": _repo_relative(onnx_path),
        "ir_xml": _repo_relative(xml_path),
        "ir_bin": _repo_relative(bin_path),
        "precision": precision,
        "manifest": _repo_relative(manifest_path),
        "parity": parity,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    summary = run(_build_arg_parser().parse_args())
    print(json.dumps(summary, indent=2))
    if summary["parity"] is not None and not summary["parity"]["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
