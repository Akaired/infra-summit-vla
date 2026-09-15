"""Шаг 1: action chunk на PRE-WELD наблюдении, снятом во время живого rollout.

Вопрос: связано ли поведение левого гриппера с состоянием, накопленным к моменту
захвата, а не с грязными targets / нормализацией / общей поломкой модели.

Почему нельзя брать наблюдение при plate_active == True: weld уже активен, MuJoCo
equality изменил физику сцены, и наблюдение перестаёт быть тем, по которому
модель принимала решение о захвате.

Что делает скрипт:
  * гоняет обычный rollout тем же кодом, что eval/eval_smolvla.py;
  * держит предыдущее наблюдение;
  * на переходе plate_active false -> true снимает ДВА наблюдения:
      obs_at_activation  -- то, по которому модель выдала действие на шаге
                            активации (снято до физики этого шага, weld ещё
                            не активен);
      obs_one_step_before -- наблюдение на один control step раньше;
  * для каждого прогоняет policy.predict_action_chunk и выгружает ВСЕ 50 шагов
    чанка: normalized, denormalized, классификацию по двум порогам, все 12
    каналов;
  * sim/grasp_assist.py НЕ изменяется -- перехват целиком здесь.

RNG: predict_action_chunk расходует энтропию (flow matching сэмплирует шум).
Состояние RNG снимается до диагностического вызова и восстанавливается после,
поэтому сам rollout остаётся побитово таким же, как без скрипта.

Пороги классификации:
  * band_zero (0.0)  -- порог предыдущей chunk-пробы на demo-кадрах, приведён
                        для прямой сопоставимости с теми числами;
  * operational      -- реальные пороги grasp assist из эффективного конфига
                        (plate.close_threshold / plate.open_threshold), по ним
                        rollout действительно принимает решения.

Запуск:
  uv run python scratch\\preweld_chunk_probe.py --config configs\\smolvla_rollout.yaml
  uv run python scratch\\preweld_chunk_probe.py --config configs\\smolvla_rollout.yaml \\
      --seeds 0,1,4,8,9 --out outputs\\eval\\preweld
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "sim"))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

from eval.eval_smolvla import (  # noqa: E402
    SmolVLARunner,
    load_rollout_config,
    load_yaml,
    resolve_path,
)
from eval.json_utils import dumps_strict  # noqa: E402
from eval.scene import EpisodeScene  # noqa: E402
from sim.grasp_assist import GraspAssist  # noqa: E402

L_GRIP = 5
R_GRIP = 11
JOINT_NAMES = [
    "left_shoulder_pan", "left_shoulder_lift", "left_elbow_flex",
    "left_wrist_flex", "left_wrist_roll", "left_gripper",
    "right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex",
    "right_wrist_flex", "right_wrist_roll", "right_gripper",
]


# --------------------------------------------------------------------------- #
def snapshot_rng() -> dict[str, Any]:
    state: dict[str, Any] = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["cpu"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def build_batch(runner: SmolVLARunner, instruction: str,
                observation: dict[str, Any]) -> dict[str, Any]:
    """Тот же батч, что собирает SmolVLARunner.predict."""
    batch: dict[str, Any] = {
        runner.state_cfg["feature_key"]: runner._state_tensor(observation),
        runner.task_cfg["task_feature_key"]: instruction,
    }
    images = observation[runner.image_cfg["images_source_key"]]
    for camera_cfg in runner.camera_cfgs:
        batch[camera_cfg["feature_key"]] = runner._image_tensor(
            images[camera_cfg["source_name"]], camera_cfg["feature_key"]
        )
    return batch


def denormalize(runner: SmolVLARunner, chunk: torch.Tensor) -> tuple[np.ndarray, str]:
    """Вернуть (denormalized chunk, как именно получилось).

    postprocessor рассчитан на один шаг; для чанка пробуем несколько форм и
    честно сообщаем, какая сработала, вместо молчаливого приведения.
    """
    flat = chunk.reshape(-1, chunk.shape[-1])

    for label, payload in (
        ("postprocessor(chunk)", chunk),
        ("postprocessor(flat)", flat),
        ("postprocessor({'action': chunk})", {"action": chunk}),
        ("postprocessor({'action': flat})", {"action": flat}),
    ):
        try:
            out = runner.postprocessor(payload)
        except Exception:  # noqa: BLE001 -- пробуем следующую форму
            continue
        if isinstance(out, dict):
            out = out.get("action")
        if out is None:
            continue
        arr = out.detach().cpu().numpy().reshape(flat.shape)
        return arr, label

    return flat.detach().cpu().numpy(), "FAILED -- значения остались normalized"


def probe(runner: SmolVLARunner, instruction: str, observation: dict[str, Any],
          tag: str) -> dict[str, Any]:
    """Полный action chunk по одному наблюдению. RNG не расходуется."""
    rng = snapshot_rng()
    try:
        batch = build_batch(runner, instruction, observation)
        processed = runner.preprocessor(batch)
        with torch.inference_mode():
            chunk = runner.policy.predict_action_chunk(processed)
        norm = chunk.detach().cpu().numpy().reshape(-1, chunk.shape[-1])
        denorm, how = denormalize(runner, chunk)
    finally:
        restore_rng(rng)

    return {"tag": tag, "normalized": norm, "denormalized": denorm,
            "denorm_method": how}


# --------------------------------------------------------------------------- #
def classify(value: float, close_threshold: float, open_threshold: float) -> dict[str, Any]:
    modes = {"closed(-0.16)": -0.16, "idle(0.0)": 0.0, "open(1.2)": 1.2}
    nearest = min(modes, key=lambda k: abs(value - modes[k]))
    if value <= close_threshold:
        operational = "CLOSED"
    elif value >= open_threshold:
        operational = "OPEN"
    else:
        operational = "DEADBAND"
    return {
        "band_zero": "OPEN" if value > 0.0 else "CLOSED",
        "operational": operational,
        "nearest_mode": nearest,
    }


def run_seed(scene: EpisodeScene, runner: SmolVLARunner, cfg: dict[str, Any],
             seed: int, out_dir: Path) -> dict[str, Any]:
    plate_cfg = cfg["grasp_assist"]["plate"]
    close_threshold = float(plate_cfg["close_threshold"])
    open_threshold = float(plate_cfg["open_threshold"])

    rollout_cfg = cfg["rollout"]
    action_repeat = int(rollout_cfg["action_repeat"])
    max_steps = int(rollout_cfg["max_policy_steps"])
    instruction = " ".join(cfg["task"]["instruction"].split())

    grasp = GraspAssist(model=scene.model, data=scene.data,
                        cfg=cfg["grasp_assist"])
    grasp.reset()
    scene.reset(seed)
    runner.reset(seed)

    observation = scene.observe()
    prev_observation = None
    captured: list[dict[str, Any]] = []
    activation_step = None

    for policy_step in range(max_steps):
        action = runner.predict(instruction, observation)
        if bool(cfg["action"]["clip_to_actuator_range"]):
            low = scene.model.actuator_ctrlrange[:, 0]
            high = scene.model.actuator_ctrlrange[:, 1]
            action = np.clip(action, low, high)

        scene.apply_action(action)

        active_before = bool(getattr(grasp, "plate_active", False))
        grasp.update(action, policy_step)
        active_after = bool(getattr(grasp, "plate_active", False))

        if (not active_before) and active_after and activation_step is None:
            activation_step = policy_step
            # observation -- то, по которому модель выдала действие этого шага:
            # снято до физики шага, weld ещё не активен.
            captured.append({
                "probe": probe(runner, instruction, observation,
                               "obs_at_activation"),
                "observation_step": policy_step,
                "plate_active_before": active_before,
                "plate_active_after": active_after,
                "state": np.asarray(observation["robot_state"],
                                    dtype=np.float64)[:12].tolist(),
            })
            if prev_observation is not None:
                captured.append({
                    "probe": probe(runner, instruction, prev_observation,
                                   "obs_one_step_before"),
                    "observation_step": policy_step - 1,
                    "plate_active_before": False,
                    "plate_active_after": False,
                    "state": np.asarray(prev_observation["robot_state"],
                                        dtype=np.float64)[:12].tolist(),
                })
            break

        for _ in range(action_repeat):
            scene.step()

        prev_observation = observation
        observation = scene.observe()

    if activation_step is None:
        print(f"seed={seed}: weld не активировался за {max_steps} шагов -- "
              f"chunk снять не с чего")
        return {"seed": seed, "activation_step": None, "captures": []}

    # ------------------------------------------------------------------ #
    rows: list[dict[str, Any]] = []
    summary_captures = []

    for cap in captured:
        pr = cap["probe"]
        denorm = pr["denormalized"]
        norm = pr["normalized"]
        n_steps = denorm.shape[0]

        counts = {"band_zero_open": 0, "operational_open": 0,
                  "operational_closed": 0, "operational_deadband": 0}

        for i in range(n_steps):
            value = float(denorm[i, L_GRIP])
            cls = classify(value, close_threshold, open_threshold)
            counts["band_zero_open"] += cls["band_zero"] == "OPEN"
            counts["operational_open"] += cls["operational"] == "OPEN"
            counts["operational_closed"] += cls["operational"] == "CLOSED"
            counts["operational_deadband"] += cls["operational"] == "DEADBAND"

            row = {
                "seed": seed,
                "checkpoint": str(resolve_path(cfg["model"]["checkpoint"])),
                "capture": pr["tag"],
                "observation_policy_step": cap["observation_step"],
                "activation_policy_step": activation_step,
                "plate_active_before": cap["plate_active_before"],
                "plate_active_after": cap["plate_active_after"],
                "chunk_step": i,
                "l_grip_normalized": float(norm[i, L_GRIP]),
                "l_grip_denormalized": value,
                "l_grip_band_zero": cls["band_zero"],
                "l_grip_operational": cls["operational"],
                "l_grip_nearest_mode": cls["nearest_mode"],
                "r_grip_denormalized": float(denorm[i, R_GRIP]),
                "close_threshold": close_threshold,
                "open_threshold": open_threshold,
                "denorm_method": pr["denorm_method"],
            }
            for j, name in enumerate(JOINT_NAMES):
                row[f"ch_{j:02d}_{name}"] = float(denorm[i, j])
            rows.append(row)

        summary_captures.append({
            "capture": pr["tag"],
            "observation_policy_step": cap["observation_step"],
            "chunk_steps": n_steps,
            "denorm_method": pr["denorm_method"],
            "state_at_observation": cap["state"],
            "n_open_band_zero": counts["band_zero_open"],
            "n_open_operational": counts["operational_open"],
            "n_closed_operational": counts["operational_closed"],
            "n_deadband_operational": counts["operational_deadband"],
            "l_grip_denormalized": [float(denorm[i, L_GRIP]) for i in range(n_steps)],
            "l_grip_normalized": [float(norm[i, L_GRIP]) for i in range(n_steps)],
        })

        # ---- построчный stdout, как требует бриф ----
        print()
        print(f"=== seed={seed} | {pr['tag']} | наблюдение на шаге "
              f"{cap['observation_step']}, активация на шаге {activation_step} ===")
        print(f"plate_active: до={cap['plate_active_before']} "
              f"после={cap['plate_active_after']}")
        print(f"denormalization: {pr['denorm_method']}")
        print(f"пороги: close={close_threshold}  open={open_threshold}  "
              f"band_zero=0.0")
        print(f"state[5] (текущий qpos гриппера) = "
              f"{cap['state'][L_GRIP]:+.4f}")
        print(f"{'step':>4} {'norm':>9} {'denorm':>9} {'band0':>7} "
              f"{'operational':>12} {'nearest':>14}")
        for i in range(n_steps):
            value = float(denorm[i, L_GRIP])
            cls = classify(value, close_threshold, open_threshold)
            print(f"{i:>4} {float(norm[i, L_GRIP]):>+9.4f} {value:>+9.4f} "
                  f"{cls['band_zero']:>7} {cls['operational']:>12} "
                  f"{cls['nearest_mode']:>14}")
        print(f"n_open (band_zero>0)     = {counts['band_zero_open']}/{n_steps}")
        print(f"n_open (operational)     = {counts['operational_open']}/{n_steps}")
        print(f"n_closed (operational)   = {counts['operational_closed']}/{n_steps}")
        print(f"n_deadband (operational) = {counts['operational_deadband']}/{n_steps}")
        if 3 <= counts["band_zero_open"] <= 9:
            print("  ! n_open в диапазоне 3-9: пограничный случай, "
                  "бинарный вывод делать нельзя -- смотреть построчно")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"preweld_chunk_seed_{seed}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV: {csv_path}")

    return {"seed": seed, "activation_step": activation_step,
            "captures": summary_captures, "csv": str(csv_path)}


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smolvla_rollout.yaml")
    parser.add_argument("--seeds", default="0",
                        help="через запятую; по умолчанию только seed 0")
    parser.add_argument("--out", default="outputs/eval/preweld")
    args = parser.parse_args()

    cfg = load_rollout_config(args.config)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    out_dir = resolve_path(args.out)

    scene = EpisodeScene(load_yaml(cfg["configs"]["sim"]),
                         cfg["configs"]["randomization"])
    runner = SmolVLARunner(cfg)
    runner.validate(scene)

    effective = {
        "config_path": str(resolve_path(args.config)),
        "checkpoint": str(resolve_path(cfg["model"]["checkpoint"])),
        "n_action_steps": int(cfg["model"]["n_action_steps"]),
        "chunk_size": int(runner.policy.config.chunk_size),
        "action_repeat": int(cfg["rollout"]["action_repeat"]),
        "max_policy_steps": int(cfg["rollout"]["max_policy_steps"]),
        "grasp_assist_plate": cfg["grasp_assist"]["plate"],
        "instruction": " ".join(cfg["task"]["instruction"].split()),
        "sim_cameras": load_yaml(cfg["configs"]["sim"])["cameras"],
        "policy_input_features": {
            k: list(v.shape) for k, v in runner.input_features.items()
        },
    }
    print("=== effective config ===")
    print(dumps_strict(effective, indent=2, ensure_ascii=False))

    episodes = [run_seed(scene, runner, cfg, seed, out_dir) for seed in seeds]

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "preweld_chunk_summary.json"
    summary_path.write_text(
        dumps_strict({"effective_config": effective, "episodes": episodes},
                     indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nJSON: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
