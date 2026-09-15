"""Диагностика plate-фазы: полный прогон до max_policy_steps без правок inference.

Скрипт переиспользует те же классы, что и eval/eval_smolvla.py
(SmolVLARunner, EpisodeScene, GraspAssist, TaskTracker) и повторяет
порядок вызовов run_episode() один в один. Ничего в inference-коде
не меняется -- добавляется только запись состояния на каждом шаге.

Зачем нужен отдельно от eval_smolvla.py:

1. Пишет CSV в UTF-8 (а не UTF-16 от PowerShell Tee-Object).
2. Пишет ВЕСЬ эпизод, а не окно шагов.
3. Рядом с флагами GraspAssist пишет события TaskTracker -- видно
   расхождение между weld-автоматом и официальной метрикой.
4. Фиксирует команду гриппера в момент подтверждения подъёма --
   это и есть детектор ложноположительного plate_lifted.
5. Считает возврат левой руки в позу захвата -- проверка гипотезы
   про повтор plate-фазы.

Запуск из корня репозитория:

    python eval/diagnose_plate.py --seeds 0
    python eval/diagnose_plate.py --seeds 0,1,2,315711 --steps 600
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

# Windows consoles default to a legacy code page (cp1251 / cp866). This report
# is written in Russian and uses arrows, so print() dies with
# UnicodeEncodeError -- AFTER every seed has already been simulated. Force
# UTF-8, and never let an encoding problem throw away a finished run.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

# Должно быть выставлено ДО инициализации CUDA-контекста, то есть до
# импорта torch, который происходит внутри eval.eval_smolvla. Без этого
# детерминированные cuBLAS-ядра недоступны.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

# Скрипт должен работать и как `python eval/diagnose_plate.py`, и как
# `python -m eval.diagnose_plate`: в первом случае sys.path[0] — это
# каталог eval/, и пакет eval не находится.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval_smolvla import (  # noqa: E402
    DEFAULT_PLACED_MIN_XY_DISPLACEMENT_M,
    SmolVLARunner,
    TaskTracker,
    evaluation_seeds,
    load_rollout_config,
    load_yaml,
)
from eval.determinism import (  # noqa: E402
    enable_determinism,
    make_render_deterministic,
)
from eval.scene import EpisodeScene  # noqa: E402
from sim.grasp_assist import GraspAssist  # noqa: E402


CSV_COLUMNS = [
    "seed",
    "step",
    "chunk",
    "chunk_idx",
    "replan",
    "grip_cmd",
    "grip_qpos",
    "mode",
    "plate_x",
    "plate_y",
    "plate_z",
    "target_dist",
    "plate_displacement",
    "plate_speed",
    "lift_vs_weld",
    "lift_vs_init",
    "site_x",
    "site_y",
    "site_z",
    "site_plate_xy",
    "weld_active",
    "weld_lifted",
    "weld_armed",
    "weld_release_at",
    "ev_drawer_opened",
    "ev_drawer_closed",
    "ev_plate_lifted",
    "ev_plate_placed",
    "left_pose_dist",
]


def compare_action_traces(
    baseline: list[np.ndarray],
    other: list[np.ndarray],
) -> dict[str, Any]:
    """Первый шаг расхождения двух прогонов и его величина."""
    first_step = None
    first_delta = 0.0
    max_delta = 0.0

    for step, (a, b) in enumerate(zip(baseline, other)):
        delta = float(np.max(np.abs(a - b)))
        max_delta = max(max_delta, delta)

        if delta > 0.0 and first_step is None:
            first_step = step
            first_delta = delta

    return {
        "identical": first_step is None,
        "first_divergent_step": first_step,
        "first_delta": first_delta,
        "max_delta": max_delta,
        "compared_steps": min(len(baseline), len(other)),
    }


def gripper_mode(
    value: float,
    close_threshold: float,
    open_threshold: float,
) -> str:
    if value <= close_threshold:
        return "CLOSED"
    if value >= open_threshold:
        return "OPEN"
    return "BETWEEN"


def left_arm_qpos_addresses(model: Any, count: int = 5) -> list[int]:
    """qpos-адреса первых `count` актуаторов (левая рука без гриппера)."""
    addresses: list[int] = []
    for actuator_index in range(count):
        joint_id = int(model.actuator_trnid[actuator_index, 0])
        if joint_id < 0:
            raise ValueError(
                f"Actuator {actuator_index} is not joint-driven"
            )
        addresses.append(int(model.jnt_qposadr[joint_id]))
    return addresses


def count_pose_returns(
    distances: list[float],
    threshold: float,
    minimum_separation: int,
) -> list[int]:
    """Индексы повторных возвратов позы в окрестность позы захвата.

    Возврат засчитывается, когда расстояние снова опускается ниже
    порога после того, как оно из этого порога выходило.
    """
    returns: list[int] = []
    outside = False
    last_return = -minimum_separation

    for index, distance in enumerate(distances):
        if distance is None or math.isnan(distance):
            continue

        if distance > threshold:
            outside = True
            continue

        if outside and index - last_return >= minimum_separation:
            returns.append(index)
            last_return = index
            outside = False

    return returns

def longest_displacement_hold(
    rows: list[dict[str, Any]],
    min_displacement: float,
    surface_tolerance: float,
    resting_speed: float,
) -> tuple[int, int | None]:
    """Same as ``longest_successful_hold`` but with the criterion the harness
    actually uses: the plate MOVED at least ``min_displacement`` from where it
    spawned, came back to the surface, and is at rest with the weld released.

    ``longest_successful_hold`` is kept unchanged because
    eval/test_eval_regressions.py pins its absolute-target semantics. See
    TaskTracker._placement_terms for why an absolute target cannot be set
    defensibly for this scene.
    """
    best_run = current_run = 0
    best_start = run_start = None

    for row in rows:
        holds = (
            row["ev_plate_placed"] is not None
            and row["ev_plate_lifted"] is not None
            and row["ev_drawer_opened"] is not None
            and row["ev_drawer_closed"] is not None
            and not row["weld_active"]
            and row["plate_displacement"] >= min_displacement
            and abs(row["lift_vs_init"]) <= surface_tolerance
            and row["plate_speed"] <= resting_speed
        )
        if holds:
            current_run += 1
            if current_run == 1:
                run_start = row["step"]
            if current_run > best_run:
                best_run, best_start = current_run, run_start
        else:
            current_run = 0

    return best_run, best_start


def longest_successful_hold(
    rows: list[dict[str, Any]],
    target_radius: float,
    surface_tolerance: float,
    resting_speed: float,
) -> tuple[int, int | None]:
    """Самая длинная серия подряд идущих шагов, где условия успеха
    держатся И weld отпущен. Возвращает (длина, шаг начала).

    TaskTracker латчит plate_placed навсегда; эпизод же засчитывается
    только если условия удержались success_hold_policy_steps подряд.
    Проверка `not weld_active` отсекает "успех" с приклеенной тарелкой.
    """
    best_run = current_run = 0
    best_start = run_start = None

    for row in rows:
        holds = (
            row["ev_plate_placed"] is not None
            and row["ev_plate_lifted"] is not None
            and row["ev_drawer_opened"] is not None
            and row["ev_drawer_closed"] is not None
            and not row["weld_active"]
            and row["target_dist"] <= target_radius
            and abs(row["lift_vs_init"]) <= surface_tolerance
            and row["plate_speed"] <= resting_speed
        )
        if holds:
            current_run += 1
            if current_run == 1:
                run_start = row["step"]
            if current_run > best_run:
                best_run, best_start = current_run, run_start
        else:
            current_run = 0

    return best_run, best_start

def diagnose_seed(
    scene: EpisodeScene,
    runner: SmolVLARunner,
    cfg: dict[str, Any],
    seed: int,
    max_steps: int,
    output_dir: Path,
    pose_return_threshold: float,
    pose_return_separation: int,
    write_csv: bool = True,
    reach_height: float = 0.10,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    scene.reset(seed)
    runner.reset(seed)

    grasp = GraspAssist(
        model=scene.model,
        data=scene.data,
        cfg=cfg["grasp_assist"],
    )
    grasp.reset()

    tracker = TaskTracker(scene, cfg["evaluation"], grasp_assist=grasp)

    instruction = " ".join(cfg["task"]["instruction"].split())
    action_repeat = int(cfg["rollout"]["action_repeat"])
    clip_action = bool(cfg["action"]["clip_to_actuator_range"])

    plate_cfg = cfg["grasp_assist"]["plate"]
    evaluation_plate_cfg = cfg["evaluation"]["plate"]

    plate_body_name = plate_cfg["object_body"]
    plate_body_id = scene.model.body(plate_body_name).id
    gripper_site_id = scene.model.site(plate_cfg["gripper_site"]).id

    gripper_joint_id = scene.model.joint("left_gripper").id
    gripper_qpos_address = int(
        scene.model.jnt_qposadr[gripper_joint_id]
    )

    gripper_action_index = int(plate_cfg["gripper_action_index"])
    close_threshold = float(plate_cfg["close_threshold"])
    open_threshold = float(plate_cfg["open_threshold"])
    minimum_lift = float(plate_cfg["minimum_lift_before_release_m"])

    target_xy = np.asarray(
        evaluation_plate_cfg["target_xy_m"],
        dtype=np.float64,
    )
    target_radius = float(evaluation_plate_cfg["target_radius_m"])
    eval_lift_delta = float(evaluation_plate_cfg["lifted_delta_z_m"])

    left_arm_addresses = left_arm_qpos_addresses(scene.model)

    # Сколько шагов чанка исполняется до переплана. При
    # n_action_steps == chunk_size политика идёт полностью
    # разомкнуто и не видит последствий своих действий внутри окна.
    n_action_steps = int(cfg["model"]["n_action_steps"])

    initial_plate_z = float(scene.data.xpos[plate_body_id][2])
    initial_plate_xy = (
        scene.data.xpos[plate_body_id][:2].copy().astype(np.float64)
    )

    observation = scene.observe()

    rows: list[dict[str, Any]] = []
    action_trace: list[np.ndarray] = []
    pose_distances: list[float] = []
    activation_pose: np.ndarray | None = None

    facts: dict[str, Any] = {
        "seed": seed,
        "weld_activated_step": None,
        "weld_activation_mode": None,
        "first_open_step": None,
        "first_open_target_dist": None,
        "first_open_lift_vs_weld": None,
        "weld_lifted_step": None,
        "weld_lifted_mode": None,
        "eval_lifted_step": None,
        "eval_lifted_mode": None,
        "weld_released_step": None,
        "eval_placed_step": None,
    }

    coupling_samples: list[float] = []
    previous_weld_active = False

    for step in range(max_steps):
        action = runner.predict(instruction, observation)

        if clip_action:
            low = scene.model.actuator_ctrlrange[:, 0]
            high = scene.model.actuator_ctrlrange[:, 1]
            action = np.clip(action, low, high)

        action_trace.append(
            np.asarray(action, dtype=np.float64).copy()
        )

        scene.apply_action(action)
        grasp.update(action, step)

        for _ in range(action_repeat):
            scene.step()

        observation = scene.observe()
        tracker.update(step, action)

        gripper_command = float(action[gripper_action_index])
        mode = gripper_mode(
            gripper_command,
            close_threshold,
            open_threshold,
        )

        plate_xyz = (
            scene.data.xpos[plate_body_id].copy().astype(np.float64)
        )
        plate_displacement = float(
            np.linalg.norm(plate_xyz[:2] - initial_plate_xy)
        )
        target_distance = float(
            np.linalg.norm(plate_xyz[:2] - target_xy)
        )
        site_xyz = (
            scene.data.site_xpos[gripper_site_id]
            .copy()
            .astype(np.float64)
        )
        site_z = float(site_xyz[2])

        # Горизонтальный промах схвата мимо тарелки. Главный признак
        # того, ведёт ли политика руку к реальной тарелке или в
        # заученную среднюю точку стола.
        site_plate_xy = float(
            np.linalg.norm(site_xyz[:2] - plate_xyz[:2])
        )

        activation_z = grasp.plate_activation_z
        lift_vs_weld = (
            float("nan")
            if activation_z is None
            else float(plate_xyz[2]) - float(activation_z)
        )
        lift_vs_init = float(plate_xyz[2]) - initial_plate_z

        left_pose = np.asarray(
            [scene.data.qpos[a] for a in left_arm_addresses],
            dtype=np.float64,
        )

        # --- фиксация ключевых моментов -------------------------------
        if grasp.plate_active and not previous_weld_active:
            facts["weld_activated_step"] = step
            facts["weld_activation_mode"] = mode
            activation_pose = left_pose.copy()

        if previous_weld_active and not grasp.plate_active:
            facts["weld_released_step"] = step

        previous_weld_active = bool(grasp.plate_active)

        if (
            facts["weld_activated_step"] is not None
            and facts["first_open_step"] is None
            and mode == "OPEN"
        ):
            facts["first_open_step"] = step
            facts["first_open_target_dist"] = target_distance
            facts["first_open_lift_vs_weld"] = lift_vs_weld

        if (
            grasp.plate_was_lifted
            and facts["weld_lifted_step"] is None
        ):
            facts["weld_lifted_step"] = step
            facts["weld_lifted_mode"] = mode

        events = tracker.events
        event_names = cfg["evaluation"]["events"]
        eval_lifted_at = events.get(event_names["plate_lifted"])
        eval_placed_at = events.get(event_names["plate_placed"])

        if (
            eval_lifted_at is not None
            and facts["eval_lifted_step"] is None
        ):
            facts["eval_lifted_step"] = eval_lifted_at
            facts["eval_lifted_mode"] = mode

        if (
            eval_placed_at is not None
            and facts["eval_placed_step"] is None
        ):
            facts["eval_placed_step"] = eval_placed_at

        if grasp.plate_active:
            coupling_samples.append(float(plate_xyz[2]) - site_z)

        if activation_pose is None:
            pose_distance = float("nan")
        else:
            pose_distance = float(
                np.linalg.norm(left_pose - activation_pose)
            )
        pose_distances.append(pose_distance)

        rows.append(
            {
                "seed": seed,
                "step": step,
                "chunk": step // n_action_steps,
                "chunk_idx": step % n_action_steps,
                "replan": step % n_action_steps == 0,
                "grip_cmd": round(gripper_command, 5),
                "grip_qpos": round(
                    float(scene.data.qpos[gripper_qpos_address]), 5
                ),
                "mode": mode,
                "plate_x": round(float(plate_xyz[0]), 5),
                "plate_y": round(float(plate_xyz[1]), 5),
                "plate_z": round(float(plate_xyz[2]), 5),
                "target_dist": round(target_distance, 5),
                "plate_displacement": round(plate_displacement, 5),
                "plate_speed": round(
                    scene.body_linear_speed(plate_body_name), 5
                ),
                "lift_vs_weld": round(lift_vs_weld, 5),
                "lift_vs_init": round(lift_vs_init, 5),
                "site_x": round(float(site_xyz[0]), 5),
                "site_y": round(float(site_xyz[1]), 5),
                "site_z": round(site_z, 5),
                "site_plate_xy": round(site_plate_xy, 5),
                "weld_active": grasp.plate_active,
                "weld_lifted": grasp.plate_was_lifted,
                "weld_armed": grasp.plate_release_armed,
                "weld_release_at": grasp.plate_release_requested_at,
                "ev_drawer_opened": events.get(
                    event_names["drawer_opened"]
                ),
                "ev_drawer_closed": events.get(
                    event_names["drawer_closed"]
                ),
                "ev_plate_lifted": eval_lifted_at,
                "ev_plate_placed": eval_placed_at,
                "left_pose_dist": round(pose_distance, 5),
            }
        )

    csv_path = output_dir / f"plate_diag_seed_{seed}.csv"
    if write_csv:
        with csv_path.open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    # --- выводы ------------------------------------------------------
    verdict: dict[str, Any] = dict(facts)
    verdict["csv"] = str(csv_path)
    verdict["steps"] = len(rows)

    verdict["weld_never_released"] = (
        facts["weld_activated_step"] is not None
        and facts["weld_released_step"] is None
    )

    first_open_lift = facts["first_open_lift_vs_weld"]
    first_open_dist = facts["first_open_target_dist"]

    verdict["premature_open"] = bool(
        first_open_lift is not None
        and not math.isnan(first_open_lift)
        and first_open_lift < minimum_lift
    )
    verdict["open_far_from_target"] = bool(
        first_open_dist is not None
        and first_open_dist > target_radius
    )
    verdict["n_action_steps"] = n_action_steps
    verdict["open_loop_window_s"] = round(
        n_action_steps
        * action_repeat
        * scene.control_decimation
        * float(scene.sim_cfg["physics"]["timestep"]),
        3,
    )

    activation_step = facts["weld_activated_step"]
    open_step = facts["first_open_step"]
    verdict["weld_chunk"] = (
        None if activation_step is None
        else activation_step // n_action_steps
    )
    verdict["first_open_chunk"] = (
        None if open_step is None else open_step // n_action_steps
    )
    # Если захват и открытие попали в один чанк, открытие было
    # спланировано ДО контакта и не является реакцией на захват.
    verdict["open_planned_before_grasp"] = bool(
        activation_step is not None
        and open_step is not None
        and activation_step // n_action_steps
        == open_step // n_action_steps
    )

    verdict["weld_lift_false_positive"] = (
        facts["weld_lifted_mode"] == "OPEN"
    )
    verdict["eval_lift_false_positive"] = (
        facts["eval_lifted_mode"] == "OPEN"
    )

    if coupling_samples:
        verdict["coupling_mean_m"] = round(
            float(np.mean(coupling_samples)), 5
        )
        verdict["coupling_std_m"] = round(
            float(np.std(coupling_samples)), 5
        )
    else:
        verdict["coupling_mean_m"] = None
        verdict["coupling_std_m"] = None

    # --- горизонтальный промах мимо тарелки --------------------------
    approach = [
        r for r in rows
        if facts["weld_activated_step"] is None
        or r["step"] <= facts["weld_activated_step"]
    ]

    # Горизонтальный промах сам по себе обманчив: рука пролетает над
    # тарелкой на высоте 15-20 см, и промах формально минимален, хотя
    # схват никогда не был на высоте захвата. Поэтому промах считается
    # только на шагах, где схват реально опущен к тарелке.
    low = [
        r for r in approach
        if r["site_z"] - r["plate_z"] <= reach_height
    ]

    closest_any = min(approach, key=lambda r: r["site_plate_xy"])
    verdict["min_xy_anywhere"] = closest_any["site_plate_xy"]
    verdict["min_xy_anywhere_step"] = closest_any["step"]
    verdict["min_xy_anywhere_height"] = round(
        closest_any["site_z"] - closest_any["plate_z"], 5
    )

    if low:
        closest_low = min(low, key=lambda r: r["site_plate_xy"])
        verdict["min_xy_at_reach"] = closest_low["site_plate_xy"]
        verdict["min_xy_at_reach_step"] = closest_low["step"]
    else:
        verdict["min_xy_at_reach"] = None
        verdict["min_xy_at_reach_step"] = None

    verdict["steps_at_reach_height"] = len(low)
    verdict["reach_height_m"] = reach_height

    closest_3d = min(
        approach,
        key=lambda r: math.hypot(
            r["site_plate_xy"], r["site_z"] - r["plate_z"]
        ),
    )
    verdict["min_dist_3d"] = round(
        math.hypot(
            closest_3d["site_plate_xy"],
            closest_3d["site_z"] - closest_3d["plate_z"],
        ),
        5,
    )
    verdict["min_dist_3d_step"] = closest_3d["step"]
    verdict["plate_spawn_xy"] = [rows[0]["plate_x"], rows[0]["plate_y"]]

    # --- настоящий успех, а не просто зафиксированное событие --------
    # TaskTracker латчит plate_placed навсегда, но эпизод засчитывается
    # только если условия держатся success_hold_policy_steps подряд.
    hold_required = int(
        cfg["rollout"]["success_hold_policy_steps"]
    )
    surface_tolerance = float(
        evaluation_plate_cfg["surface_z_tolerance_m"]
    )
    resting_speed = float(
        evaluation_plate_cfg["resting_linear_speed_mps"]
    )

    min_displacement = float(
        evaluation_plate_cfg.get(
            "placed_min_xy_displacement_m",
            DEFAULT_PLACED_MIN_XY_DISPLACEMENT_M,
        )
    )

    # The harness criterion.
    best_run, best_start = longest_displacement_hold(
        rows, min_displacement, surface_tolerance, resting_speed
    )

    # The absolute-target criterion the spec still pins, kept for comparison
    # so a disagreement between the two is visible rather than silent.
    legacy_run, legacy_start = longest_successful_hold(
        rows, target_radius, surface_tolerance, resting_speed
    )

    verdict["success_hold_required"] = hold_required
    verdict["success_hold_best"] = best_run
    verdict["success_hold_start"] = best_start
    verdict["real_success"] = best_run >= hold_required
    verdict["placed_min_xy_displacement_m"] = min_displacement
    verdict["success_hold_best_legacy_target"] = legacy_run
    verdict["success_hold_start_legacy_target"] = legacy_start
    # Событие записано, но условия не удержались -> метрика завышена.
    verdict["placed_latched_but_failed"] = bool(
        facts["eval_placed_step"] is not None
        and best_run < hold_required
    )

    activation_step = facts["weld_activated_step"]
    if activation_step is None:
        verdict["pose_returns"] = []
    else:
        tail = pose_distances[activation_step:]
        returns = count_pose_returns(
            tail,
            pose_return_threshold,
            pose_return_separation,
        )
        verdict["pose_returns"] = [
            activation_step + index for index in returns
        ]

    verdict["pose_return_count"] = len(verdict["pose_returns"])
    verdict["eval_lift_delta_threshold_m"] = eval_lift_delta

    return verdict, action_trace


def print_verdict(verdict: dict[str, Any]) -> None:
    seed = verdict["seed"]
    print()
    print(f"=== SEED {seed} ===")
    print(f"шагов прогнано: {verdict['steps']}")
    print(f"CSV: {verdict['csv']}")

    activation = verdict["weld_activated_step"]
    if activation is None:
        print("weld не активировался ни разу — тарелку не тронули")
    else:
        print(
            f"weld активирован: шаг {activation} "
            f"(гриппер {verdict['weld_activation_mode']})"
        )

    print(
        f"чанк: n_action_steps={verdict['n_action_steps']}, "
        f"разомкнутое окно {verdict['open_loop_window_s']} с"
    )
    if verdict["weld_chunk"] is not None:
        print(f"  weld в чанке {verdict['weld_chunk']}")

    if verdict["first_open_step"] is not None:
        print(
            f"первый OPEN после захвата: шаг "
            f"{verdict['first_open_step']} "
            f"(чанк {verdict['first_open_chunk']}), "
            f"target_dist={verdict['first_open_target_dist']:.4f}, "
            f"lift_vs_weld={verdict['first_open_lift_vs_weld']:.4f}"
        )
        if verdict["premature_open"]:
            print("  ! открытие ДО подтверждения подъёма")
        if verdict["open_far_from_target"]:
            print("  ! тарелка вне целевого радиуса при открытии")
        if verdict["open_planned_before_grasp"]:
            print(
                "  ! открытие и захват в одном чанке — команда "
                "была спланирована до контакта, это не реакция"
            )

    if verdict["weld_lifted_step"] is not None:
        print(
            f"weld lifted: шаг {verdict['weld_lifted_step']} "
            f"(гриппер {verdict['weld_lifted_mode']})"
        )
        if verdict["weld_lift_false_positive"]:
            print("  ! подъём засчитан при раскрытых челюстях")

    if verdict["eval_lifted_step"] is not None:
        print(
            f"eval plate_lifted: шаг {verdict['eval_lifted_step']} "
            f"(гриппер {verdict['eval_lifted_mode']})"
        )
        if verdict["eval_lift_false_positive"]:
            print("  ! МЕТРИКА ЛОЖНОПОЛОЖИТЕЛЬНАЯ")
    else:
        print("eval plate_lifted: не сработало")

    print(
        f"тарелка на старте: "
        f"({verdict['plate_spawn_xy'][0]:+.4f}, "
        f"{verdict['plate_spawn_xy'][1]:+.4f})"
    )
    print(
        f"промах в любой точке: "
        f"{verdict['min_xy_anywhere']:.4f} м "
        f"на шаге {verdict['min_xy_anywhere_step']}, "
        f"но схват был на {verdict['min_xy_anywhere_height']:+.4f} м "
        f"над тарелкой"
    )
    if verdict["min_xy_at_reach"] is None:
        print(
            f"  схват НИ РАЗУ не опускался к тарелке ближе "
            f"{verdict['reach_height_m']} м по высоте"
        )
    else:
        print(
            f"  промах на высоте захвата: "
            f"{verdict['min_xy_at_reach']:.4f} м на шаге "
            f"{verdict['min_xy_at_reach_step']} "
            f"({verdict['steps_at_reach_height']} шагов на высоте)"
        )
    print(
        f"  минимальное 3D-расстояние: "
        f"{verdict['min_dist_3d']:.4f} м на шаге "
        f"{verdict['min_dist_3d_step']}"
    )

    if verdict["eval_placed_step"] is not None:
        print(f"eval plate_placed: шаг {verdict['eval_placed_step']}")
    print(
        f"условия успеха держались "
        f"{verdict['success_hold_best']} шагов подряд "
        f"(нужно {verdict['success_hold_required']}) -> "
        f"{'УСПЕХ' if verdict['real_success'] else 'провал'}"
    )
    if verdict["placed_latched_but_failed"]:
        print(
            "  ! plate_placed записан, но условия не удержались — "
            "метрика завышена"
        )

    if verdict["weld_never_released"]:
        print("! weld так и не отпущен до конца эпизода")
    elif verdict["weld_released_step"] is not None:
        print(f"weld отпущен: шаг {verdict['weld_released_step']}")

    if verdict["coupling_mean_m"] is not None:
        print(
            f"связь plate_z - site_z при активном weld: "
            f"{verdict['coupling_mean_m']:+.4f} "
            f"± {verdict['coupling_std_m']:.4f} м"
        )

    print(
        f"возвратов левой руки в позу захвата: "
        f"{verdict['pose_return_count']} "
        f"{verdict['pose_returns'][:10]}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Полный трейс plate-фазы без изменения inference-кода"
        )
    )
    parser.add_argument(
        "--config",
        default="configs/smolvla_rollout.yaml",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="через запятую, например 0,1,2. По умолчанию — из конфига",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="по умолчанию rollout.max_policy_steps",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="по умолчанию <output.directory>/diag",
    )
    parser.add_argument(
        "--pose-return-threshold",
        type=float,
        default=0.25,
        help="рад, порог близости позы к позе захвата",
    )
    parser.add_argument(
        "--pose-return-separation",
        type=int,
        default=20,
        help="минимальный интервал между засчитанными возвратами",
    )
    parser.add_argument(
        "--reach-height",
        type=float,
        default=0.10,
        help=(
            "на сколько метров схват должен опуститься к тарелке, "
            "чтобы промах считался промахом захвата, а не пролёта"
        ),
    )
    parser.add_argument(
        "--deterministic-render",
        action="store_true",
        help=(
            "выключить сглаживание и тени в офскрин-рендере — "
            "единственный измеренный источник расхождения"
        ),
    )
    parser.add_argument(
        "--keep-shadows",
        action="store_true",
        help=(
            "с --deterministic-render выключить только сглаживание, "
            "тени оставить — ближе к картинке обучения"
        ),
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "детерминированные ядра: cudnn.deterministic, без TF32, "
            "без автоподбора алгоритмов"
        ),
    )
    parser.add_argument(
        "--strict-determinism",
        action="store_true",
        help=(
            "падать на операции без детерминированной реализации, "
            "а не предупреждать"
        ),
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "прогнать каждый сид N раз в одном процессе и сравнить "
            "действия пошагово — проверка воспроизводимости"
        ),
    )
    args = parser.parse_args()

    if args.deterministic:
        enable_determinism(strict=args.strict_determinism)
        print(
            "детерминированный режим включён "
            f"(CUBLAS_WORKSPACE_CONFIG="
            f"{os.environ.get('CUBLAS_WORKSPACE_CONFIG')})"
        )

    # load_rollout_config, not load_yaml: the rollout config carries only
    # rollout-specific OVERRIDES for grasp_assist; the physics lives in
    # configs/grasp_assist.yaml and has to be merged in.
    cfg = load_rollout_config(args.config)

    seeds = (
        [int(part) for part in args.seeds.split(",") if part.strip()]
        if args.seeds
        else evaluation_seeds(cfg)
    )
    max_steps = args.steps or int(cfg["rollout"]["max_policy_steps"])

    output_dir = Path(
        args.out or (Path(cfg["output"]["directory"]) / "diag")
    )
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    scene = EpisodeScene(
        load_yaml(cfg["configs"]["sim"]),
        cfg["configs"]["randomization"],
    )
    if args.deterministic_render:
        make_render_deterministic(
            scene, disable_shadows=not args.keep_shadows
        )
        print(
            "детерминированный рендер: сглаживание выключено, тени "
            + ("оставлены" if args.keep_shadows else "выключены")
        )

    runner = SmolVLARunner(cfg)
    runner.validate(scene)

    verdicts: list[dict[str, Any]] = []
    repeat_reports: list[dict[str, Any]] = []

    try:
        for seed in seeds:
            baseline_trace: list[np.ndarray] | None = None

            for repetition in range(max(1, args.repeat)):
                label = (
                    ""
                    if args.repeat <= 1
                    else f", повтор {repetition + 1}/{args.repeat}"
                )
                print(
                    f"\n--- прогон seed={seed}, "
                    f"{max_steps} шагов{label} ---"
                )
                verdict, trace = diagnose_seed(
                    scene=scene,
                    runner=runner,
                    cfg=cfg,
                    seed=seed,
                    max_steps=max_steps,
                    output_dir=output_dir,
                    pose_return_threshold=args.pose_return_threshold,
                    pose_return_separation=(
                        args.pose_return_separation
                    ),
                    write_csv=repetition == 0,
                    reach_height=args.reach_height,
                )

                if repetition == 0:
                    baseline_trace = trace
                    verdicts.append(verdict)
                    print_verdict(verdict)
                    continue

                # repetition > 0 достижимо только после нулевого,
                # где baseline_trace уже присвоен.
                assert baseline_trace is not None

                report = compare_action_traces(baseline_trace, trace)
                report["seed"] = seed
                report["repetition"] = repetition + 1
                repeat_reports.append(report)

                if report["identical"]:
                    print(
                        f"  повтор {repetition + 1}: действия "
                        f"совпали на всех "
                        f"{report['compared_steps']} шагах"
                    )
                else:
                    print(
                        f"  повтор {repetition + 1}: РАСХОЖДЕНИЕ "
                        f"с шага {report['first_divergent_step']}, "
                        f"величина {report['first_delta']:.3e}, "
                        f"максимум за эпизод "
                        f"{report['max_delta']:.3e}"
                    )
    finally:
        for renderer in getattr(scene, "_renderers", {}).values():
            renderer.close()

    summary_path = output_dir / "plate_diag_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "deterministic": bool(args.deterministic),
                "repeat": int(args.repeat),
                "episodes": verdicts,
                "repeat_checks": repeat_reports,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if repeat_reports:
        identical = sum(
            1 for r in repeat_reports if r["identical"]
        )
        print()
        print("=== ВОСПРОИЗВОДИМОСТЬ ===")
        print(
            f"совпавших повторов: {identical}/{len(repeat_reports)}"
        )
        for report in repeat_reports:
            if report["identical"]:
                continue
            print(
                f"  seed {report['seed']}: расхождение с шага "
                f"{report['first_divergent_step']}, "
                f"максимум {report['max_delta']:.3e}"
            )

    print()
    print("=== СВОДКА ===")
    print(
        f"{'seed':>6} {'weld':>5} {'open':>5} {'lifted':>6} "
        f"{'placed':>6} {'промах↓':>8} {'3D':>7} {'держ.':>5} {'успех':>6} "
        f"{'ложн.lift':>9} {'ложн.placed':>11}"
    )
    for verdict in verdicts:
        print(
            f"{verdict['seed']:>6} "
            f"{str(verdict['weld_activated_step']):>5} "
            f"{str(verdict['first_open_step']):>5} "
            f"{str(verdict['eval_lifted_step']):>6} "
            f"{str(verdict['eval_placed_step']):>6} "
            f"{(f'{v:.4f}' if (v := verdict['min_xy_at_reach']) is not None else '—'):>8} "
            f"{verdict['min_dist_3d']:>7.4f} "
            f"{verdict['success_hold_best']:>5} "
            f"{('да' if verdict['real_success'] else 'нет'):>6} "
            f"{str(verdict['eval_lift_false_positive']):>9} "
            f"{str(verdict['placed_latched_but_failed']):>11}"
        )

    real = sum(1 for v in verdicts if v["real_success"])
    latched = sum(
        1 for v in verdicts if v["eval_placed_step"] is not None
    )
    lifted_events = sum(
        1 for v in verdicts if v["eval_lifted_step"] is not None
    )
    false_lifts = sum(
        1 for v in verdicts if v["eval_lift_false_positive"]
    )
    print()
    print(
        f"настоящих успехов: {real}/{len(verdicts)}; "
        f"plate_placed записан {latched} раз; "
        f"plate_lifted записан {lifted_events} раз, "
        f"из них ложных {false_lifts}"
    )
    print()
    print(f"сводка: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
