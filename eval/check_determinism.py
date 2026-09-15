"""Развести источники недетерминизма в eval: сцена, рендер, политика.

Полный прогон расходится с шага 0, но причин может быть три, и лечатся
они по-разному. Скрипт проверяет каждую по отдельности, без роллаутов.

A. Сброс сцены. Два подряд `scene.reset(seed)` должны давать
   побитово одинаковые qpos/qvel. Если нет — виновата утечка
   состояния между эпизодами.

B. Рендер. Два подряд `scene.observe()` на одном и том же `data`
   должны давать одинаковые пиксели. Если нет — недетерминирован
   MuJoCo-рендер, и политика получает разный вход при одной сцене.

C. Политика. Один и тот же observation после `runner.reset(seed)`
   должен давать одинаковое действие. Если нет — недетерминированы
   CUDA-ядра. Отдельно проверяется, меняется ли действие без
   переseed-а: это показывает, тянет ли SmolVLA шум из RNG.

Запуск из корня репозитория:

    python eval/check_determinism.py --seeds 0,3,6
    python eval/check_determinism.py --seeds 0,3,6 --deterministic
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mujoco  # noqa: E402

from eval.diagnose_plate import (  # noqa: E402
    enable_determinism,
    make_render_deterministic,
)
from eval.eval_smolvla import (  # noqa: E402
    load_rollout_config,
    SmolVLARunner,
    load_yaml,
)
from eval.scene import EpisodeScene  # noqa: E402


def clear_equalities(model: Any, data: Any, cfg: dict[str, Any]) -> None:
    """Погасить weld/connect И в data, И в model.eq_active0.

    mj_resetData восстанавливает data.eq_active из model.eq_active0,
    поэтому активная связь, оставшаяся от прошлого эпизода, переживает
    сброс сцены и участвует в settle следующего сида.
    """
    for section in ("drawer", "plate"):
        name = cfg["grasp_assist"][section]["equality_name"]
        eq_id = model.equality(name).id
        data.eq_active[eq_id] = 0
        model.eq_active0[eq_id] = 0


def state_of(data: Any) -> tuple[np.ndarray, np.ndarray]:
    return data.qpos.copy(), data.qvel.copy()


def image_digest(images: dict[str, np.ndarray]) -> dict[str, str]:
    return {
        name: hashlib.sha256(
            np.ascontiguousarray(array).tobytes()
        ).hexdigest()[:16]
        for name, array in sorted(images.items())
    }


def check_seed(
    scene: EpisodeScene,
    runner: SmolVLARunner,
    cfg: dict[str, Any],
    seed: int,
    clear_first: bool,
) -> dict[str, Any]:
    instruction = " ".join(cfg["task"]["instruction"].split())
    result: dict[str, Any] = {"seed": seed}

    # --- A. сброс сцены -------------------------------------------
    if clear_first:
        clear_equalities(scene.model, scene.data, cfg)
    scene.reset(seed)
    qpos_1, qvel_1 = state_of(scene.data)
    observation_1 = scene.observe()

    if clear_first:
        clear_equalities(scene.model, scene.data, cfg)
    scene.reset(seed)
    qpos_2, qvel_2 = state_of(scene.data)

    result["qpos_delta"] = float(np.max(np.abs(qpos_1 - qpos_2)))
    result["qvel_delta"] = float(np.max(np.abs(qvel_1 - qvel_2)))
    result["scene_identical"] = (
        result["qpos_delta"] == 0.0 and result["qvel_delta"] == 0.0
    )

    # --- B. рендер на одном и том же data --------------------------
    observation_2 = scene.observe()
    observation_3 = scene.observe()
    digest_2 = image_digest(observation_2["images"])
    digest_3 = image_digest(observation_3["images"])

    result["render_identical"] = digest_2 == digest_3
    result["render_mismatch"] = [
        name for name in digest_2 if digest_2[name] != digest_3[name]
    ]

    pixel_deltas = {
        name: int(
            np.max(
                np.abs(
                    observation_2["images"][name].astype(np.int32)
                    - observation_3["images"][name].astype(np.int32)
                )
            )
        )
        for name in observation_2["images"]
    }
    result["render_max_pixel_delta"] = max(pixel_deltas.values())

    # --- C. политика на ОДНОМ И ТОМ ЖЕ observation -----------------
    runner.reset(seed)
    action_1 = np.asarray(
        runner.predict(instruction, observation_1), dtype=np.float64
    )

    runner.reset(seed)
    action_2 = np.asarray(
        runner.predict(instruction, observation_1), dtype=np.float64
    )

    result["policy_delta_same_seed"] = float(
        np.max(np.abs(action_1 - action_2))
    )
    result["policy_identical"] = (
        result["policy_delta_same_seed"] == 0.0
    )

    # Очистить очередь чанка, НЕ переseed-ив RNG: если действие
    # изменилось, значит SmolVLA берёт шум из RNG и переseed в
    # reset() действительно нужен.
    runner.policy.reset()
    action_3 = np.asarray(
        runner.predict(instruction, observation_1), dtype=np.float64
    )
    result["policy_delta_no_reseed"] = float(
        np.max(np.abs(action_1 - action_3))
    )

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Раздельная проверка источников недетерминизма"
    )
    parser.add_argument(
        "--config", default="configs/smolvla_rollout.yaml"
    )
    parser.add_argument("--seeds", default="0,3,6")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="включить детерминированные ядра перед проверкой",
    )
    parser.add_argument(
        "--deterministic-render",
        action="store_true",
        help="выключить сглаживание и тени в офскрин-рендере",
    )
    parser.add_argument(
        "--keep-shadows",
        action="store_true",
        help=(
            "выключить только сглаживание, тени оставить — меньше "
            "расхождение с картинкой, на которой обучалась политика"
        ),
    )
    parser.add_argument(
        "--no-clear-equalities",
        action="store_true",
        help=(
            "НЕ гасить weld перед сбросом — воспроизводит текущее "
            "поведение eval со всеми его утечками"
        ),
    )
    args = parser.parse_args()

    if args.deterministic:
        enable_determinism()
        print("детерминированный режим включён")

    # load_rollout_config, не load_yaml: физика захвата живёт во внешнем
    # configs/grasp_assist.yaml, а rollout-конфиг несёт только
    # переопределения. С load_yaml сюда приходит grasp_assist: {} и
    # clear_equalities падает на KeyError: 'drawer'.
    cfg = load_rollout_config(args.config)
    seeds = [
        int(part) for part in args.seeds.split(",") if part.strip()
    ]

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

    results = []
    try:
        for seed in seeds:
            print(f"\n=== seed {seed} ===")
            result = check_seed(
                scene,
                runner,
                cfg,
                seed,
                clear_first=not args.no_clear_equalities,
            )
            results.append(result)

            print(
                f"A. сцена: "
                f"{'совпала' if result['scene_identical'] else 'РАЗОШЛАСЬ'}"
                f"  (qpos Δ={result['qpos_delta']:.3e}, "
                f"qvel Δ={result['qvel_delta']:.3e})"
            )
            print(
                f"B. рендер: "
                f"{'совпал' if result['render_identical'] else 'РАЗОШЁЛСЯ'}"
                f"  (макс. разница пикселя "
                f"{result['render_max_pixel_delta']})"
            )
            if result["render_mismatch"]:
                print(f"   камеры: {result['render_mismatch']}")
            print(
                f"C. политика: "
                f"{'совпала' if result['policy_identical'] else 'РАЗОШЛАСЬ'}"
                f"  (Δ при том же seed="
                f"{result['policy_delta_same_seed']:.3e}, "
                f"Δ без переseed="
                f"{result['policy_delta_no_reseed']:.3e})"
            )
    finally:
        for renderer in getattr(scene, "_renderers", {}).values():
            renderer.close()

    print()
    print("=== ВЫВОД ===")
    scene_bad = [r["seed"] for r in results if not r["scene_identical"]]
    render_bad = [
        r["seed"] for r in results if not r["render_identical"]
    ]
    policy_bad = [
        r["seed"] for r in results if not r["policy_identical"]
    ]

    if scene_bad:
        print(f"сцена недетерминирована на сидах {scene_bad}")
    if render_bad:
        print(f"рендер недетерминирован на сидах {render_bad}")
    if policy_bad:
        print(f"политика недетерминирована на сидах {policy_bad}")
    if not (scene_bad or render_bad or policy_bad):
        print(
            "все три компонента детерминированы ПРИ ЭТИХ настройках. "
            "Это не значит, что полный прогон воспроизводим: остаётся "
            "утечка состояния между эпизодами (model.eq_active0). "
            "Проверять через diagnose_plate.py --repeat 2 с теми же "
            "флагами рендера."
        )

    reseed_matters = [
        r["seed"]
        for r in results
        if r["policy_delta_no_reseed"] > 0.0
    ]
    if reseed_matters:
        print(
            f"SmolVLA берёт шум из RNG (действие меняется без "
            f"переseed) на сидах {reseed_matters} — переseed в "
            f"runner.reset обязателен"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
