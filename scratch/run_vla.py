"""Запуск обученной VLA одним эпизодом с произвольной инструкцией.

Точка интеграции для ASR (Speechmatics): распознанный текст подаётся как
строка, без правок в модельном тракте — `SmolVLARunner.predict` уже
принимает инструкцию параметром на каждый вызов.

ВАЖНО, прочитать до подключения ASR
-----------------------------------
Датасет demo-100 содержит РОВНО ОДНУ инструкцию (`meta/tasks.parquet`:
`task_index: [0]`, одна строка на все 100 эпизодов):

    "Open the drawer, close the drawer, pick up the plate, and place the
     plate in the center of the table."

Модель никогда не видела другого текста. Языковой энкодер обучался на одном
предложении, поэтому текст не несёт различающего сигнала: подача любой
другой фразы НЕ изменит поведение робота осмысленным образом. Он выполнит
ту же самую последовательность.

Это значит:
  * интеграцию ASR можно и нужно отлаживать уже сейчас — тракт рабочий;
  * но демонстрировать «робот слушается голоса» на этом чекпоинте нельзя;
  * для реального языкового управления нужен датасет с несколькими
    задачами и переобучение.

Скрипт печатает предупреждение, если поданная инструкция отличается от
обучающей.

Что внутри
----------
Ничего не реализуется заново: используется тот же `run_episode()`, что и
`eval/eval_smolvla.py`, поэтому поведение совпадает с eval шаг в шаг,
включая grasp assist, критерии успеха, запись видео и детерминизм.

Примеры
-------
    uv run python scratch\\run_vla.py --instruction "Open the drawer, ..."
    uv run python scratch\\run_vla.py --seed 3 --instruction "..."

Из своего кода:
    from scratch.run_vla import VLASession
    session = VLASession("configs/smolvla_rollout.yaml")   # модель грузится один раз
    record = session.run(asr_text, seed=0)
    print(record["success"], record["event_steps"])
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "sim"))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

import imageio_ffmpeg  # noqa: E402
import mediapy  # noqa: E402

from eval.determinism import (  # noqa: E402
    enable_determinism,
    make_render_deterministic,
)
from eval.eval_smolvla import (  # noqa: E402
    SmolVLARunner,
    load_rollout_config,
    load_yaml,
    resolve_path,
    run_episode,
    select_output_directory,
)
from eval.json_utils import dumps_strict  # noqa: E402
from eval.scene import EpisodeScene  # noqa: E402

# meta/tasks.parquet датасета demo-100, task_index 0 -- единственная задача.
TRAINING_INSTRUCTION = (
    "Open the drawer, close the drawer, pick up the plate, and place the "
    "plate in the center of the table."
)


class VLASession:
    """Загруженная модель плюс сцена. Держать одну на процесс.

    Загрузка SmolVLM2-500M занимает секунды, поэтому конструктор вызывается
    один раз, а `run()` — сколько угодно раз.
    """

    def __init__(self, config_path: str = "configs/smolvla_rollout.yaml") -> None:
        self.cfg = load_rollout_config(config_path)

        rollout_cfg = self.cfg.get("rollout", {})
        if bool(rollout_cfg.get("deterministic_kernels", False)):
            enable_determinism()

        if bool(self.cfg["output"]["video"]["enabled"]):
            mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())

        self.scene = EpisodeScene(
            load_yaml(self.cfg["configs"]["sim"]),
            self.cfg["configs"]["randomization"],
        )

        if bool(rollout_cfg.get("deterministic_render", True)):
            make_render_deterministic(self.scene)

        self.runner = SmolVLARunner(self.cfg)
        self.runner.validate(self.scene)

        print(f"Checkpoint: {resolve_path(self.cfg['model']['checkpoint'])}")
        print("Готово к приёму инструкций.")

    # ------------------------------------------------------------------ #
    def run(
        self,
        instruction: str,
        seed: int = 0,
        output_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Один эпизод с поданной инструкцией. Возвращает запись эпизода."""
        instruction = " ".join(str(instruction).split())

        if not instruction:
            raise ValueError("Инструкция пустая")

        if instruction != TRAINING_INSTRUCTION:
            print()
            print("!" * 70)
            print("ВНИМАНИЕ: инструкция отличается от единственной обучающей.")
            print(f"  подана:   {instruction}")
            print(f"  обучающая: {TRAINING_INSTRUCTION}")
            print("Датасет demo-100 содержит ровно одну задачу, поэтому текст")
            print("не влияет на поведение. Робот выполнит ту же самую")
            print("последовательность. Тракт ASR при этом проверяется корректно.")
            print("!" * 70)
            print()

        # run_episode берёт инструкцию из cfg, поэтому подменяем в копии:
        # так используется ровно тот же цикл, что в eval, без дублирования.
        episode_cfg = copy.deepcopy(self.cfg)
        episode_cfg["task"]["instruction"] = instruction

        if output_dir is None:
            output_dir = select_output_directory(
                episode_cfg["output"], [seed]
            )

        record = run_episode(
            scene=self.scene,
            runner=self.runner,
            cfg=episode_cfg,
            seed=seed,
            output_dir=output_dir,
        )
        record["instruction"] = instruction
        record["seed"] = seed

        print()
        print(f"seed={seed}: {'SUCCESS' if record['success'] else 'FAILURE'}")
        print(f"events: {record['event_steps']}")
        print(
            "перемещение тарелки: "
            f"{record['plate_xy_displacement_m_final']:.4f} м "
            f"(порог {episode_cfg['evaluation']['plate']['placed_min_xy_displacement_m']})"
        )

        return record


# ---------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smolvla_rollout.yaml")
    parser.add_argument(
        "--instruction",
        default=TRAINING_INSTRUCTION,
        help="текст инструкции; по умолчанию — обучающая",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        default=None,
        help="каталог вывода; по умолчанию свежий под output.directory/runs",
    )
    args = parser.parse_args()

    session = VLASession(args.config)
    record = session.run(
        args.instruction,
        seed=args.seed,
        output_dir=Path(args.out) if args.out else None,
    )

    print()
    print(dumps_strict(record, indent=2, ensure_ascii=False))
    return 0 if record["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
