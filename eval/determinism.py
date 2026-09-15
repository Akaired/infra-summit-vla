"""Устранение источников недетерминизма в eval.

Вынесено из eval/diagnose_plate.py, чтобы главный харнесс
(eval/eval_smolvla.py) мог включать то же самое, а /eval оставался
независимо запускаемым модулем (CONTRIBUTING.md §3).

Замерено eval/check_determinism.py: доминирующий источник — рендер
(±1 младший бит на всех трёх камерах), а не CUDA-ядра (~1e-7).
Поэтому make_render_deterministic включается по умолчанию, а
enable_determinism — по требованию.
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np


def enable_determinism(
    seed: int = 0,
    strict: bool = False,
    disable_tf32: bool = True,
) -> None:
    """Убрать источники недетерминизма в инференсе.

    SmolVLARunner.reset уже сеет torch и cuda на каждом эпизоде, так
    что RNG не при чём. Остаются недетерминированные CUDA-ядра
    (атомарные редукции, автоподбор алгоритмов cuDNN) — расхождение
    на уровне 1e-7 за 600 шагов замкнутого контура превращается в
    другой исход эпизода.
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Автоподбор алгоритма по таймингам даёт разный выбор от запуска
    # к запуску в зависимости от загрузки GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # warn_only=True: если у какой-то операции нет детерминированной
    # реализации, прогон не падает, а печатает предупреждение —
    # по нему видно, что именно мешает.
    torch.use_deterministic_algorithms(True, warn_only=not strict)


def make_render_deterministic(
    scene: Any,
    disable_msaa: bool = True,
    disable_shadows: bool = True,
) -> None:
    """Убрать недетерминизм офскрин-рендера MuJoCo.

    Замерено: два подряд render() одной и той же сцены расходятся на
    ±1 младший бит на всех трёх камерах. При pixel_divisor=255 это
    возмущение входа ~0.004, которое даёт ~1e-3 на действии и за 600
    шагов замкнутого контура разносит эпизод.

    Сглаживание (offsamples) читается при СОЗДАНИИ GL-контекста,
    поэтому рендереры пересоздаются. Тени — флаг сцены, он ставится
    на уже созданном рендерере.
    """
    import mujoco

    if disable_msaa:
        scene.model.vis.quality.offsamples = 0

        for renderer in scene._renderers.values():
            renderer.close()

        scene._renderers = {
            camera["name"]: mujoco.Renderer(
                scene.model,
                height=int(camera["height"]),
                width=int(camera["width"]),
            )
            for camera in scene._cameras
        }

    if disable_shadows:
        # mujoco не поставляет .pyi и py.typed, поэтому Pylance не
        # видит enum'ы из C-расширения и ругается на mjtRndFlag.
        # В рантайме он есть (проверено на mujoco 3.13).
        shadow_flag = mujoco.mjtRndFlag.mjRND_SHADOW  # type: ignore[attr-defined]

        for renderer in scene._renderers.values():
            renderer.scene.flags[shadow_flag] = 0
