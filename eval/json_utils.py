"""JSON-хелперы: в артефактах eval не должно быть NaN/Infinity.

json.dumps по умолчанию пишет их как голые токены NaN/Infinity, что
не является валидным JSON -- такой episodes.jsonl не читается ни
одним строгим парсером.
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np


def sanitize_json(value: Any) -> Any:
    """Рекурсивно заменяет нечисловые float'ы на None и приводит
    numpy-скаляры и массивы к питоновским типам."""
    if isinstance(value, dict):
        return {key: sanitize_json(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]

    if isinstance(value, np.ndarray):
        return [sanitize_json(item) for item in value.tolist()]

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, bool):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    return value


def dumps_strict(value: Any, **kwargs: Any) -> str:
    """json.dumps, который физически не может выдать NaN/Infinity."""
    return json.dumps(sanitize_json(value), allow_nan=False, **kwargs)