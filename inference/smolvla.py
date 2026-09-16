"""Tensor-only OpenVINO bridge for exports made by ``policy.openvino_export``."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np


class SmolVLAOpenVINOBackend:
    """Apply the checkpoint's LeRobot processors around an OpenVINO sampler."""

    def __init__(self, compiled_model, manifest_path: Path):
        try:
            import torch
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
        except ImportError as exc:  # pragma: no cover - optional hardware path
            raise RuntimeError(
                "SmolVLA OpenVINO inference requires the existing `.[policy,inference]` extras"
            ) from exc
        self._torch = torch
        self._token_key = OBS_LANGUAGE_TOKENS
        self._token_mask_key = OBS_LANGUAGE_ATTENTION_MASK
        self._compiled_model = compiled_model
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise RuntimeError(f"cannot read SmolVLA OpenVINO manifest {manifest_path}: {exc}") from exc
        if manifest.get("format_version") != 1:
            raise RuntimeError(f"unsupported SmolVLA OpenVINO manifest: {manifest_path}")
        self._manifest = manifest
        checkpoint = Path(manifest["checkpoint"])
        if not checkpoint.is_dir():
            raise RuntimeError(f"SmolVLA checkpoint for preprocessing not found: {checkpoint}")
        config = SmolVLAConfig.from_pretrained(str(checkpoint))
        config.device = "cpu"
        vlm_path = manifest.get("vlm_model_path")
        if vlm_path:
            path = Path(vlm_path)
            if not (path / "config.json").is_file():
                raise RuntimeError(f"SmolVLM2 backbone for preprocessing not found: {path}")
            config.vlm_model_name = str(path)
        # Match export: Transformers 5.5's default SDPA mask is not traceable
        # by OpenVINO. Prefix work stays native, but eager attention also keeps
        # the loaded backbone consistent with the exported velocity step.
        from policy.openvino_export import _force_eager_transformers_attention

        with _force_eager_transformers_attention():
            self._policy = SmolVLAPolicy.from_pretrained(str(checkpoint), config=config).to("cpu").eval()
        if self._policy.config.adapt_to_pi_aloha:
            raise RuntimeError("adapt_to_pi_aloha checkpoints are not supported by this OpenVINO bridge")
        self._preprocess, self._postprocess = make_pre_post_processors(
            self._policy.config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": "cpu"}},
        )

    @property
    def action_dim(self) -> int:
        return int(self._policy.config.action_feature.shape[0])

    @property
    def image_shape(self) -> tuple[int, int, int]:
        feature = next(iter(self._policy.config.image_features.values()))
        _, height, width = feature.shape
        return (height, width, 3)

    def _raw_batch(self, instruction: str, observation, robot_state):
        if not isinstance(observation, Mapping):
            raise TypeError("SmolVLA OpenVINO inference requires a mapping of checkpoint feature names to images")
        source = observation.get("images", observation)
        if not isinstance(source, Mapping):
            raise TypeError("observation.images must be a mapping")
        raw = dict(source)
        raw["task"] = instruction
        state_key = self._manifest["state_feature_key"]
        if state_key not in raw:
            if isinstance(robot_state, Mapping):
                value = robot_state.get(state_key, robot_state.get("joint_positions"))
            else:
                value = robot_state
            if value is None:
                raise ValueError(f"robot_state must provide {state_key!r} or 'joint_positions'")
            raw[state_key] = np.asarray(value, dtype=np.float32)
        return self._preprocess(raw)

    def predict(self, instruction: str, observation, robot_state=None) -> np.ndarray:
        batch = self._raw_batch(instruction, observation, robot_state)
        images, image_masks = self._policy.prepare_images(batch)
        state = self._policy.prepare_state(batch)
        noise = self._torch.randn(
            (state.shape[0], self._manifest["chunk_size"], self._manifest["max_action_dim"]),
            dtype=self._torch.float32,
        )
        prefix, prefix_pad, prefix_att = self._policy.model.embed_prefix(
            images, image_masks, batch[self._token_key], batch[self._token_mask_key].bool(), state
        )
        x_t = noise
        for index in range(self._manifest["num_steps"]):
            timestep = self._torch.full(
                (x_t.shape[0],), 1.0 - index / self._manifest["num_steps"], dtype=self._torch.float32
            )
            values = (prefix, prefix_pad, prefix_att, x_t, timestep)
            result = self._compiled_model({
                self._compiled_model.input(input_index): value.detach().cpu().numpy()
                for input_index, value in enumerate(values)
            })
            velocity = next(iter(result.values()))
            x_t = x_t - self._torch.from_numpy(velocity) / self._manifest["num_steps"]
        actions = x_t.detach().cpu().numpy()
        raw_action = self._torch.from_numpy(actions[:, :, : self.action_dim])
        action = self._postprocess(raw_action)
        # SmolVLA predicts its fixed internal chunk (50 for this checkpoint).
        # ``InferenceRuntime`` owns the public chunking contract, so one backend
        # call must yield exactly one action rather than leaking that internal
        # horizon to callers.
        return np.asarray(action.detach().cpu().numpy()[0, 0], dtype=np.float32)
