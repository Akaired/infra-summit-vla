from __future__ import annotations

import imageio_ffmpeg
import mediapy

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import mediapy
import mujoco
import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.scene import EpisodeScene
from sim.grasp_assist import GraspAssist
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with resolve_path(str(path)).open("r", encoding="utf-8") as stream:
        result = yaml.safe_load(stream)

    if not isinstance(result, dict):
        raise ValueError(f"Config must contain a mapping: {path}")

    return result


def nested_value(mapping: dict[str, Any], dotted_key: str) -> Any:
    value: Any = mapping

    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"Config key not found: {dotted_key}")
        value = value[key]

    return value


def compiled_actuator_names(model: Any) -> list[str]:
    names: list[str] = []

    for actuator_id in range(model.nu):
        name = model.actuator(actuator_id).name

        if not name:
            raise ValueError(f"Actuator {actuator_id} has no name")

        names.append(str(name))

    return names


class SmolVLARunner:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        model_cfg = cfg["model"]

        self.device = torch.device(model_cfg["device"])
        self.checkpoint = resolve_path(model_cfg["checkpoint"])

        if not self.checkpoint.is_dir():
            raise FileNotFoundError(
                f"Checkpoint directory does not exist: {self.checkpoint}"
            )

        self.policy = SmolVLAPolicy.from_pretrained(
            str(self.checkpoint),
            device=str(self.device),
        )
        n_action_steps = int(model_cfg["n_action_steps"])
        chunk_size = int(self.policy.config.chunk_size)

        if not 1 <= n_action_steps <= chunk_size:
            raise ValueError(
                "model.n_action_steps must be between 1 and "
                f"checkpoint chunk_size={chunk_size}, got {n_action_steps}"
            )

        self.policy.config.n_action_steps = n_action_steps

        print(
            f"Action chunk: generated={chunk_size}, "
            f"executed_before_replan={n_action_steps}"
        )
        
        self.policy.eval()
        input_features = self.policy.config.input_features
        output_features = self.policy.config.output_features

        if input_features is None:
            raise ValueError("Checkpoint has no input_features")

        if output_features is None:
            raise ValueError("Checkpoint has no output_features")

        self.input_features: dict[str, Any] = dict(input_features)
        self.output_features: dict[str, Any] = dict(output_features)

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=str(self.checkpoint),
            preprocessor_overrides={
                "device_processor": {"device": str(self.device)}
            },
        )

        self.state_cfg = cfg["observation"]["state"]
        self.image_cfg = cfg["observation"]
        self.camera_cfgs = cfg["observation"]["cameras"]
        self.action_cfg = cfg["action"]
        self.task_cfg = cfg["task"]

    def validate(self, scene: EpisodeScene) -> None:
        input_features = self.input_features
        output_features = self.output_features

        state_key = self.state_cfg["feature_key"]
        state_size = int(self.state_cfg["size"])

        if state_key not in input_features:
            raise KeyError(
                f"Checkpoint does not contain state feature {state_key!r}"
            )

        checkpoint_state_shape = tuple(input_features[state_key].shape)
        if checkpoint_state_shape != (state_size,):
            raise ValueError(
                "State mismatch: config requests "
                f"{state_size}, checkpoint expects {checkpoint_state_shape}"
            )

        configured_image_keys = {
            camera["feature_key"] for camera in self.camera_cfgs
        }
        checkpoint_image_keys = {
            key
            for key, feature in input_features.items()
            if len(tuple(feature.shape)) == 3
        }

        if configured_image_keys != checkpoint_image_keys:
            raise ValueError(
                "Camera feature mismatch.\n"
                f"Config:     {sorted(configured_image_keys)}\n"
                f"Checkpoint: {sorted(checkpoint_image_keys)}"
            )

        action_key = self.action_cfg["feature_key"]
        action_size = int(self.action_cfg["size"])

        if action_key not in output_features:
            raise KeyError(
                f"Checkpoint does not contain action feature {action_key!r}"
            )

        checkpoint_action_shape = tuple(output_features[action_key].shape)
        if checkpoint_action_shape != (action_size,):
            raise ValueError(
                "Action mismatch: config requests "
                f"{action_size}, checkpoint expects {checkpoint_action_shape}"
            )

        if scene.model.nu != action_size:
            raise ValueError(
                f"MJCF has {scene.model.nu} actuators, "
                f"but config expects {action_size}"
            )

        actual_names = compiled_actuator_names(scene.model)
        expected_names = list(self.action_cfg["actuator_names"])

        if actual_names != expected_names:
            raise ValueError(
                "Actuator order mismatch.\n"
                f"Config: {expected_names}\n"
                f"MJCF:   {actual_names}"
            )

    def reset(self, environment_seed: int) -> None:
        seed = environment_seed + int(
            self.cfg["model"]["torch_seed_offset"]
        )
        torch.manual_seed(seed)

        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        self.policy.reset()

        if hasattr(self.preprocessor, "reset"):
            self.preprocessor.reset()

        if hasattr(self.postprocessor, "reset"):
            self.postprocessor.reset()

    def _state_tensor(self, observation: dict[str, Any]) -> torch.Tensor:
        source_key = self.state_cfg["source_key"]
        start = int(self.state_cfg["start"])
        size = int(self.state_cfg["size"])

        source = np.asarray(observation[source_key], dtype=np.float32)
        state = source[start : start + size]

        if state.shape != (size,):
            raise ValueError(
                f"State slice must be ({size},), got {state.shape}"
            )

        return torch.from_numpy(np.ascontiguousarray(state))

    def _image_tensor(
        self,
        image: np.ndarray,
        feature_key: str,
    ) -> torch.Tensor:
        image = np.asarray(image)

        if image.ndim != 3:
            raise ValueError(
                f"Image {feature_key!r} must be HWC, got {image.shape}"
            )

        tensor = torch.from_numpy(
            np.ascontiguousarray(image)
        ).permute(2, 0, 1).float()

        tensor = tensor / float(self.image_cfg["pixel_divisor"])

        expected_shape = tuple(
            self.input_features[feature_key].shape
        )
        expected_channels, expected_height, expected_width = expected_shape

        if tensor.shape[0] != expected_channels:
            raise ValueError(
                f"Image {feature_key!r} has {tensor.shape[0]} channels, "
                f"checkpoint expects {expected_channels}"
            )

        if tuple(tensor.shape[1:]) != (
            expected_height,
            expected_width,
        ):
            resize_cfg = self.image_cfg["resize"]
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(expected_height, expected_width),
                mode=resize_cfg["mode"],
                align_corners=bool(resize_cfg["align_corners"]),
                antialias=bool(resize_cfg["antialias"]),
            ).squeeze(0)

        return tensor

    def predict(
        self,
        instruction: str,
        observation: dict[str, Any],
    ) -> np.ndarray:
        batch: dict[str, Any] = {
            self.state_cfg["feature_key"]: self._state_tensor(observation),
            self.task_cfg["task_feature_key"]: instruction,
        }

        images = observation[self.image_cfg["images_source_key"]]

        for camera_cfg in self.camera_cfgs:
            source_name = camera_cfg["source_name"]
            feature_key = camera_cfg["feature_key"]

            if source_name not in images:
                raise KeyError(
                    f"Camera {source_name!r} is missing. "
                    f"Available cameras: {sorted(images)}"
                )

            batch[feature_key] = self._image_tensor(
                images[source_name],
                feature_key,
            )

        processed = self.preprocessor(batch)

        with torch.inference_mode():
            action = self.policy.select_action(processed)

        action = self.postprocessor(action)
        result = action.detach().cpu().numpy().reshape(-1)

        expected_size = int(self.action_cfg["size"])
        if result.shape != (expected_size,):
            raise ValueError(
                f"Policy action must be ({expected_size},), "
                f"got {result.shape}"
            )

        return result.astype(np.float32)



class TaskTracker:
    def __init__(
        self,
        scene: EpisodeScene,
        cfg: dict[str, Any],
    ) -> None:
        self.scene = scene
        self.cfg = cfg
        self.event_names = cfg["events"]
        self.sequence = list(cfg["sequence"])
        self.events: dict[str, int | None] = {
        name: None for name in self.sequence
        }

        drawer_cfg = cfg["drawer"]
        self.drawer_low, self.drawer_high = scene.joint_range(
            drawer_cfg["joint_name"]
        )

        self.initial_plate_position = scene.body_xpos(
            cfg["plate"]["body_name"]
        )

    def _drawer_fraction(self) -> float:
        drawer_cfg = self.cfg["drawer"]
        position = self.scene.joint_qpos(drawer_cfg["joint_name"])
        travel = self.drawer_high - self.drawer_low

        if travel <= 0:
            raise ValueError("Drawer joint range must be positive")

        return (position - self.drawer_low) / travel

    def _plate_values(self) -> tuple[np.ndarray, float]:
        plate_cfg = self.cfg["plate"]
        position = self.scene.body_xpos(plate_cfg["body_name"])
        speed = self.scene.body_linear_speed(plate_cfg["body_name"])
        return position, speed

    def update(self, policy_step: int) -> None:
        names = self.event_names
        drawer_fraction = self._drawer_fraction()
        plate_position, plate_speed = self._plate_values()

        opened = names["drawer_opened"]
        closed = names["drawer_closed"]
        lifted = names["plate_lifted"]
        placed = names["plate_placed"]

        if (
            self.events[opened] is None
            and drawer_fraction
            >= float(
                self.cfg["drawer"]["opened_travel_fraction"]
            )
        ):
            self.events[opened] = policy_step

        if (
            self.events[opened] is not None
            and self.events[closed] is None
            and drawer_fraction
            <= float(
                self.cfg["drawer"]["closed_travel_fraction"]
            )
        ):
            self.events[closed] = policy_step

        lifted_delta = (
            plate_position[2] - self.initial_plate_position[2]
        )
        if (
            self.events[closed] is not None
            and self.events[lifted] is None
            and lifted_delta
            >= float(self.cfg["plate"]["lifted_delta_z_m"])
        ):
            self.events[lifted] = policy_step

        target_xy = np.asarray(
            self.cfg["plate"]["target_xy_m"],
            dtype=np.float64,
        )
        target_distance = float(
            np.linalg.norm(plate_position[:2] - target_xy)
        )
        surface_error = abs(
            float(
                plate_position[2]
                - self.initial_plate_position[2]
            )
        )

        plate_is_placed = (
            target_distance
            <= float(self.cfg["plate"]["target_radius_m"])
            and surface_error
            <= float(
                self.cfg["plate"]["surface_z_tolerance_m"]
            )
            and plate_speed
            <= float(
                self.cfg["plate"]["resting_linear_speed_mps"]
            )
        )

        if (
            self.events[lifted] is not None
            and self.events[placed] is None
            and plate_is_placed
        ):
            self.events[placed] = policy_step

    def currently_successful(self) -> bool:
        if any(self.events[name] is None for name in self.sequence):
            return False

        drawer_fraction = self._drawer_fraction()
        plate_position, plate_speed = self._plate_values()
        target_xy = np.asarray(
            self.cfg["plate"]["target_xy_m"],
            dtype=np.float64,
        )

        target_distance = float(
            np.linalg.norm(plate_position[:2] - target_xy)
        )
        surface_error = abs(
            float(
                plate_position[2]
                - self.initial_plate_position[2]
            )
        )

        return (
            drawer_fraction
            <= float(
                self.cfg["drawer"]["closed_travel_fraction"]
            )
            and target_distance
            <= float(self.cfg["plate"]["target_radius_m"])
            and surface_error
            <= float(
                self.cfg["plate"]["surface_z_tolerance_m"]
            )
            and plate_speed
            <= float(
                self.cfg["plate"]["resting_linear_speed_mps"]
            )
        )

    def report(self) -> dict[str, Any]:
        plate_position, plate_speed = self._plate_values()
        target_xy = np.asarray(
            self.cfg["plate"]["target_xy_m"],
            dtype=np.float64,
        )

        return {
            "success": self.currently_successful(),
            "event_steps": self.events,
            "completed_events": [
                name
                for name in self.sequence
                if self.events[name] is not None
            ],
            "drawer_travel_fraction_final": self._drawer_fraction(),
            "plate_position_initial": self.initial_plate_position.tolist(),
            "plate_position_final": plate_position.tolist(),
            "plate_target_distance_m_final": float(
                np.linalg.norm(plate_position[:2] - target_xy)
            ),
            "plate_linear_speed_mps_final": plate_speed,
        }


def evaluation_seeds(cfg: dict[str, Any]) -> list[int]:
    seed_cfg = cfg["evaluation"]["seeds"]
    source = load_yaml(seed_cfg["source_config"])
    seeds = nested_value(source, seed_cfg["source_key"])

    if not isinstance(seeds, list) or not seeds:
        raise ValueError("Evaluation seed list must be a non-empty list")

    return [int(seed) for seed in seeds]


def save_video(
    frames: list[np.ndarray],
    output_dir: Path,
    seed: int,
    cfg: dict[str, Any],
) -> str | None:
    video_cfg = cfg["output"]["video"]

    if not bool(video_cfg["enabled"]) or not frames:
        return None

    filename = video_cfg["filename_pattern"].format(seed=seed)
    path = output_dir / filename
    
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    mediapy.set_ffmpeg(ffmpeg_path)

    mediapy.write_video(
        str(path),
        frames,
        fps=int(video_cfg["fps"]),
    )

    return str(path.relative_to(REPO_ROOT))


def run_episode(
    scene: EpisodeScene,
    runner: SmolVLARunner,
    cfg: dict[str, Any],
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    started_at = time.perf_counter()

    scene.reset(seed)
    runner.reset(seed)

    grasp = GraspAssist(
        model=scene.model,
        data=scene.data,
        cfg=cfg["grasp_assist"],
    )
    grasp.reset()

    tracker = TaskTracker(scene, cfg["evaluation"])
    rollout_cfg = cfg["rollout"]
    video_cfg = cfg["output"]["video"]

    instruction = " ".join(cfg["task"]["instruction"].split())
    action_repeat = int(rollout_cfg["action_repeat"])
    max_policy_steps = int(rollout_cfg["max_policy_steps"])
    progress_every = int(
        rollout_cfg["progress_every_policy_steps"]
    )
    video_every = int(video_cfg["every_n_policy_steps"])
    required_hold = int(
        rollout_cfg["success_hold_policy_steps"]
    )

    frames: list[np.ndarray] = []
    successful_steps = 0
    policy_steps = 0

    observation = scene.observe()

    for policy_step in range(max_policy_steps):
        action = runner.predict(instruction, observation)

        if bool(cfg["action"]["clip_to_actuator_range"]):
            low = scene.model.actuator_ctrlrange[:, 0]
            high = scene.model.actuator_ctrlrange[:, 1]
            action = np.clip(action, low, high)

        scene.apply_action(action)
        grasp.update(action, policy_step)

        for _ in range(action_repeat):
            scene.step()

        observation = scene.observe()
        tracker.update(policy_step)
        policy_steps = policy_step + 1

        if (
            bool(video_cfg["enabled"])
            and policy_step % video_every == 0
        ):
            frames.append(
                observation["images"][video_cfg["camera"]].copy()
            )

        if tracker.currently_successful():
            successful_steps += 1
        else:
            successful_steps = 0

        if progress_every > 0 and policy_steps % progress_every == 0:
            completed = len(
                tracker.report()["completed_events"]
            )
            total = len(cfg["evaluation"]["sequence"])
            print(
                f"seed={seed} step={policy_steps}/{max_policy_steps} "
                f"events={completed}/{total}"
            )

        if (
            bool(rollout_cfg["stop_on_success"])
            and successful_steps >= required_hold
        ):
            break

    report = tracker.report()
    report.update(
        {
            "seed": seed,
            "instruction": instruction,
            "policy_steps": policy_steps,
            "physics_steps": (
                policy_steps
                * action_repeat
                * scene.control_decimation
            ),
            "wall_time_s": time.perf_counter() - started_at,
            "video_path": save_video(
                frames,
                output_dir,
                seed,
                cfg,
            ),
        }
    )

    return report


def parse_seed_override(raw: str | None) -> list[int] | None:
    if raw is None:
        return None

    return [
        int(value.strip())
        for value in raw.split(",")
        if value.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if bool(cfg["output"]["video"]["enabled"]):
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        mediapy.set_ffmpeg(ffmpeg_path)
        print("FFmpeg:", ffmpeg_path)

    scene = EpisodeScene(
        load_yaml(cfg["configs"]["sim"]),
        cfg["configs"]["randomization"],
    )
    runner = SmolVLARunner(cfg)
    runner.validate(scene)

    print("Configuration, scene and checkpoint are compatible.")
    
    state_feature_key = cfg["observation"]["state"]["feature_key"]

    print(
        "State shape:",
        tuple(runner.input_features[state_feature_key].shape),
    )
    
    print("Actuators:", compiled_actuator_names(scene.model))

    if args.check_only:
        return 0

    seeds = (
        parse_seed_override(args.seeds)
        if args.seeds is not None
        else evaluation_seeds(cfg)
    )

    if not seeds:
        raise ValueError("No evaluation seeds selected")

    output_cfg = cfg["output"]
    output_dir = resolve_path(output_cfg["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes_path = output_dir / output_cfg["episodes_filename"]
    summary_path = output_dir / output_cfg["summary_filename"]

    if episodes_path.exists() and not bool(output_cfg["overwrite"]):
        raise FileExistsError(
            f"Output already exists: {episodes_path}"
        )

    records: list[dict[str, Any]] = []

    with episodes_path.open("w", encoding="utf-8") as stream:
        for seed in seeds:
            print(f"\nStarting evaluation seed {seed}")

            record = run_episode(
                scene=scene,
                runner=runner,
                cfg=cfg,
                seed=seed,
                output_dir=output_dir,
            )
            records.append(record)

            stream.write(json.dumps(record) + "\n")
            stream.flush()

            print(
                f"seed={seed}: "
                f"{'SUCCESS' if record['success'] else 'FAILURE'}"
            )
            print("events:", record["event_steps"])
            print("video:", record["video_path"])

    success_count = sum(
        int(record["success"]) for record in records
    )

    summary = {
        "checkpoint": cfg["model"]["checkpoint"],
        "instruction": " ".join(
            cfg["task"]["instruction"].split()
        ),
        "seeds": seeds,
        "episodes": len(records),
        "successes": success_count,
        "success_ratio": f"{success_count}/{len(records)}",
        "per_event_completed": {
            event: sum(
                record["event_steps"][event] is not None
                for record in records
            )
            for event in cfg["evaluation"]["sequence"]
        },
        "episodes_file": str(episodes_path.relative_to(REPO_ROOT)),
    }

    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nEvaluation complete")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())