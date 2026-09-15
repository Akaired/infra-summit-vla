"""Live demo web server for the bimanual VLA dinner-table task.

Not graded (apps/web/README.md, PRD §3.5) — this is polish for the live demo
only. It runs the *real* pipeline underneath, though: EpisodeScene from /eval
(the same class the harness and eval_smolvla.py use) and whatever Policy
build_policy() resolves. Nothing about the sim or the policy is reimplemented
here; this file only bridges that existing loop to a browser over WebSocket.

Usage:
    python apps/web/server.py --config configs/eval.yaml --policy dummy
    python apps/web/server.py --config configs/eval.yaml --policy openvino

--policy is the same flag eval/run_episodes.py takes, and the same
eval.policy_interface.build_policy() call resolves it — so this server needs
no changes when the team's real SmolVLA/OpenVINO backend lands, as long as it
is exposed as an eval.policy_interface.Policy (see inference/README.md and
eval/policy_interface.py). Point --config at whatever eval config wires that
policy in; nothing here hardcodes "dummy".

No cloud dependency: binds to 0.0.0.0 so it is reachable over LAN (e.g. the
Intel machine) but talks to nothing outside this process and MuJoCo.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from eval.common import load_yaml  # noqa: E402
from eval.policy_interface import build_policy  # noqa: E402
from eval.scene import EpisodeScene  # noqa: E402
from eval.subtasks import SubtaskTracker  # noqa: E402

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "fastapi + uvicorn are required for the demo server.\n"
        "Install with: pip install fastapi 'uvicorn[standard]'\n"
        f"(original error: {exc})"
    ) from exc


# --------------------------------------------------------------------------- #
# Natural-language -> instruction passthrough.
#
# The policy contract (eval/policy_interface.py, inference/README.md) takes a
# free-text instruction string; it does not require a fixed vocabulary. The
# chat box therefore sends whatever the user types straight through as
# `instruction`, unchanged, on every step. This module does NOT parse or
# rewrite commands into a private grammar — doing so would silently diverge
# from what the harness and eval_smolvla.py actually feed the policy, which
# defeats the point of a demo meant to show the real pipeline.
# --------------------------------------------------------------------------- #


class LiveEpisode:
    """One running MuJoCo episode, driven by chat instructions over time.

    Thin orchestration around EpisodeScene + Policy — the same two seams
    eval/run_episodes.py uses. No physics, policy, or scene logic lives here.
    """

    def __init__(self, scene: EpisodeScene, policy: Any, eval_cfg: dict[str, Any]) -> None:
        self.scene = scene
        self.policy = policy
        self.eval_cfg = eval_cfg
        self.instruction = "Waiting for a command."
        self.tracker: SubtaskTracker | None = None
        self.running = False
        self.policy_steps = 0
        self._build_tracker()

    def _build_tracker(self) -> None:
        try:
            tracker = SubtaskTracker(self.scene, self.eval_cfg, self.scene.sim_cfg)
            tracker.bind_spawn_reference()
            self.tracker = tracker
        except Exception:
            # Subtask config/scene mismatch shouldn't take the viewer down —
            # the live view still works, just without subtask badges.
            self.tracker = None

    def reset(self, seed: int) -> dict[str, Any]:
        self.scene.reset(seed)
        self.instruction = "Waiting for a command."
        self.policy_steps = 0
        self._build_tracker()
        return self.state_payload(seed=seed)

    def set_instruction(self, instruction: str) -> None:
        instruction = instruction.strip()
        if instruction:
            self.instruction = instruction
            self.running = True

    def stop(self) -> None:
        self.running = False

    def tick(self) -> dict[str, Any]:
        """Advance one policy step if an instruction is active."""
        if not self.running:
            return self.state_payload()

        observation = self.scene.observe()
        action = self.policy.predict(self.instruction, observation, observation["robot_state"])
        self.scene.apply_action(action)
        self.scene.step()
        self.policy_steps += 1

        if self.tracker is not None:
            self.tracker.update(self.policy_steps)

        return self.state_payload()

    def state_payload(self, seed: int | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "state",
            "instruction": self.instruction,
            "running": self.running,
            "policy_steps": self.policy_steps,
            "robot_state": self.scene.robot_state().tolist(),
            "objects": self._object_positions(),
        }
        if seed is not None:
            payload["seed"] = seed
        if self.tracker is not None:
            payload["subtasks"] = dict(self.tracker.status)
        return payload

    def _object_positions(self) -> dict[str, list[float]]:
        objects_cfg = self.scene.sim_cfg.get("objects", {})
        out: dict[str, list[float]] = {}
        for key, value in objects_cfg.items():
            names = value if isinstance(value, list) else [value]
            for name in names:
                try:
                    out[name] = self.scene.body_xpos(name).tolist()
                except Exception:
                    continue
        return out


def describe_arm_rig(scene: EpisodeScene) -> dict[str, Any]:
    """Read the compiled MJCF's arm kinematic chain: per actuated joint, the
    body it drives, that body's offset from its parent, the joint axis/range,
    and the link's own capsule/box geoms (for a faithful three.js redraw).

    Reads only already-compiled MuJoCo model arrays -- no path, name, or
    number here is hardcoded; it all comes from sim/assets/*.xml via the
    model /eval already built. Skips any actuator whose joint's body prefix
    doesn't match sim.yaml:robot's left/right arm prefixes (i.e. gripper
    fingers driven by equality constraints, not a chain link).
    """
    import mujoco

    model = scene.model
    prefixes = (
        scene.sim_cfg["robot"]["left_arm_prefix"],
        scene.sim_cfg["robot"]["right_arm_prefix"],
    )

    def geom_descriptor(body_id: int) -> list[dict[str, Any]]:
        geoms = []
        for gid in range(model.ngeom):
            if int(model.geom_bodyid[gid]) != body_id:
                continue
            gtype = int(model.geom_type[gid])
            size = model.geom_size[gid].tolist()
            geoms.append({
                "type": mujoco.mjtGeom(gtype).name.replace("mjGEOM_", "").lower(),
                "size": size,
                "pos": model.geom_pos[gid].tolist(),
                # wxyz; MuJoCo compiles a <geom fromto=...> into an equivalent
                # pos+quat+size, so this is populated even though the source
                # XML used fromto, not pos/quat directly.
                "quat": model.geom_quat[gid].tolist(),
            })
        return geoms

    bases = {}
    for prefix in prefixes:
        base_name = f"{prefix}arm_base"
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_name)
        if body_id < 0:
            continue
        bases[base_name] = {
            "body_pos": model.body_pos[body_id].tolist(),
            "geoms": geom_descriptor(body_id),
        }

    links = []
    for act_id in range(model.nu):
        joint_id = int(model.actuator_trnid[act_id, 0])
        if joint_id < 0:
            continue
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        if not any(joint_name.startswith(p) for p in prefixes):
            continue

        body_id = int(model.jnt_bodyid[joint_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        parent_id = int(model.body_parentid[body_id])
        parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id) or ""

        links.append({
            "actuator_index": act_id,
            "joint_name": joint_name,
            "joint_type": mujoco.mjtJoint(int(model.jnt_type[joint_id])).name,
            "axis": model.jnt_axis[joint_id].tolist(),
            "range": model.jnt_range[joint_id].tolist(),
            "body_name": body_name,
            "parent_body": parent_name,
            "body_pos": model.body_pos[body_id].tolist(),
            "geoms": geom_descriptor(body_id),
        })

    return {"bases": bases, "links": links}


def describe_static_scene(scene: EpisodeScene) -> dict[str, Any]:
    """Read every static (non-arm) body's own geoms once, so the frontend can
    draw the actual table/drawer/room instead of a guessed placeholder.

    Mirrors describe_arm_rig(): generic over whatever
    sim/assets/dinner_table_dual_so101.xml defines, no body/geom name
    hardcoded here beyond skipping the arm bodies (already covered by
    describe_arm_rig) and mesh-typed geoms (YCB visual/collision meshes,
    already covered by the "objects" positions the frontend gets from
    /api/info and draws with its own primitives). Excludes collision-only
    geoms (MuJoCo render group 3-5, invisible in MuJoCo's own viewer too).
    """
    import mujoco

    model = scene.model
    prefixes = (
        scene.sim_cfg["robot"]["left_arm_prefix"],
        scene.sim_cfg["robot"]["right_arm_prefix"],
    )

    bodies: dict[str, Any] = {}
    for body_id in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if not body_name or body_name == "world":
            continue
        if any(body_name.startswith(p) for p in prefixes):
            continue  # arm bodies: already described by describe_arm_rig()

        geoms = []
        for gid in range(model.ngeom):
            if int(model.geom_bodyid[gid]) != body_id:
                continue
            if int(model.geom_group[gid]) >= 3:
                continue  # collision-only, hidden in MuJoCo's own viewer
            gtype = int(model.geom_type[gid])
            type_name = mujoco.mjtGeom(gtype).name.replace("mjGEOM_", "").lower()
            if type_name == "mesh":
                continue  # YCB objects: frontend already draws these from "objects"
            rgba = model.geom_rgba[gid].tolist()
            geoms.append({
                "type": type_name,
                "size": model.geom_size[gid].tolist(),
                "pos": model.geom_pos[gid].tolist(),
                "quat": model.geom_quat[gid].tolist(),
                "rgba": rgba,
            })
        if not geoms:
            continue

        # Joint info (e.g. the drawer's slide joint) lets the frontend know
        # this body moves and how, instead of treating it as static decor.
        joint_id = int(model.body_jntadr[body_id])
        joint = None
        if joint_id >= 0 and int(model.body_jntnum[body_id]) > 0:
            joint = {
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or "",
                "type": mujoco.mjtJoint(int(model.jnt_type[joint_id])).name,
                "axis": model.jnt_axis[joint_id].tolist(),
                "range": model.jnt_range[joint_id].tolist(),
            }

        bodies[body_name] = {
            "body_pos": model.body_pos[body_id].tolist(),
            "geoms": geoms,
            "joint": joint,
        }

    return {"bodies": bodies}


def build_app(config_path: str, policy_name: str, primary_camera_override: str | None) -> FastAPI:
    eval_cfg = load_yaml(config_path)
    sim_cfg = load_yaml(eval_cfg["configs"]["sim"])
    randomization_cfg_path = eval_cfg["configs"]["randomization"]
    policy_cfg_path = eval_cfg["configs"].get("policy", "configs/policy.yaml")
    primary_camera = primary_camera_override or eval_cfg["policy"]["primary_camera"]

    scene = EpisodeScene(sim_cfg, randomization_cfg_path)
    policy = build_policy(policy_name, policy_cfg_path, primary_camera)
    episode = LiveEpisode(scene, policy, eval_cfg)

    randomization_cfg = load_yaml(randomization_cfg_path)
    default_seeds = randomization_cfg.get("seeds", [0])
    episode.reset(seed=int(default_seeds[0]) if default_seeds else 0)

    tick_hz = 10.0  # policy ticks/sec while an instruction is running
    tick_interval = 1.0 / tick_hz

    app = FastAPI()
    static_dir = Path(__file__).resolve().parent / "static"

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(str(Path(__file__).resolve().parent / "index.html"))

    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/api/info")
    async def info() -> dict[str, Any]:
        return {
            "policy": policy_name,
            "primary_camera": primary_camera,
            "seeds": load_yaml(randomization_cfg_path).get("seeds", []),
            "objects": sim_cfg.get("objects", {}),
            # Static arm geometry (link lengths/offsets, joint axes/ranges),
            # read once from the compiled MJCF so the frontend can build the
            # exact same kinematic chain sim/assets/dinner_table_dual_so101.xml
            # defines, instead of guessing proportions. Actuator order here
            # MUST match scene.robot_state()'s qpos order (actuator order),
            # which is what every "robot_state" WS message uses.
            "arm_rig": describe_arm_rig(scene),
            # Table, drawer, room walls/floor -- everything the frontend
            # needs to draw the actual scene instead of a single guessed
            # slab. Static per session (read once, not per WS tick).
            "static_scene": describe_static_scene(scene),
        }

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_text(json.dumps(episode.state_payload()))

        async def run_ticks() -> None:
            while True:
                await asyncio.sleep(tick_interval)
                if episode.running:
                    try:
                        payload = episode.tick()
                    except Exception as exc:  # noqa: BLE001
                        payload = {"type": "error", "message": str(exc)}
                        episode.running = False
                    await websocket.send_text(json.dumps(payload))

        ticker = asyncio.create_task(run_ticks())
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    await websocket.send_text(
                        json.dumps({"type": "error", "message": "invalid JSON"})
                    )
                    continue

                kind = message.get("type")
                if kind == "instruction":
                    episode.set_instruction(str(message.get("text", "")))
                    await websocket.send_text(json.dumps(episode.state_payload()))
                elif kind == "reset":
                    seed = int(message.get("seed", 0))
                    payload = episode.reset(seed)
                    await websocket.send_text(json.dumps(payload))
                elif kind == "stop":
                    episode.stop()
                    await websocket.send_text(json.dumps(episode.state_payload()))
                else:
                    await websocket.send_text(
                        json.dumps({"type": "error", "message": f"unknown message type {kind!r}"})
                    )
        except WebSocketDisconnect:
            pass
        finally:
            ticker.cancel()

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--policy", default="dummy", choices=["dummy", "openvino"])
    parser.add_argument("--camera", default=None, help="Override policy.primary_camera")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    app = build_app(args.config, args.policy, args.camera)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
