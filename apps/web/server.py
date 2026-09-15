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

    def _object_positions(self) -> dict[str, dict[str, list[float]]]:
        """Per tracked object: world position AND orientation.

        Position alone (the original version of this method) is enough for
        the free-standing table objects but wrong for anything the frontend
        renders as a real oriented mesh (plate/cup/spoon/fork) or a moving
        rigid shell (the drawer) -- those need the body's actual world
        quaternion too, or they render pinned to a fixed default rotation
        while the physics spins/rotates them. data.xquat is always populated
        by MuJoCo for every body; no name or index here is hardcoded beyond
        the object names sim.yaml:objects already lists.
        """
        objects_cfg = self.scene.sim_cfg.get("objects", {})
        out: dict[str, dict[str, list[float]]] = {}
        for key, value in objects_cfg.items():
            names = value if isinstance(value, list) else [value]
            for name in names:
                try:
                    body_id = self.scene.model.body(name).id
                    out[name] = {
                        "pos": self.scene.body_xpos(name).tolist(),
                        "quat": self.scene.data.xquat[body_id].tolist(),  # wxyz
                    }
                except Exception:
                    continue
        return out


# Texture image per tracked body, for the visual mesh describe_object_meshes()
# exports. Unlike everything else in this file, this one small map IS
# hand-written rather than read back from the compiled model: MuJoCo's
# compiler bakes OBJ/MTL/PNG assets into internal arrays and does not
# preserve the original source file path anywhere on MjModel (confirmed:
# there is no mesh_pathadr/tex_pathadr -- mj_saveXML re-emits asset *names*,
# not source paths). The texture files themselves already live in this repo
# at sim/assets/ycb/<dir>/texture_map.png (see the <mesh>/<material> tags in
# sim/assets/dinner_table_dual_so101.xml); this only records which directory
# goes with which tracked body name, copied into apps/web/static/ycb/ by
# hand alongside this file. A body absent here (or one with no mesh geom at
# all, like "bottle") just renders untextured/flat-shaded on the frontend.
OBJECT_TEXTURES = {
    "plate": "ycb/plate/texture_map.png",
    "cup": "ycb/g_cups/texture_map.png",
    "spoon_1": "ycb/spoon/texture_map.png",
    "spoon_2": "ycb/spoon/texture_map.png",
    "fork_1": "ycb/fork/texture_map.png",
    "fork_2": "ycb/fork/texture_map.png",
}


def describe_object_meshes(scene: EpisodeScene) -> dict[str, Any]:
    """Export the actual visual mesh (vertices + triangles + UVs, if any) for
    every tracked object body that has one, read directly from the compiled
    model -- not the source .obj files, and not a guessed primitive.

    Why not just serve the .obj files under sim/assets/ycb/ directly: MuJoCo
    re-centers/re-orients a mesh asset into its own inertial frame at compile
    time (mesh_pos/mesh_quat), and that same compiled mesh is what the
    physics actually simulates contact against -- reading it back from the
    model guarantees the viewer draws exactly what the sim sees, with no
    separate OBJ-parsing path that could drift out of sync (different vertex
    winding, a re-exported/rescaled OBJ, etc). Only objects listed in
    sim.yaml:objects are considered; a body with no mesh geom (e.g. "bottle",
    a primitive cylinder) is simply absent from the result -- the frontend
    already draws primitives for those.

    Frame chain per MuJoCo's own compiled-mesh convention: world position
    for vertex v of this geom's mesh is
        body_xpos + body_xmat @ (geom_pos + geom_mat @ mesh_vert[v])
    NOT mesh_pos/mesh_quat as well -- model.mesh_vert is already the
    COMPILED mesh (MuJoCo re-centers/re-aligns each mesh asset to its own
    inertial frame at compile time, and bakes that exact offset into the
    referencing geom's geom_pos/geom_quat; mesh_pos/mesh_quat only matter if
    you instead load raw vertices from the *original* OBJ/STL file, which
    this does not). Using both double-applies the same offset -- confirmed
    against a real case here: this repo's mesh geoms (plate/cup/spoon/fork)
    have no explicit pos/quat in their <geom> tag, so the compiler set
    geom_pos/geom_quat equal to mesh_pos/mesh_quat exactly (identity-compose
    with an unset geom pose), and applying both rotated the mesh through
    that same rotation twice, landing objects on their side instead of flat.
    Sending geom_pos/geom_quat (already relative to the body, like
    describe_arm_rig's link geoms) lets the frontend bake that chain once at
    load time, then just move/rotate the body each tick from the live
    "objects" pos+quat -- no per-vertex work at runtime.
    """
    import mujoco

    model = scene.model
    objects_cfg = scene.sim_cfg.get("objects", {})
    tracked_names: set[str] = set()
    for value in objects_cfg.values():
        tracked_names.update(value if isinstance(value, list) else [value])

    out: dict[str, Any] = {}
    for name in tracked_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            continue
        mesh_geom = None
        for gid in range(model.ngeom):
            if int(model.geom_bodyid[gid]) != body_id:
                continue
            if int(model.geom_type[gid]) != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            if int(model.geom_group[gid]) >= 3:
                continue  # collision-only mesh; the *_visual_geom is group 0
            mesh_geom = gid
            break
        if mesh_geom is None:
            continue  # e.g. "bottle": a primitive geom, no mesh to export

        mesh_id = int(model.geom_dataid[mesh_geom])
        v0, vn = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
        f0, fn = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
        t0 = int(model.mesh_texcoordadr[mesh_id])
        tn = int(model.mesh_texcoordnum[mesh_id]) if t0 >= 0 else 0

        entry: dict[str, Any] = {
            "vertices": model.mesh_vert[v0 : v0 + vn].tolist(),
            "faces": model.mesh_face[f0 : f0 + fn].tolist(),
            "geom_pos": model.geom_pos[mesh_geom].tolist(),
            "geom_quat": model.geom_quat[mesh_geom].tolist(),  # wxyz
        }
        if tn > 0:
            entry["texcoords"] = model.mesh_texcoord[t0 : t0 + tn].tolist()
            face_texcoord_attr = getattr(model, "mesh_facetexcoord", None)
            if face_texcoord_attr is not None and len(face_texcoord_attr) > 0:
                entry["face_texcoords"] = face_texcoord_attr[f0 : f0 + fn].tolist()

        rgba = model.geom_rgba[mesh_geom].tolist()
        if rgba != [0.5, 0.5, 0.5, 1.0]:  # MuJoCo's material-driven default
            entry["rgba"] = rgba

        texture_path = OBJECT_TEXTURES.get(name)
        if texture_path:
            entry["texture"] = f"/static/{texture_path}"

        out[name] = entry

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

    # Every tracked object (sim.yaml:objects) is already drawn and moved by
    # the frontend's own "objects" WS-driven path (either a real mesh via
    # object_meshes, or -- for a body with no mesh, like "bottle", a
    # primitive cylinder -- meshForObject()'s shape-matched fallback).
    # "drawer" is the one exception: it needs its actual multi-panel shell
    # geometry (floor/walls/handle), which only this function reads, so it
    # stays in static_scene -- the frontend keeps a reference to that one
    # group (jointedBodies) and repositions THAT from live "objects" data
    # instead of drawing a second, competing primitive for it. Every other
    # tracked body (bottle, plate, cup, spoon_*, fork_*) must NOT also get a
    # static_scene entry, or the frontend ends up with two separate,
    # independently-updated objects for the same body -- one that tracks the
    # physics and one static leftover from its initial pose.
    objects_cfg = scene.sim_cfg.get("objects", {})
    tracked_names: set[str] = set()
    for value in objects_cfg.values():
        tracked_names.update(value if isinstance(value, list) else [value])
    tracked_names.discard("drawer")

    bodies: dict[str, Any] = {}
    for body_id in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if not body_name or body_name == "world":
            continue
        if any(body_name.startswith(p) for p in prefixes):
            continue  # arm bodies: already described by describe_arm_rig()
        if body_name in tracked_names:
            continue  # tracked object: drawn/moved via "objects", not here

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
            # Real visual meshes (vertices/faces/UVs) for tracked objects
            # that have one -- plate/cup/spoon/fork -- read from the same
            # compiled model the physics uses, not a guessed sphere/box.
            "object_meshes": describe_object_meshes(scene),
        }

    @app.get("/api/debug_axis")
    async def debug_axis() -> dict:
        """Diagnostic-only: for each tracked mesh object, compute the real
        long axis (via PCA on mesh_vert) in the COMPILED-MESH local frame,
        then transform it through geom_quat (compiled-mesh -> body) and the
        live body_quat (body -> world) to see where it actually points in
        world/mj space right now. This is ground truth independent of any
        client-side rendering code."""
        import numpy as np
        meshes = describe_object_meshes(scene)
        out = {}
        for name in ["fork_1", "spoon_1"]:
            if name not in meshes:
                continue
            verts = np.array(meshes[name]["vertices"])
            centroid = verts.mean(axis=0)
            centered = verts - centroid
            # PCA: long axis = eigenvector of largest eigenvalue of covariance
            cov = centered.T @ centered
            eigvals, eigvecs = np.linalg.eigh(cov)
            long_axis_mesh_local = eigvecs[:, np.argmax(eigvals)]

            geom_quat = meshes[name]["geom_quat"]  # wxyz
            def quat2mat(q):
                w, x, y, z = q
                return np.array([
                    [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                    [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                    [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
                ])
            R_geom = quat2mat(geom_quat)
            long_axis_body = R_geom @ long_axis_mesh_local

            import mujoco as _mj
            bid = _mj.mj_name2id(scene.model, _mj.mjtObj.mjOBJ_BODY, name)
            body_quat = scene.data.xquat[bid].tolist()  # wxyz, world
            R_body = quat2mat(body_quat)
            long_axis_world = (R_body @ long_axis_body).tolist()

            out[name] = {
                "centroid_mesh_local": centroid.tolist(),
                "long_axis_mesh_local": long_axis_mesh_local.tolist(),
                "long_axis_body_frame": long_axis_body.tolist(),
                "long_axis_world_frame": long_axis_world,
            }
        return out

    @app.get("/api/debug_state")
    async def debug_state() -> dict:
        """Diagnostic-only: current live episode.state_payload() objects
        block (pos+quat per tracked body) -- ground truth for what the
        WS stream is actually sending right now."""
        payload = episode.state_payload()
        return payload.get("objects", {})

    # --- Diagnostic-only endpoints below, left in deliberately -----------
    # Not used by index.html, not on the graded path. Kept because they were
    # exactly what resolved the fork/spoon "standing up" investigation: they
    # let geom_pos/geom_quat/live-orientation be read directly, without
    # wading through the (huge) per-object vertex arrays /api/info returns.
    # Cheap, read-only, side-effect-free -- safe to leave mounted.
    @app.get("/api/debug_geom")
    async def debug_geom() -> dict:
        """Diagnostic-only: geom_pos/geom_quat for tracked mesh objects,
        without the (huge) vertex/face arrays. Not used by index.html."""
        meshes = describe_object_meshes(scene)
        return {
            name: {"geom_pos": m["geom_pos"], "geom_quat": m["geom_quat"]}
            for name, m in meshes.items()
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
