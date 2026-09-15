# `/apps/web` — live demo viewer (optional, not graded)

**Branch:** `feature/viewer` · **Status:** implemented, not yet run against a real OpenVINO backend

A live three.js viewer + chat box: type an instruction, watch the bimanual
episode run in the browser in real time. Polish for the demonstration/pitch,
never the graded pipeline — see "Read this before starting work here" below,
which still applies.

## What it actually is

- `server.py` — FastAPI + WebSocket bridge around the **existing** eval seams:
  `eval.scene.EpisodeScene` (same class `eval/run_episodes.py` and
  `eval/eval_smolvla.py` use) and `eval.policy_interface.build_policy()`
  (same `--policy {dummy,openvino}` switch `run_episodes.py` takes). No
  physics, policy, or scene logic is reimplemented here — this file only
  bridges that loop to a browser. Also exposes `GET /api/info`, which reads
  the compiled MJCF once (`describe_arm_rig`) and reports the arm's real
  kinematic chain — link offsets, joint axes/ranges, each link's own
  capsule/box geoms — so the frontend can rebuild the *exact* rig
  `sim/assets/dinner_table_dual_so101.xml` defines instead of guessing
  proportions.
- `index.html` — three.js scene, arms rendered via the **real SO-101 URDF +
  STL meshes** (`apps/web/static/so101/`, from
  [`TheRobotStudio/SO-ARM100`](https://github.com/TheRobotStudio/SO-ARM100),
  Apache-2.0), loaded twice at runtime with `urdf-loader` — one `URDFRobot`
  instance per arm. Each instance is positioned at the compiled MJCF's
  `left_arm_base` / `right_arm_base` offset (from `GET /api/info`'s
  `arm_rig.bases`), so the real meshes sit where the sim's placeholder arms
  sit. Table objects (plate/cup/cutlery/bottle) are simple primitives
  positioned from the same live state; a full YCB mesh import was judged not
  worth the time against tonight's deadline.
- The chat box sends whatever text you type straight through as the
  `instruction` string on every policy tick — no private command grammar,
  so what you see is exactly what the policy is actually being asked.

## Known fact about the rig (so nobody "fixes" this by accident)

`sim/assets/dinner_table_dual_so101.xml` is explicitly a placeholder: the
arms are built from primitive capsules/boxes (5 revolute joints + 1 slide
gripper per arm), not real SO-101 meshes — it's a physics stand-in, not a
visual one. The **viewer no longer renders that placeholder geometry at
all**: `index.html` loads the real SO-101 URDF/STL rig instead (see above).

Because the URDF is a different skeleton than the MJCF placeholder (different
joint names, slightly different proportions), the mapping from sim state to
rendered pose is by **index, not name**: `robot_state`'s first 6 actuated
qpos values (this arm's qpos, in actuator order) drive the URDF's own 6
joints in its natural base-to-gripper order — `shoulder_pan, shoulder_lift,
elbow_flex, wrist_flex, wrist_roll, gripper` — via `ARM_JOINT_ORDER` in
`index.html`. Motion is faithful to the real policy's action sequence and
timing; absolute pose will not match the sim's capsule geometry
pixel-for-pixel. `describe_arm_rig()` in `server.py` is unchanged and still
only used to place the two arm bases (`arm_rig.bases`) — its per-link
`arm_rig.links` geometry (capsule/box sizes) is no longer consumed by the
frontend now that meshes come from the URDF instead.

If a real SO-101 MJCF ever replaces the placeholder in `sim/assets/`, this
index-based mapping should be revisited — a name-based mapping would then be
possible and more robust.

## Run it

```bash
# same venv as eval/run_episodes.py
pip install fastapi "uvicorn[standard]"

# dummy policy (works today, no checkpoint needed)
python apps/web/server.py --config configs/eval.yaml --policy dummy

# real backend, once /inference exposes an eval.policy_interface.Policy
python apps/web/server.py --config configs/eval.yaml --policy openvino
```

Open `http://localhost:8000` (or `http://<machine-ip>:8000` from another
device on the LAN — the server binds `0.0.0.0`). Pick a seed, type an
instruction (e.g. "open the top drawer"), hit Send.

## Read this before starting work here

This is **cut-first**. It earns zero rubric points, MuJoCo's built-in viewer
is sufficient for the video requirement, and PRD §4 explicitly rejects cloud
hosting for anything on the graded path. If the eval harness, the benchmark,
or the 10-seed video are not finished, that work comes first.

No cloud deployment: local or LAN only. The server talks to nothing outside
this process and MuJoCo.
