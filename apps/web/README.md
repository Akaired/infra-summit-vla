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
- `index.html` — three.js scene. Builds that kinematic chain as nested
  `Object3D` pivots (mirroring the MJCF's nested `<body>` structure) and
  rotates/translates them from every WebSocket `robot_state` update — so the
  viewer's arm pose **is** the physics state, not a separate animation. Table
  objects (plate/cup/cutlery/bottle) are simple primitives positioned from
  the same live state; a full YCB mesh import was judged not worth the time
  against tonight's deadline (the rig itself has no mesh geometry to import
  anyway — see below).
- The chat box sends whatever text you type straight through as the
  `instruction` string on every policy tick — no private command grammar,
  so what you see is exactly what the policy is actually being asked.

## Known fact about the rig (so nobody "fixes" this by accident)

`sim/assets/dinner_table_dual_so101.xml` is explicitly a placeholder: the
arms are built from primitive capsules/boxes (5 revolute joints + 1 slide
gripper per arm), not real SO-101 meshes or a URDF. There is currently no
mesh-accurate rig anywhere in this repo to import. If a real SO-101 MJCF/URDF
ever replaces the placeholder, `describe_arm_rig()` in `server.py` needs no
change — it reads geometry from the compiled model generically — but
`meshFromMujocoGeom()` in `index.html` only knows capsule/cylinder/box; a
mesh-based rig would need a mesh loader added there too.

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
