# `/apps/web` — live demo viewer (optional, not graded)

**Status:** available on `main`; not yet run against a real OpenVINO backend

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
  the compiled MJCF once and reports:
  - `arm_rig` (`describe_arm_rig`) — the arm's real kinematic chain: link
    offsets, joint axes/ranges, each link's own capsule/box geoms.
  - `static_scene` (`describe_static_scene`) — every other static/kinematic
    body's own geoms (table top + legs, the drawer's floor/walls/handle,
    room floor/walls), generic over whatever
    `sim/assets/dinner_table_dual_so101.xml` actually defines — nothing
    about the table or drawer's size/position is hardcoded in `server.py`
    or `index.html`, it's all read from the compiled model. Bodies with a
    joint (currently just `drawer`, a slide joint) are flagged so the
    frontend knows that body moves at runtime rather than treating it as
    fixed decor.
- `index.html` — three.js scene built from the **same compiled MJCF** used by
  physics. Arm links use the real SO-101 STL meshes under
  `sim/assets/so101/`, with body-relative positions, rest quaternions, live
  joint rotations, and each geom's resolved material color exported by
  `server.py`. The table, drawer housing, drawer, and room come from
  `static_scene`; plate/cup/cutlery/bowl use their compiled YCB meshes and
  textures; the bottle uses its four real primitive geoms. The drawer group
  is moved every tick from live physics state. `OrbitControls` supports
  orbit, zoom, and pan.
- The chat box sends whatever text you type straight through as the
  `instruction` string on every policy tick — no private command grammar,
  so what you see is exactly what the policy is actually being asked.

## Rig source of truth

`sim/assets/dinner_table_dual_so101.xml` contains the two real SO-101
kinematic chains and references the vendored STL assets in
`sim/assets/so101/`. The viewer does not load a second URDF or maintain a
parallel visual skeleton. `describe_arm_rig()` reads body hierarchy, geom
transforms, mesh data, materials, axes, ranges, and actuator indices from the
compiled model; the frontend only performs the MuJoCo-to-three.js coordinate
conversion and applies live joint state.

## Run it

```bash
# Run from the repository root. The transient --with dependencies do not
# modify uv.lock.
uv run --frozen \
  --extra sim \
  --extra dummy \
  --with fastapi \
  --with "uvicorn[standard]" \
  python apps/web/server.py --config configs/eval.yaml --policy dummy

# real backend, once /inference exposes an eval.policy_interface.Policy
# replace the final `dummy` above with `openvino`
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
