"""
Smoke test for the placeholder dual-arm dinner-table scene.

Checks, in order:
  1. MJCF compiles.
  2. Every name sim.yaml promises (cameras, object bodies, arm prefixes)
     actually resolves in the compiled model -- this is the contract /policy,
     /inference, and /eval all rely on.
  3. The scene is stable at rest for a few hundred steps (no NaNs / explosion).
  4. No unexpected left/right arm self-collision at the home pose.
  5. Both wrist cameras + overhead camera can render an offscreen frame.

Run: python3 sim/smoke_test.py
"""
import sys
import yaml
import numpy as np
import mujoco

CONFIG_PATH = "configs/sim.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def check_names(model, cfg):
    print("\n--- Name contract check (sim.yaml vs compiled MJCF) ---")
    ok = True

    # Cameras
    cam_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                 for i in range(model.ncam)]
    for cam in cfg["cameras"]:
        status = "OK" if cam["name"] in cam_names else "MISSING"
        if status == "MISSING":
            ok = False
        print(f"  camera '{cam['name']}': {status}")

    # Objects
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                  for i in range(model.nbody)]
    obj_cfg = cfg["objects"]
    flat_objects = [obj_cfg["drawer"], obj_cfg["plate"], obj_cfg["cup"], obj_cfg["bottle"]] + obj_cfg["cutlery"]
    for name in flat_objects:
        status = "OK" if name in body_names else "MISSING"
        if status == "MISSING":
            ok = False
        print(f"  object body '{name}': {status}")

    # Arm prefixes
    left_pref = cfg["robot"]["left_arm_prefix"]
    right_pref = cfg["robot"]["right_arm_prefix"]
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                   for i in range(model.njnt)]
    left_joints = [n for n in joint_names if n and n.startswith(left_pref)]
    right_joints = [n for n in joint_names if n and n.startswith(right_pref)]
    print(f"  left-prefixed joints ({left_pref}*): {len(left_joints)} found -> {left_joints}")
    print(f"  right-prefixed joints ({right_pref}*): {len(right_joints)} found -> {right_joints}")
    if not left_joints or not right_joints:
        ok = False

    return ok


def apply_home_qpos(model, data, cfg):
    """Start from sim.yaml's measured home_qpos, not raw qpos0. qpos0 is
    just wherever the MJCF's default joint values happen to land -- for the
    real SO-101 (mid-range calibration), that's a curled-up pose where the
    two arms visibly overlap (verified: up to 4.5cm penetration), not a
    pose the system ever actually uses. DomainRandomizer.reset() always
    moves to home_qpos immediately; checking self-collision/stability
    anywhere else tests a state that doesn't matter and produces false
    alarms on any future robot swap, exactly as it just did on this one."""
    mujoco.mj_resetData(model, data)
    home = cfg.get("robot", {}).get("home_qpos")
    if home:
        for joint_name, value in home.items():
            adr = model.jnt_qposadr[model.joint(joint_name).id]
            data.qpos[adr] = value
    mujoco.mj_forward(model, data)


def check_stability(model, data, cfg, n_steps=500):
    print(f"\n--- Stability check ({n_steps} steps from home_qpos, no control input) ---")
    apply_home_qpos(model, data, cfg)
    for i in range(n_steps):
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            print(f"  FAIL: non-finite state detected at step {i}")
            return False
    max_qvel = float(np.max(np.abs(data.qvel)))
    print(f"  OK: {n_steps} steps stepped cleanly. max |qvel| at end = {max_qvel:.4f}")
    if max_qvel > 5.0:
        print("  WARNING: velocities still large -- scene may not have settled (check object spawn heights).")
    return True


def check_self_collision(model, data, cfg, max_penetration_m=0.001):
    print("\n--- Left/right arm self-collision check at home_qpos ---")
    apply_home_qpos(model, data, cfg)
    left_geoms = set()
    right_geoms = set()
    for i in range(model.ngeom):
        body_id = model.geom_bodyid[i]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if body_name.startswith("left_"):
            left_geoms.add(i)
        elif body_name.startswith("right_"):
            right_geoms.add(i)

    # Real overlap (more than max_penetration_m), not the small sub-mm
    # "touching" every resting contact shows in MuJoCo's solver -- that's
    # the normal representation of contact, not instability. Verified: the
    # real SO-101's equilibrium pose shows 8 contacts, all under 0.3mm.
    real_hits = []
    for c in range(data.ncon):
        con = data.contact[c]
        g1, g2 = con.geom1, con.geom2
        if (g1 in left_geoms and g2 in right_geoms) or (g1 in right_geoms and g2 in left_geoms):
            if con.dist < -max_penetration_m:
                real_hits.append(con.dist)
    print(f"  cross left/right contacts deeper than {max_penetration_m*1000:.1f}mm: {len(real_hits)}")
    if real_hits:
        print(f"  FAIL: real self-collision at home_qpos (worst: {min(real_hits)*1000:.2f}mm) "
              f"-- widen base separation or check geometry.")
        return False
    print("  OK: no meaningful self-collision between arms at home_qpos.")
    return True


def check_cameras_render(model, data):
    print("\n--- Offscreen camera render check ---")
    ok = True
    renderer = mujoco.Renderer(model, height=240, width=320)
    for cam_name in ["overhead", "wrist_left", "wrist_right"]:
        try:
            renderer.update_scene(data, camera=cam_name)
            frame = renderer.render()
            print(f"  camera '{cam_name}': rendered frame shape {frame.shape}")
        except Exception as e:
            print(f"  camera '{cam_name}': FAILED to render ({e})")
            ok = False
    return ok


def main():
    cfg = load_config()
    mjcf_path = cfg["scene"]["mjcf"]
    print(f"Loading MJCF: {mjcf_path}")
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)
    print(f"Compiled OK: {model.nbody} bodies, {model.njnt} joints, {model.ncam} cameras, {model.ngeom} geoms")

    results = {
        "name_contract": check_names(model, cfg),
        "stability": check_stability(model, data, cfg),
        "self_collision": check_self_collision(model, data, cfg),
        "camera_render": check_cameras_render(model, data),
    }

    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")

    if not all(results.values()):
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
