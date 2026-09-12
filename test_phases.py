import mujoco
import mujoco.viewer
import numpy as np
import time
import sys
sys.path.insert(0, "sim")
sys.path.insert(0, ".")

from randomization import DomainRandomizer
from scripted_episode_ik import build_drawer_and_cutlery_episode
from ik_demo_utils import (
    activate_grasp_connect, deactivate_grasp_connect,
    activate_grasp_weld, deactivate_grasp_weld,
)

model = mujoco.MjModel.from_xml_path("sim/assets/dinner_table_dual_so101.xml")
data = mujoco.MjData(model)
randomizer = DomainRandomizer(model, "configs/randomization.yaml")
randomizer.reset(data, 10010)
mujoco.mj_forward(model, data)

bid = model.body("right_gripper").id
OBJECTS = ["plate", "cup", "bottle",
           "spoon_1", "fork_1"]

ACTUATORS = ["left_shoulder_pan", "left_shoulder_lift",
             "left_elbow_flex", "left_wrist_flex",
             "left_wrist_roll", "left_gripper",
             "right_shoulder_pan", "right_shoulder_lift",
             "right_elbow_flex", "right_wrist_flex",
             "right_wrist_roll", "right_gripper"]

start_xy = {n: data.xpos[model.body(n).id][:2].copy() for n in OBJECTS}
start_z = {n: data.xpos[model.body(n).id][2] for n in OBJECTS}
drawer_jid = model.joint("drawer_slide").id
drawer_qadr = model.jnt_qposadr[drawer_jid]
drawer_start = data.qpos[drawer_qadr]

print(f"Drawer start position: {drawer_start:.4f}\n")

qpos_trace, gripper_trace, grasp_events = build_drawer_and_cutlery_episode(model, data, randomizer)
print(f"Trajectory built: {len(qpos_trace)} steps")
print(f"  plate_grasp_pos_err_cm = {grasp_events['plate_grasp_pos_err_cm']:.2f}")
print(f"  plate_grasp_angle_deg  = {grasp_events['plate_grasp_angle_deg']:.2f}\n")

# Left-arm plate grasp: release step (gripper reopens)
plate_open_step = None
prev_left_val = None
for i, g in enumerate(gripper_trace):
    if g is None:
        continue
    side, val = g
    if side != "left":
        continue
    if prev_left_val is not None and prev_left_val < 0.0 and val > 0.0:
        plate_open_step = i
        break
    prev_left_val = val
print(f"plate open step: {plate_open_step}")

left_gripper_bodies = {
    model.body("left_gripper").id,
    model.body("left_moving_jaw_so101_v1").id,
}
plate_bid = model.body("plate").id
plate_connect_armed = True

# Replay with live viewer
viewer = mujoco.viewer.launch_passive(model, data)

first_move = {}
for step in range(len(qpos_trace)):
    if not viewer.is_running():
        break
    for act_name in ACTUATORS:
        if act_name.endswith("_gripper"):
            continue
        act_id = model.actuator(act_name).id
        jid = model.actuator_trnid[act_id, 0]
        adr = model.jnt_qposadr[jid]
        data.ctrl[act_id] = qpos_trace[step][adr]

    gval = gripper_trace[step]
    if gval is not None:
        side, val = gval
        data.ctrl[model.actuator(f"{side}_gripper").id] = val

    # Right arm grasp connect (drawer handle).
    if step == grasp_events["activate_step"]:
        handle_world_now = data.geom_xpos[model.geom("drawer_handle").id].copy()
        activate_grasp_connect(model, data, grasp_events["eq_name"],
                                grasp_events["body1"], grasp_events["body2"],
                                handle_world_now)
    if step == grasp_events["deactivate_step"]:
        deactivate_grasp_connect(model, data, grasp_events["eq_name"])

    # Left arm plate grasp: WELD, not connect.
    if plate_connect_armed:
        for c in range(data.ncon):
            con = data.contact[c]
            b1 = int(model.geom_bodyid[con.geom1])
            b2 = int(model.geom_bodyid[con.geom2])
            if ((b1 in left_gripper_bodies and b2 == plate_bid)
                    or (b2 in left_gripper_bodies and b1 == plate_bid)):
                activate_grasp_weld(model, data, "left_grasp_weld",
                                     "left_gripper", "plate")
                print(f"  [step {step}] plate WELD ACTIVATED at first contact")
                plate_connect_armed = False
                break

    if plate_open_step is not None and step == plate_open_step:
        deactivate_grasp_weld(model, data, "left_grasp_weld")
        print(f"  [step {step}] plate WELD DEACTIVATED (release)")

    for _ in range(10):
        mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(0.005)

    # Full diagnostic window: covers tail-phase, plate pick, and the extra
    # 6-second hold phase (steps 360-659). Prints plate xyz at every sampled
    # step so we can see whether the plate stays at lift height through the
    # hold window (grasp retains) or drifts/falls (grasp fails).
    if step in (0, 100, 199, 200, 220, 240, 260, 280, 300, 320, 340,
                360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560,
                580, 600, 620, 640, 659):
        L_site = data.site_xpos[model.site("left_gripperframe").id]
        R_site = data.site_xpos[model.site("right_gripperframe").id]
        plate_xyz = data.xpos[plate_bid]
        print(f"step {step:4d}  drawer={data.qpos[drawer_qadr]:.3f}  "
              f"R_site=[{R_site[0]:.3f} {R_site[1]:.3f} {R_site[2]:.3f}]  "
              f"L_site=[{L_site[0]:.3f} {L_site[1]:.3f} {L_site[2]:.3f}]  "
              f"plate=[{plate_xyz[0]:.3f} {plate_xyz[1]:.3f} {plate_xyz[2]:.3f}]")

    for n in OBJECTS:
        if n in first_move:
            continue
        shift = float(np.linalg.norm(data.xpos[model.body(n).id][:2] - start_xy[n]))
        if shift > 0.03:
            first_move[n] = step

viewer.close()

print("\n=== STEP 3: first move of each object >3cm ===")
if not first_move:
    print("  no objects moved")
else:
    for n, step in first_move.items():
        print(f"  {n} at step {step}")

drawer_end = data.qpos[drawer_qadr]
print(f"\n=== Drawer motion ===")
print(f"  start: {drawer_start:.4f}  end: {drawer_end:.4f}  "
      f"travel: {(drawer_end - drawer_start)*100:.2f} cm")

print(f"\n=== Final object shifts (XY and Z separately) ===")
for n in OBJECTS:
    end = data.xpos[model.body(n).id]
    xy_shift = float(np.linalg.norm(end[:2] - start_xy[n]))
    z_shift = float(end[2] - start_z[n])
    print(f"  {n}: xy={xy_shift*100:6.1f}cm  z={z_shift*100:+6.2f}cm  "
          f"final=[{end[0]:.3f} {end[1]:.3f} {end[2]:.3f}]")

print("\n=== plate Z trajectory through hold phase ===")
print("  (look at the step-by-step prints above: 360 to 659)")
print(f"  plate final z: {data.xpos[plate_bid][2]:.4f}")

print("\n=== cutlery position after replay ===")
for n in sorted(randomizer.last_in_drawer_items):
    print(f"  {n}: {data.xpos[model.body(n).id][:2]}")