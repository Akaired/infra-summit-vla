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
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 8010
randomizer.reset(data, SEED)
print(f"=== SEED = {SEED} ===")
mujoco.mj_forward(model, data)

plate_bid = model.body("plate").id
plate_pos = data.xpos[plate_bid].copy()
print(f"plate start = ({plate_pos[0]:+.4f}, {plate_pos[1]:+.4f}, {plate_pos[2]:+.4f})")

OBJECTS = ["plate", "cup", "bottle", "spoon_1", "fork_1"]
ACTUATORS = ["left_shoulder_pan", "left_shoulder_lift",
             "left_elbow_flex", "left_wrist_flex",
             "left_wrist_roll", "left_gripper",
             "right_shoulder_pan", "right_shoulder_lift",
             "right_elbow_flex", "right_wrist_flex",
             "right_wrist_roll", "right_gripper"]

start_xy = {n: data.xpos[model.body(n).id][:2].copy() for n in OBJECTS}
start_z = {n: data.xpos[model.body(n).id][2] for n in OBJECTS}
drawer_qadr = model.jnt_qposadr[model.joint("drawer_slide").id]
drawer_start = data.qpos[drawer_qadr]

qpos_trace, gripper_trace, grasp_events = build_drawer_and_cutlery_episode(model, data, randomizer)
print(f"Trajectory built: {len(qpos_trace)} steps")
print(f"  plate_grasp_pos_err_cm = {grasp_events['plate_grasp_pos_err_cm']:.2f}")
print(f"  plate_grasp_angle_deg  = {grasp_events['plate_grasp_angle_deg']:.2f}\n")

# Commanded release step: where the LEFT gripper is commanded OPEN after
# having been commanded CLOSED. Simplified spec: no site_z gate on when to
# consider this "the" release step -- the actual weld deactivation timing
# is handled below by the hard timeout / site_z<0.85 check.
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
plate_weld_armed = True
plate_weld_released = False

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

    if step == grasp_events["activate_step"]:
        handle_world_now = data.geom_xpos[model.geom("drawer_handle").id].copy()
        activate_grasp_connect(model, data, grasp_events["eq_name"],
                                grasp_events["body1"], grasp_events["body2"],
                                handle_world_now)
    if step == grasp_events["deactivate_step"]:
        deactivate_grasp_connect(model, data, grasp_events["eq_name"])

    # NO GATES. Weld fires on ANY contact between any left-gripper body and
    # the plate, regardless of gripper position. Per updated spec: we are
    # not doing a clean rim grasp anymore, just glue-on-contact-and-carry.
    if plate_weld_armed:
        for c in range(data.ncon):
            con = data.contact[c]
            b1 = int(model.geom_bodyid[con.geom1])
            b2 = int(model.geom_bodyid[con.geom2])
            if ((b1 in left_gripper_bodies and b2 == plate_bid)
                    or (b2 in left_gripper_bodies and b1 == plate_bid)):
                activate_grasp_weld(model, data, "left_grasp_weld",
                                     "left_gripper", "plate")
                print(f"  [step {step}] plate WELD ACTIVATED (first contact, no gate)")
                plate_weld_armed = False
                break

    # Release when the commanded gripper value has gone to OPEN, with a
    # HARD TIMEOUT: 40 steps after the open command, release unconditionally
    # even if the site hasn't descended -- OR release early once site_z has
    # dropped below 0.85. This replaces the old site_z<0.80-only gate, which
    # could leave the weld active forever if the arm never got that low
    # (observed: weld stuck active to end of episode, plate dragged home).
    if plate_open_step is not None and step >= plate_open_step and not plate_weld_released:
        steps_since_open = step - plate_open_step
        site_z = data.site_xpos[model.site("left_gripperframe").id][2]
        if steps_since_open >= 40 or site_z < 0.85:
            deactivate_grasp_weld(model, data, "left_grasp_weld")
            print(f"  [step {step}] plate WELD DEACTIVATED "
                  f"(steps_since_open={steps_since_open}, site_z={site_z:.3f})")
            plate_weld_released = True

    for _ in range(10):
        mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(0.005)

    if step in (0, 100, 199, 200, 220, 240, 260, 280, 300, 320, 340,
                360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560,
                580, 600, 620, 640, 659):
        L_site = data.site_xpos[model.site("left_gripperframe").id]
        plate_xyz = data.xpos[plate_bid]
        print(f"step {step:4d}  L_site=[{L_site[0]:.3f} {L_site[1]:.3f} {L_site[2]:.3f}]  "
              f"plate=[{plate_xyz[0]:.3f} {plate_xyz[1]:.3f} {plate_xyz[2]:.3f}]")

    for n in OBJECTS:
        if n in first_move:
            continue
        shift = float(np.linalg.norm(data.xpos[model.body(n).id][:2] - start_xy[n]))
        if shift > 0.03:
            first_move[n] = step

# Verify the release actually happened. Never crash on this -- per spec,
# print a warning and continue if it somehow never fired.
if not plate_weld_released:
    print(f"WARNING: weld never released on seed {SEED} -- forcing release now.")
    deactivate_grasp_weld(model, data, "left_grasp_weld")

print("")
print("=== Final object shifts ===")
for n in OBJECTS:
    end = data.xpos[model.body(n).id]
    xy_shift = float(np.linalg.norm(end[:2] - start_xy[n]))
    z_shift = float(end[2] - start_z[n])
    print(f"  {n}: xy={xy_shift*100:6.1f}cm  z={z_shift*100:+6.2f}cm  final=[{end[0]:.3f} {end[1]:.3f} {end[2]:.3f}]")

print("")
print("Viewer open. Close the window or Ctrl+C to exit.")
while viewer.is_running():
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(0.01)
viewer.close()
