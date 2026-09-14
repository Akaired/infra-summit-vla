"""
test_phases.py — visual + numeric check of the full episode.

Run with:   python test_phases.py <seed>
Opens a MuJoCo viewer. To capture output to a file:
    python test_phases.py 3 > log_seed3.txt
"""
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

# ---- CORRECT PATHS (relative to repo root) ----
MODEL_PATH = "sim/assets/dinner_table_dual_so101.xml"
RANDOMIZATION_CFG = "configs/randomization.yaml"

# Set to False if you want the window to close itself at the end.
HOLD_VIEWER_OPEN = True

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
randomizer = DomainRandomizer(model, RANDOMIZATION_CFG)
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 8010
randomizer.reset(data, SEED)
print(f"=== SEED = {SEED} ===")
mujoco.mj_forward(model, data)

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
print(f"  plate_place_xy = {grasp_events['plate_place_xy']}")
print(f"  cup_place_xy   = {grasp_events['cup_place_xy']}")
print(f"  cup_grasp_xyz  = {grasp_events['cup_grasp_xyz']}")

# ---------- Find transitions ----------
def find_first_transition(gripper_trace, side):
    prev = None
    for i, g in enumerate(gripper_trace):
        if g is None:
            continue
        s, val = g
        if s != side:
            continue
        if prev is not None and prev < 0 and val > 0:
            return i
        prev = val
    return None

def find_last_transition(gripper_trace, side):
    last = None
    prev = None
    for i, g in enumerate(gripper_trace):
        if g is None:
            continue
        s, val = g
        if s != side:
            continue
        if prev is not None and prev < 0 and val > 0:
            last = i
        prev = val
    return last

plate_open_step = find_first_transition(gripper_trace, "left")
cup_open_step = find_last_transition(gripper_trace, "right")
print(f"plate open step: {plate_open_step}")
print(f"cup open step:   {cup_open_step}")
print()

# ---------- Body IDs ----------
left_gripper_bodies = {
    model.body("left_gripper").id,
    model.body("left_moving_jaw_so101_v1").id,
}
right_gripper_bodies = {
    model.body("right_gripper").id,
    model.body("right_moving_jaw_so101_v1").id,
}
plate_bid = model.body("plate").id
cup_bid = model.body("cup").id
bottle_bid = model.body("bottle").id
drawer_bid = model.body("drawer").id
housing_bid = model.body("drawer_housing").id
L_site_id = model.site("left_gripperframe").id
R_site_id = model.site("right_gripperframe").id

right_arm_body_ids = set()
for bid in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
    if name.startswith("right_"):
        right_arm_body_ids.add(bid)

drawer_geom_ids = set()
housing_geom_ids = set()
for gid in range(model.ngeom):
    bb = int(model.geom_bodyid[gid])
    if bb == drawer_bid:
        drawer_geom_ids.add(gid)
    elif bb == housing_bid:
        housing_geom_ids.add(gid)


def count_contacts_with(bids_a, body_b):
    n = 0
    for c in range(data.ncon):
        con = data.contact[c]
        b1 = int(model.geom_bodyid[con.geom1])
        b2 = int(model.geom_bodyid[con.geom2])
        if (b1 in bids_a and b2 == body_b) or (b2 in bids_a and b1 == body_b):
            n += 1
    return n


def min_right_arm_to_drawer_dist():
    best = 1e9
    for bid in right_arm_body_ids:
        p = data.xpos[bid]
        for gid in drawer_geom_ids:
            q = data.geom_xpos[gid]
            d = float(np.linalg.norm(p - q))
            if d < best:
                best = d
    return best


# ---------- Weld state ----------
plate_weld_armed = True
plate_weld_released = False
cup_weld_armed = True
cup_weld_released = False


def should_print(s):
    if s < 200:
        return s % 50 == 0
    if 200 <= s < 350:
        return s % 10 == 0
    if 350 <= s < 500:
        return s % 20 == 0
    if 500 <= s < 620:
        return s % 10 == 0
    if 620 <= s < 800:
        return s % 20 == 0
    return s % 50 == 0


viewer = mujoco.viewer.launch_passive(model, data)
print("Step trace  |  L_site  R_site  drawer  plate  cup  |  contacts")
print("-" * 100)

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

    # ---- Drawer handle connect ----
    if step == grasp_events["activate_step"]:
        handle_world_now = data.geom_xpos[model.geom("drawer_handle").id].copy()
        activate_grasp_connect(model, data, grasp_events["eq_name"],
                                grasp_events["body1"], grasp_events["body2"],
                                handle_world_now)
        print(f"  [step {step}] DRAWER connect ACTIVATED")
    if step == grasp_events["deactivate_step"]:
        deactivate_grasp_connect(model, data, grasp_events["eq_name"])
        print(f"  [step {step}] DRAWER connect DEACTIVATED")

    # ---- Plate weld ----
    if plate_weld_armed:
        for c in range(data.ncon):
            con = data.contact[c]
            b1 = int(model.geom_bodyid[con.geom1])
            b2 = int(model.geom_bodyid[con.geom2])
            if ((b1 in left_gripper_bodies and b2 == plate_bid)
                    or (b2 in left_gripper_bodies and b1 == plate_bid)):
                activate_grasp_weld(model, data, "left_grasp_weld", "left_gripper", "plate")
                Ls = data.site_xpos[L_site_id]
                pp = data.xpos[plate_bid]
                print(f"  [step {step}] PLATE WELD ACTIVATED  "
                      f"L_site=({Ls[0]:+.3f},{Ls[1]:+.3f},{Ls[2]:+.3f})  "
                      f"plate=({pp[0]:+.3f},{pp[1]:+.3f},{pp[2]:+.3f})")
                plate_weld_armed = False
                break
    if plate_open_step is not None and step >= plate_open_step and not plate_weld_released:
        steps_since_open = step - plate_open_step
        site_z = data.site_xpos[L_site_id][2]
        plate_place_z = grasp_events["plate_place_z"]
        gripper_at_plate_height = site_z < (plate_place_z + 0.02)
        if (gripper_at_plate_height and steps_since_open >= 5) or steps_since_open >= 80:
            deactivate_grasp_weld(model, data, "left_grasp_weld")
            print(f"  [step {step}] PLATE WELD DEACTIVATED  "
                  f"(site_z={site_z:.3f}, plate_z={plate_place_z:.3f}, "
                  f"steps_since_open={steps_since_open})")
            plate_weld_released = True

    # ---- Cup weld ----
    if cup_weld_armed:
        for c in range(data.ncon):
            con = data.contact[c]
            b1 = int(model.geom_bodyid[con.geom1])
            b2 = int(model.geom_bodyid[con.geom2])
            if ((b1 in right_gripper_bodies and b2 == cup_bid)
                    or (b2 in right_gripper_bodies and b1 == cup_bid)):
                activate_grasp_weld(model, data, "right_grasp_weld", "right_gripper", "cup")
                Rs = data.site_xpos[R_site_id]
                cp = data.xpos[cup_bid]
                print(f"  [step {step}] CUP WELD ACTIVATED  "
                      f"R_site=({Rs[0]:+.3f},{Rs[1]:+.3f},{Rs[2]:+.3f})  "
                      f"cup=({cp[0]:+.3f},{cp[1]:+.3f},{cp[2]:+.3f})")
                cup_weld_armed = False
                break
    if cup_open_step is not None and step >= cup_open_step and not cup_weld_released:
        steps_since_open = step - cup_open_step
        site_z = data.site_xpos[R_site_id][2]
        cup_place_z = grasp_events["cup_place_z"]
        gripper_at_cup_height = site_z < (cup_place_z + 0.02)
        if (gripper_at_cup_height and steps_since_open >= 5) or steps_since_open >= 80:
            deactivate_grasp_weld(model, data, "right_grasp_weld")
            print(f"  [step {step}] CUP WELD DEACTIVATED  "
                  f"(site_z={site_z:.3f}, cup_z={cup_place_z:.3f}, "
                  f"steps_since_open={steps_since_open})")
            cup_weld_released = True

    # ---- Drawer stall diagnostics ----
    if 200 <= step <= 400 and step % 10 == 0:
        dq = float(data.qpos[drawer_qadr])
        dr_contacts = count_contacts_with(right_gripper_bodies, drawer_bid)
        grip_q = float(data.qpos[model.jnt_qposadr[model.joint("right_gripper").id]])
        Rs = data.site_xpos[R_site_id]
        print(f"  [step {step}] DRAWER  qpos={dq:+.4f}  "
              f"R_site=({Rs[0]:+.3f},{Rs[1]:+.3f},{Rs[2]:+.3f})  "
              f"right_gripper_joint={grip_q:+.3f}  "
              f"contact_right_gripper_drawer={dr_contacts}")

    # ---- Branch reset distance check ----
    if 500 <= step <= 620 and step % 10 == 0:
        d = min_right_arm_to_drawer_dist()
        Rs = data.site_xpos[R_site_id]
        print(f"  [step {step}] BRANCH RESET  "
              f"R_site=({Rs[0]:+.3f},{Rs[1]:+.3f},{Rs[2]:+.3f})  "
              f"min_right_body_to_drawer_geom={d:.3f}")

    for _ in range(10):
        mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(0.005)

    if should_print(step):
        Ls = data.site_xpos[L_site_id]
        Rs = data.site_xpos[R_site_id]
        pp = data.xpos[plate_bid]
        cp = data.xpos[cup_bid]
        dq = float(data.qpos[drawer_qadr])
        c_plate = count_contacts_with(left_gripper_bodies, plate_bid)
        c_cup = count_contacts_with(right_gripper_bodies, cup_bid)
        c_bottle = (count_contacts_with(left_gripper_bodies, bottle_bid)
                    + count_contacts_with(right_gripper_bodies, bottle_bid))
        print(f"  step {step:>4d}  "
              f"L_site=({Ls[0]:+.3f},{Ls[1]:+.3f},{Ls[2]:+.3f})  "
              f"R_site=({Rs[0]:+.3f},{Rs[1]:+.3f},{Rs[2]:+.3f})  "
              f"drawer={dq:+.4f}  "
              f"plate=({pp[0]:+.3f},{pp[1]:+.3f},{pp[2]:+.3f})  "
              f"cup=({cp[0]:+.3f},{cp[1]:+.3f},{cp[2]:+.3f})  "
              f"| contacts: plate={c_plate} cup={c_cup} bottle={c_bottle}")

# ---- Force-release any weld that never fired ----
if not plate_weld_released:
    print(f"WARNING: plate weld never released on seed {SEED} -- forcing release now.")
    deactivate_grasp_weld(model, data, "left_grasp_weld")
if not cup_weld_released:
    print(f"WARNING: cup weld never released on seed {SEED} -- forcing release now.")
    deactivate_grasp_weld(model, data, "right_grasp_weld")

print()
print("=== Final state ===")
drawer_final = float(data.qpos[drawer_qadr])
print(f"  drawer: qpos={drawer_final:.4f}  "
      f"closed={'YES' if abs(drawer_final) < 0.01 else 'NO -- stalled partway'}")
for n in OBJECTS:
    end = data.xpos[model.body(n).id]
    xy_shift = float(np.linalg.norm(end[:2] - start_xy[n]))
    z_shift = float(end[2] - start_z[n])
    print(f"  {n}: xy={xy_shift*100:6.1f}cm  z={z_shift*100:+6.2f}cm  "
          f"final=[{end[0]:.3f} {end[1]:.3f} {end[2]:.3f}]")

plate_place_xy = grasp_events["plate_place_xy"]
cup_place_xy = grasp_events["cup_place_xy"]
plate_moved_from_place = float(np.linalg.norm(data.xpos[plate_bid][:2] - plate_place_xy)) * 100
cup_near_target = float(np.linalg.norm(data.xpos[cup_bid][:2] - cup_place_xy)) * 1000
print(f"  plate moved from its placed position: {plate_moved_from_place:.2f}cm  "
      f"({'OK, <15cm' if plate_moved_from_place < 15 else 'FLAG: >15cm'})")
print(f"  cup final distance from cup_place_xy target: {cup_near_target:.1f}mm")

cup_up = data.xmat[cup_bid].reshape(3, 3) @ np.array([0, 0, 1])
plate_up = data.xmat[plate_bid].reshape(3, 3) @ np.array([0, 0, 1])
bottle_up = data.xmat[bottle_bid].reshape(3, 3) @ np.array([0, 0, 1])
print(f"  cup upright: {'YES' if abs(cup_up[2]) > 0.7 else 'NO -- tipped'}")
print(f"  plate upright: {'YES' if abs(plate_up[2]) > 0.7 else 'NO -- tipped'}")
print(f"  bottle upright: {'YES' if abs(bottle_up[2]) > 0.7 else 'NO -- tipped'}")
print(f"  dist cup-to-plate: "
      f"{np.linalg.norm(data.xpos[cup_bid][:2] - data.xpos[plate_bid][:2]) * 100:.1f} cm "
      f"(min safe = {(0.089 + 0.039) * 100:.1f} cm)")

print()
if HOLD_VIEWER_OPEN:
    print("Viewer open. Close the window or Ctrl+C to exit.")
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
viewer.close()