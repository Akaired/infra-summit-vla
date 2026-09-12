"""
Bimanual episode: right arm opens drawer AND holds it open, left arm
retrieves cutlery (currently skipped), right arm closes drawer, left arm
picks up the plate and places it near table center.
"""
import numpy as np
import mink
import mujoco
from ik_demo_utils import (
    waypoint_sequence, joint_space_lerp,
    compute_gripper_closing_axis_local, compute_plate_rim_grasp,
)

APPROACH_Z_OFFSET = 0.08
GRASP_OPEN = 1.2
GRASP_CLOSED = -0.16
CONTROL_DECIMATION = 10

# Cutlery retrieval is fully implemented below but SKIPPED, not deleted --
# the moving-jaw convex hull cannot retain a thin cutlery capsule under lift.
SKIP_RETRIEVE = True

# Calibrated handle grasp pose -- measured, two days of tuning. DO NOT TOUCH.
HANDLE_POS = np.array([-0.01374231, 0.12252117, 0.91137901])
HANDLE_QUAT = np.array([0.04128911, -0.13647269, 0.68135373, 0.71793281])

MANUAL_ENTRY_QPOS_5 = [-1.15, -0.14, 0.879, -0.0663, -1.46]
MANUAL_ENTRY_QUAT = np.array([0.62552, 0.68241, 0.10020, -0.36470])
LEFT_ARM_JOINTS_5 = ["left_shoulder_pan", "left_shoulder_lift", "left_elbow_flex",
                     "left_wrist_flex", "left_wrist_roll"]

HOUSING_SAFE_Y_OUTSIDE = 0.15
LID_CLEARANCE_Z = 0.895

# Place zone: rectangle in front of a seated human. The plate is placed
# somewhere inside this rectangle, sampled per-seed with REJECTION against
# the actual positions of the cup and bottle after reset -- so the plate
# is never placed where an obstacle already sits. Rejection (not radial
# push) keeps the chosen point inside the zone even when the cup happens
# to spawn near its centre.
PLACE_ZONE_X = (-0.10, 0.00)
PLACE_ZONE_Y = (-0.10, -0.02)
# Safety gap beyond pure contact. Larger than yaml min_gap_between_objects_m
# (0.018) because the plate slides a few cm on release.
PLACE_SAFETY_GAP = 0.05

# Plate hangs ~15-18cm below the gripper site when welded off-axis. At
# TRANSIT_Z=1.02 the plate bottom sits at ~0.84, right at cup top (0.85),
# so the plate clips the cup mid-flight and shoves it. 1.18 puts the plate
# bottom at ~1.00, above both cup and bottle tops.
TRANSIT_Z = 1.18


def _replay_phase_on_scratch(model, initial_qpos, qpos_phase, grip_phase):
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = initial_qpos
    scratch.qvel[:] = 0.0
    mujoco.mj_forward(model, scratch)

    for step in range(len(qpos_phase)):
        for act_id in range(model.nu):
            jid = model.actuator_trnid[act_id, 0]
            adr = model.jnt_qposadr[jid]
            scratch.ctrl[act_id] = qpos_phase[step][adr]
        gval = grip_phase[step]
        if gval is not None:
            side, val = gval
            scratch.ctrl[model.actuator(f"{side}_gripper").id] = val
        for _ in range(CONTROL_DECIMATION):
            mujoco.mj_step(model, scratch)

    return scratch


def build_drawer_and_cutlery_episode(model, data, randomizer):
    configuration = mink.Configuration(model)
    configuration.update(data.qpos.copy())

    handle_frame = "right_gripperframe"
    right_home_stage = np.array([0.22, 0.00, 0.90])
    left_home_stage = np.array([-0.22, 0.00, 0.90])

    slide_axis = model.jnt_axis[model.joint("drawer_slide").id]
    slide_range = model.jnt_range[model.joint("drawer_slide").id][1]
    pull_target = HANDLE_POS + slide_axis * slide_range

    # -------- PHASE 1: right arm opens drawer --------
    open_waypoints = [
        (handle_frame, right_home_stage, None, GRASP_OPEN),
        (handle_frame, HANDLE_POS + [0, 0, APPROACH_Z_OFFSET], HANDLE_QUAT, GRASP_OPEN),
        (handle_frame, HANDLE_POS, HANDLE_QUAT, GRASP_OPEN),
        (handle_frame, HANDLE_POS, HANDLE_QUAT, GRASP_CLOSED),
        (handle_frame, pull_target, HANDLE_QUAT, GRASP_CLOSED),
    ]
    qpos_open, grip_open = waypoint_sequence(
        configuration, model, open_waypoints, steps_per_segment=40
    )
    grasp_activate_step = len(qpos_open) - 40

    scratch = _replay_phase_on_scratch(model, data.qpos.copy(), qpos_open, grip_open)

    # -------- PHASE 2: LEFT retrieves; RIGHT holds handle open --------
    qpos_retr = []
    grip_retr = []

    def add(segment_qpos, segment_grip):
        qpos_retr.extend(segment_qpos)
        grip_retr.extend(segment_grip)

    for name in ([] if SKIP_RETRIEVE else sorted(randomizer.last_in_drawer_items)):
        cutlery_pos = scratch.xpos[model.body(name).id].copy()

        seg, g = waypoint_sequence(
            configuration, model,
            [("left_gripperframe", left_home_stage, None, GRASP_OPEN, None)],
            steps_per_segment=30,
            secondary_frame="right_gripperframe", secondary_xyz=pull_target,
            secondary_quat=HANDLE_QUAT, secondary_pos_cost=1.0, secondary_ori_cost=0.15,
        )
        add(seg, g)

        seg = joint_space_lerp(configuration, model, LEFT_ARM_JOINTS_5,
                                MANUAL_ENTRY_QPOS_5, steps=40)
        add(seg, [("left", GRASP_OPEN)] * len(seg))

        seg, g = waypoint_sequence(
            configuration, model,
            [("left_gripperframe", cutlery_pos, MANUAL_ENTRY_QUAT, GRASP_OPEN, "drawer")],
            steps_per_segment=30,
        )
        add(seg, g)

        seg, g = waypoint_sequence(
            configuration, model,
            [("left_gripperframe", cutlery_pos, MANUAL_ENTRY_QUAT, GRASP_CLOSED, "drawer")],
            steps_per_segment=20,
        )
        add(seg, g)

        lift_pos = np.array([cutlery_pos[0], cutlery_pos[1], LID_CLEARANCE_Z])
        seg, g = waypoint_sequence(
            configuration, model,
            [("left_gripperframe", lift_pos, MANUAL_ENTRY_QUAT, GRASP_CLOSED, "drawer")],
            steps_per_segment=20,
        )
        add(seg, g)

        retreat_pos = np.array([cutlery_pos[0], HOUSING_SAFE_Y_OUTSIDE, LID_CLEARANCE_Z])
        seg, g = waypoint_sequence(
            configuration, model,
            [("left_gripperframe", retreat_pos, MANUAL_ENTRY_QUAT, GRASP_CLOSED, "drawer")],
            steps_per_segment=20,
        )
        add(seg, g)

        z_height = float(cutlery_pos[2])
        place_pos = np.array([-0.15, -0.15, z_height])
        seg, g = waypoint_sequence(
            configuration, model,
            [
                ("left_gripperframe", place_pos + [0, 0, APPROACH_Z_OFFSET], None, GRASP_CLOSED, None),
                ("left_gripperframe", place_pos, None, GRASP_CLOSED, None),
                ("left_gripperframe", place_pos, None, GRASP_OPEN, None),
            ],
            steps_per_segment=20,
            secondary_frame="right_gripperframe", secondary_xyz=pull_target,
            secondary_quat=HANDLE_QUAT, secondary_pos_cost=1.0, secondary_ori_cost=0.15,
        )
        add(seg, g)

    # -------- PHASE 3: right closes drawer, left picks + places plate --------
    right_tail_waypoints = [
        (handle_frame, HANDLE_POS, HANDLE_QUAT, GRASP_CLOSED),  # push closed
        (handle_frame, HANDLE_POS, HANDLE_QUAT, GRASP_OPEN),    # release
        (handle_frame, right_home_stage, None, GRASP_OPEN),     # retract
    ]

    # LEFT arm plate grasp -- SIMPLIFIED per updated spec: no orientation
    # correction, no adaptive rim_fraction/grasp_side search. Fixed
    # rim_fraction=0.9, grasp_side="near". target_quat=None on every
    # waypoint. Approach -> descend -> close -> lift -> transit -> descend
    # -> open -> retract. The gripper welds to the plate at whatever angle
    # first contact happens to occur; carrying and approximate placement is
    # the only goal, not a clean rim straddle.
    plate_pos = data.xpos[model.body("plate").id].copy()
    left_base_xy = np.array([-0.22, 0.20])
    closing_axis_local = compute_gripper_closing_axis_local(
        model, data, "left_gripper", "left_moving_jaw_so101_v1", "left_gripper",
        reference_qpos_5=[0.0, 0.0, 0.0, 0.0, 0.0], arm_joint_names=LEFT_ARM_JOINTS_5,
    )
    rim_xyz, outside_xyz, plate_quat, plate_grasp_pos_err_cm, plate_grasp_angle_deg = compute_plate_rim_grasp(
        model, data, configuration,
        plate_center_xy=plate_pos[:2], left_base_xy=left_base_xy,
        plate_radius=0.089, plate_z=float(plate_pos[2]),
        closing_axis_local=closing_axis_local, arm_joint_names=LEFT_ARM_JOINTS_5,
        gripper_frame_name="left_gripperframe", home_stage_xyz=left_home_stage,
        rim_fraction=0.9, grasp_side="near",
    )

    # Sample a free spot from PLACE_ZONE by rejection against the ACTUAL
    # current positions of cup and bottle. Deterministic per seed (RNG
    # seeded from the plate's start XY). Rejection -- not radial push --
    # so the chosen point always stays INSIDE the zone; a radial push can
    # drift the point outside the defined region.
    plate_r = 0.089
    cup_r = 0.06
    bottle_r = 0.04
    rng = np.random.RandomState(
        int((abs(plate_pos[0]) * 1e6 + abs(plate_pos[1]) * 1e6)) % (2 ** 31)
    )
    obstacles = [
        (data.xpos[model.body("cup").id][:2].copy(),
         plate_r + cup_r + PLACE_SAFETY_GAP),
        (data.xpos[model.body("bottle").id][:2].copy(),
         plate_r + bottle_r + PLACE_SAFETY_GAP),
    ]
    place_xy = None
    for _ in range(200):
        cand = np.array([
            rng.uniform(PLACE_ZONE_X[0], PLACE_ZONE_X[1]),
            rng.uniform(PLACE_ZONE_Y[0], PLACE_ZONE_Y[1]),
        ])
        if all(np.linalg.norm(cand - obs_xy) >= need
               for obs_xy, need in obstacles):
            place_xy = cand
            break
    if place_xy is None:
        # Rejection failed -- fall back to zone center, pushed radially
        # away from any obstacle it overlaps. Should not happen for these
        # zone sizes but is here as a safety net.
        print(f"[place] rejection failed after 200 tries; using fallback. "
              f"plate_start=({plate_pos[0]:.3f}, {plate_pos[1]:.3f})")
        place_xy = np.array([
            0.5 * (PLACE_ZONE_X[0] + PLACE_ZONE_X[1]),
            0.5 * (PLACE_ZONE_Y[0] + PLACE_ZONE_Y[1]),
        ])
        for obs_xy, need in obstacles:
            d = float(np.linalg.norm(place_xy - obs_xy))
            if d < need:
                if d < 1e-6:
                    place_xy = obs_xy + np.array([-1.0, 0.0]) * need
                else:
                    place_xy = obs_xy + (place_xy - obs_xy) / d * need
    print(f"[place] plate_start=({plate_pos[0]:.3f},{plate_pos[1]:.3f}) "
          f"cup=({data.xpos[model.body('cup').id][0]:.3f},{data.xpos[model.body('cup').id][1]:.3f}) "
          f"-> place_xy=({place_xy[0]:.3f},{place_xy[1]:.3f})")

    place_pos = np.array([place_xy[0], place_xy[1], float(rim_xyz[2])])
    place_above = place_pos + np.array([0.0, 0.0, APPROACH_Z_OFFSET])
    rim_above = rim_xyz + np.array([0.0, 0.0, APPROACH_Z_OFFSET])

    left_tail_waypoints = [
        ("left_gripperframe", left_home_stage, None, GRASP_OPEN, None),
        ("left_gripperframe", outside_xyz, None, GRASP_OPEN, "plate"),
        ("left_gripperframe", rim_above, None, GRASP_OPEN, "plate"),
        ("left_gripperframe", rim_xyz, None, GRASP_OPEN, "plate"),
        ("left_gripperframe", rim_xyz, None, GRASP_CLOSED, "plate"),
        ("left_gripperframe", rim_above, None, GRASP_CLOSED, "plate"),
        ("left_gripperframe", np.array([place_xy[0], place_xy[1], TRANSIT_Z]),
         None, GRASP_CLOSED, "plate"),
        ("left_gripperframe", place_above, None, GRASP_CLOSED, "plate"),
        ("left_gripperframe", place_pos, None, GRASP_CLOSED, "plate"),
        ("left_gripperframe", place_pos, None, GRASP_OPEN, "plate"),
        ("left_gripperframe", place_above, None, GRASP_OPEN, "plate"),
        ("left_gripperframe", left_home_stage, None, GRASP_OPEN, None),
    ]

    qpos_right_tail, grip_right_tail = waypoint_sequence(
        configuration, model, right_tail_waypoints, steps_per_segment=20
    )
    qpos_left_tail, grip_left_tail = waypoint_sequence(
        configuration, model, left_tail_waypoints, steps_per_segment=40
    )

    qpos_tail = qpos_right_tail + qpos_left_tail
    grip_tail = grip_right_tail + grip_left_tail
    grasp_deactivate_step = len(qpos_open) + len(qpos_retr) + 20

    qpos_trace = qpos_open + qpos_retr + qpos_tail
    gripper_trace = grip_open + grip_retr + grip_tail
    grasp_events = {
        "activate_step": grasp_activate_step,
        "deactivate_step": grasp_deactivate_step,
        "eq_name": "right_grasp_connect",
        "body1": "right_moving_jaw_so101_v1",
        "body2": "drawer",
        "plate_grasp_pos_err_cm": plate_grasp_pos_err_cm,
        "plate_grasp_angle_deg": plate_grasp_angle_deg,
    }
    return qpos_trace, gripper_trace, grasp_events