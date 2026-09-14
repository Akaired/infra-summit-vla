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
    compute_cup_grasp_target, activate_grasp_weld, deactivate_grasp_weld,
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
RIGHT_ARM_JOINTS_5 = ["right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex",
                      "right_wrist_flex", "right_wrist_roll"]

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

# Phase 4: cup placement relative to where the plate actually landed, not
# a fixed absolute point -- teaches "place relative to the plate", the
# correct table-setting abstraction.
# CUP_OFFSET_FROM_PLATE magnitude must clear need_plate = cup_outer_r (~0.039)
# + plate_r (0.089) + CUP_SAFETY_GAP (0.05) = 0.178, even after jitter in the
# worst-case direction. (0.16,0.04) has magnitude 0.165 -- UNDER 0.178, so
# the plate-rejection sampler (Fix B) would reject every candidate and
# always hit the fallback. (0.20,0.06) has magnitude 0.209; worst-case
# jitter (-0.02,-0.02) gives (0.18,0.04), magnitude 0.1844 -- still clears
# 0.178 with ~6mm margin.
CUP_OFFSET_FROM_PLATE = np.array([0.20, 0.06])
CUP_JITTER = np.array([0.020, 0.020])
CUP_SAFETY_GAP = 0.05
RIGHT_BASE_XY = np.array([0.22, 0.20])
# Separate from the plate's own TRANSIT_Z (frozen, plate phase is not
# touched). Reduced from 1.30 to 1.00 (Option A) -- the bypass concept
# this altitude was added for is deleted; 1.00 is the ordinary transit
# height with no bypass logic depending on it.
CUP_TRANSIT_Z = 1.00


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


def _find_first_transition(gripper_trace, side):
    """First closed->open transition for the given side."""
    prev = None
    for i, g in enumerate(gripper_trace):
        if g is None:
            continue
        s, val = g
        if s != side:
            continue
        if prev is not None and prev < 0.0 and val > 0.0:
            return i
        prev = val
    return None


def _replay_plate_phase_on_scratch(model, initial_qpos, qpos_phase, grip_phase):
    """Replay the drawer-close + plate-pick-and-place phase on a scratch
    MjData, WITH the same weld-arming/release logic the real episode replay
    uses (contact-triggered activate, commanded-open + hard-timeout
    release) -- not just blind ctrl playback. Without this, the scratch
    replay can't show where the plate actually ends up or what it actually
    does to the cup, because a plain physics replay has nothing holding the
    plate to the gripper (that's the whole reason the weld exists) and the
    plate would just sit wherever incidental contact leaves it instead of
    being carried and placed like it really is at runtime.

    Returns the scratch MjData after the full phase (drawer close + plate
    grasp/carry/place/release) has actually run, including the weld having
    fired and released, so plate/cup/bottle positions reflect what really
    happened physically -- not the pre-episode state.
    """
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = initial_qpos
    scratch.qvel[:] = 0.0
    mujoco.mj_forward(model, scratch)

    plate_open_step = _find_first_transition(grip_phase, "left")
    left_gripper_bodies = {model.body("left_gripper").id, model.body("left_moving_jaw_so101_v1").id}
    plate_bid = model.body("plate").id
    left_site_id = model.site("left_gripperframe").id
    plate_weld_armed = True
    plate_weld_released = False

    for step in range(len(qpos_phase)):
        for act_id in range(model.nu):
            jid = model.actuator_trnid[act_id, 0]
            adr = model.jnt_qposadr[jid]
            scratch.ctrl[act_id] = qpos_phase[step][adr]
        gval = grip_phase[step]
        if gval is not None:
            side, val = gval
            scratch.ctrl[model.actuator(f"{side}_gripper").id] = val

        if plate_weld_armed:
            for c in range(scratch.ncon):
                con = scratch.contact[c]
                b1 = int(model.geom_bodyid[con.geom1])
                b2 = int(model.geom_bodyid[con.geom2])
                if ((b1 in left_gripper_bodies and b2 == plate_bid)
                        or (b2 in left_gripper_bodies and b1 == plate_bid)):
                    activate_grasp_weld(model, scratch, "left_grasp_weld", "left_gripper", "plate")
                    plate_weld_armed = False
                    break
        if (plate_open_step is not None and step >= plate_open_step
                and not plate_weld_released):
            steps_since_open = step - plate_open_step
            site_z = scratch.site_xpos[left_site_id][2]
            if steps_since_open >= 40 or site_z < 0.85:
                deactivate_grasp_weld(model, scratch, "left_grasp_weld")
                plate_weld_released = True

        for _ in range(CONTROL_DECIMATION):
            mujoco.mj_step(model, scratch)

    if not plate_weld_released:
        deactivate_grasp_weld(model, scratch, "left_grasp_weld")

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

    # -------- PHASE 4: right arm picks up the cup, places near the plate --------
    # Built AFTER a scratch dry-run of the drawer-close + plate phase, and
    # planned against the ACTUAL resulting state, not the pre-episode one.
    # The plate phase physically moves the cup (and sometimes nudges the
    # bottle) on the way to placing the plate -- planning the cup phase from
    # data.xpos at the top of the function reads stale positions and misses
    # by several cm, sometimes routing straight through where the bottle
    # used to be but no longer is. This mirrors exactly how the drawer
    # phase already handles cutlery (_replay_phase_on_scratch), just with
    # the plate weld active during the replay so the plate is actually
    # carried, not just nudged by bare contact.
    scratch_after_plate = _replay_plate_phase_on_scratch(
        model, data.qpos.copy(), qpos_tail, grip_tail
    )
    plate_actual_xy = scratch_after_plate.xpos[model.body("plate").id][:2].copy()
    cup_actual_xy_now = scratch_after_plate.xpos[model.body("cup").id][:2].copy()
    bottle_actual_xy = scratch_after_plate.xpos[model.body("bottle").id][:2].copy()
    print(f"[phase4 scratch] plate planned=({place_xy[0]:.3f},{place_xy[1]:.3f}) "
          f"actual=({plate_actual_xy[0]:.3f},{plate_actual_xy[1]:.3f})  "
          f"cup moved to=({cup_actual_xy_now[0]:.3f},{cup_actual_xy_now[1]:.3f})  "
          f"bottle at=({bottle_actual_xy[0]:.3f},{bottle_actual_xy[1]:.3f})")

    # Same mechanic as the plate: target_quat=None everywhere (verified:
    # 10/10 seeds reach with 0.00cm error and 6-11 jaw-cup contacts each when
    # position-only; adding any orientation target costs 6-8cm and 0/10
    # seeds actually touch the cup). Weld to the FIXED body right_gripper,
    # never the moving jaw (moving jaw -> 21deg rotation drift measured
    # earlier). No orientation correction, no gate beyond first contact,
    # hard-timeout release -- identical philosophy to the plate phase.
    #
    # compute_cup_grasp_target reads the cup's REAL collision-mesh geometry
    # from `scratch_after_plate`, i.e. at its ACTUAL post-plate-phase
    # position/orientation, not the pre-episode one.
    cup_grasp_xyz, cup_approach_xyz, cup_inner_r, cup_outer_r = compute_cup_grasp_target(
        model, scratch_after_plate, RIGHT_BASE_XY
    )
    cup_above = cup_grasp_xyz + np.array([0.0, 0.0, 0.08])

    # Cup place is relative to where the plate ACTUALLY ended up (not
    # `place_xy`, the pre-plate-phase target), and rejects against BOTH
    # where the bottle ACTUALLY is now AND the plate's actual footprint --
    # the original sampler only checked the bottle, so a cup_place could
    # (and on seed 6, did) land 2mm inside the plate's outer edge.
    bottle_r = 0.04
    plate_r = 0.089
    need_bottle = cup_outer_r + bottle_r + CUP_SAFETY_GAP
    need_plate = cup_outer_r + plate_r + CUP_SAFETY_GAP
    cup_rng = np.random.RandomState(
        int((abs(plate_actual_xy[0]) * 1e6 + abs(plate_actual_xy[1]) * 1e6 + 7)) % (2 ** 31)
    )
    cup_place_xy = None
    for _ in range(200):
        jitter = cup_rng.uniform(-CUP_JITTER, CUP_JITTER)
        cand = plate_actual_xy + CUP_OFFSET_FROM_PLATE + jitter
        d_bottle = float(np.linalg.norm(cand - bottle_actual_xy))
        d_plate = float(np.linalg.norm(cand - plate_actual_xy))
        if d_bottle >= need_bottle and d_plate >= need_plate:
            cup_place_xy = cand
            break
    if cup_place_xy is None:
        # Fallback also respects both distances now, pushed away from
        # whichever obstacle (plate or bottle) it's closest to violating.
        cup_place_xy = plate_actual_xy + CUP_OFFSET_FROM_PLATE
        d_bottle = float(np.linalg.norm(cup_place_xy - bottle_actual_xy))
        if d_bottle < need_bottle:
            direction = (cup_place_xy - bottle_actual_xy) / d_bottle if d_bottle > 1e-6 else np.array([1.0, 0.0])
            cup_place_xy = bottle_actual_xy + direction * need_bottle
        d_plate = float(np.linalg.norm(cup_place_xy - plate_actual_xy))
        if d_plate < need_plate:
            direction = (cup_place_xy - plate_actual_xy) / d_plate if d_plate > 1e-6 else np.array([1.0, 0.0])
            cup_place_xy = plate_actual_xy + direction * need_plate
    print(f"[cup place] plate_actual=({plate_actual_xy[0]:.3f},{plate_actual_xy[1]:.3f}) "
          f"-> cup_place=({cup_place_xy[0]:.3f},{cup_place_xy[1]:.3f})  "
          f"dist_to_plate={float(np.linalg.norm(cup_place_xy-plate_actual_xy)):.3f} "
          f"(need>={need_plate:.3f})")

    cup_place_pos = np.array([cup_place_xy[0], cup_place_xy[1], float(cup_grasp_xyz[2])])
    cup_place_above = cup_place_pos + np.array([0.0, 0.0, APPROACH_Z_OFFSET])

    # --- Branch-consistency reset, BEFORE any Cartesian IK toward the cup ---
    # reach_cup.py's isolated 10/10 test always starts from a FRESH scene
    # reset, i.e. the right arm at its measured gravity-equilibrium joint
    # values. On this redundant 5-DOF arm the same Cartesian home position
    # is reachable via more than one elbow/wrist branch, and only some
    # branches keep the closing axis radial through the cup's 5.2mm wall.
    # joint_space_lerp (not Cartesian IK) back to that exact reference
    # configuration before any Cartesian approach to the cup.
    right_home_qpos_5 = [
        randomizer.robot_equilibrium_qpos[model.jnt_qposadr[model.joint(jn).id]]
        for jn in RIGHT_ARM_JOINTS_5
    ]
    qpos_branch_reset = joint_space_lerp(configuration, model, RIGHT_ARM_JOINTS_5,
                                          right_home_qpos_5, steps=40)
    grip_branch_reset = [("right", GRASP_OPEN)] * len(qpos_branch_reset)

    # Bypass waypoint REMOVED (Option A). Two attempts (single waypoint at
    # z=1.30, dual at z=1.10/0.95) both made things worse, verified with
    # real per-seed logs: the bottle-on-path check used the straight-line
    # home->cup_grasp segment, but the arm never actually travels that
    # segment (mink's CollisionAvoidanceLimit already routes it, same as
    # the isolated reach_cup.py test that got 10/10 with no bypass at all).
    # Home sits close to the bottle's resting position on several seeds
    # (seed 7: home-to-bottle = 0.115m, under the bypass's own 0.19m
    # trigger threshold) even when the bottle is nowhere near the actual
    # swept path, so the check fired on seeds where the bottle was never
    # actually in the way. Where it fired, the extra 1.30m-altitude
    # waypoint stalled the arm (verified: R_site never reached it, e.g.
    # seed 7 step 850 z=1.266 vs commanded 1.30), so by the time IK caught
    # up the descent had already run long enough for the z-gated weld to
    # fire against the wrong part of the cup, and the carry became
    # unstable (both seed 6 and seed 7 logs show the cup ending up tipped
    # after a wide, unstable swing). Deleting this entirely.
    right_cup_waypoints = [
        (handle_frame, cup_approach_xyz, None, GRASP_OPEN, "cup"),
        (handle_frame, cup_grasp_xyz, None, GRASP_OPEN, "cup"),
        (handle_frame, cup_grasp_xyz, None, GRASP_CLOSED, "cup"),
        (handle_frame, cup_above, None, GRASP_CLOSED, "cup"),
        (handle_frame, np.array([cup_place_xy[0], cup_place_xy[1], CUP_TRANSIT_Z]),
         None, GRASP_CLOSED, "cup"),
        (handle_frame, cup_place_above, None, GRASP_CLOSED, "cup"),
        (handle_frame, cup_place_pos, None, GRASP_CLOSED, "cup"),
        (handle_frame, cup_place_pos, None, GRASP_OPEN, "cup"),
        (handle_frame, cup_place_above, None, GRASP_OPEN, "cup"),
        (handle_frame, right_home_stage, None, GRASP_OPEN, None),
    ]

    qpos_cup_tail, grip_cup_tail = waypoint_sequence(
        configuration, model, right_cup_waypoints, steps_per_segment=70,
        secondary_frame="left_gripperframe", secondary_xyz=left_home_stage,
        secondary_quat=None, secondary_pos_cost=1.0,
    )
    qpos_cup_tail = qpos_branch_reset + qpos_cup_tail
    grip_cup_tail = grip_branch_reset + grip_cup_tail

    qpos_tail = qpos_tail + qpos_cup_tail
    grip_tail = grip_tail + grip_cup_tail
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
        "plate_place_xy": place_xy,
        "plate_actual_xy_at_phase4_start": plate_actual_xy,
        "cup_place_xy": cup_place_xy,
        "cup_grasp_xyz": cup_grasp_xyz,
        "cup_actual_xy_at_phase4_start": cup_actual_xy_now,
        "plate_place_z": float(place_pos[2]),
        "cup_place_z": float(cup_place_pos[2]),
    }
    return qpos_trace, gripper_trace, grasp_events