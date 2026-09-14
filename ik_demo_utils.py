"""
IK-waypoint helpers with dense frame-by-frame trajectory IK and continuous
collision avoidance against obstacles.
"""
import mink
import mujoco
import numpy as np


def actuator_qpos_addresses(model):
    addrs = []
    for act_id in range(model.nu):
        joint_id = model.actuator_trnid[act_id, 0]
        addrs.append(int(model.jnt_qposadr[joint_id]))
    return addrs


def _geoms_by_body_prefix(model, prefix):
    gids = []
    for gid in range(model.ngeom):
        bid = model.geom_bodyid[gid]
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if bname.startswith(prefix):
            gids.append(gid)
    return gids


def get_avoidance_limits(model, exclude_body=None):
    # NOTE: deliberately does NOT include "table" -- the plate rim itself
    # sits only 1-2cm above the table surface (measured: rim mid-thickness
    # is 4-9mm above the plate's CoM, table top is ~9mm below that), so any
    # avoidance limit against the table with a few-cm minimum distance would
    # make the grasp itself unreachable. Table clearance during TRANSIT is
    # instead guaranteed explicitly by TRANSIT_Z in scripted_episode_ik.py
    # (verified >3cm above the tallest fixed obstacle) rather than via this
    # soft velocity-limiting constraint, which can't distinguish "near table
    # because grasping" from "near table because transiting badly".
    #
    # minimum_distance_from_collisions raised from 0.015 to 0.03 (3cm) for
    # bottle/cup/drawer/plate -- this IS a global change (every IK solve,
    # both arms), addressing the measured bottle/cup transit clearance
    # requirement. The wrist camera mount (left_wrist_camera_mount,
    # left_wrist_camera, children of left_gripper) is already covered here:
    # `left_geoms` below is built by "left_" body-name PREFIX match, which
    # catches the camera bodies along with the rest of the arm -- no
    # separate camera-specific limit is needed, they get the same 3cm.
    tall_objects = ["bottle", "bowl", "plate", "cup", "drawer"]
    skip_geoms = {"drawer_handle"}

    obj_geoms = []
    for gid in range(model.ngeom):
        bid = model.geom_bodyid[gid]
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if gname in skip_geoms:
            continue
        if (bname in tall_objects or any(x in gname for x in tall_objects)) and bname != exclude_body:
            obj_geoms.append(gid)

    left_geoms = _geoms_by_body_prefix(model, "left_")
    right_geoms = _geoms_by_body_prefix(model, "right_")

    limits = []
    for arm_geoms in (left_geoms, right_geoms):
        if not arm_geoms or not obj_geoms:
            continue
        lim = mink.CollisionAvoidanceLimit(
            model,
            geom_pairs=[(arm_geoms, obj_geoms)],
            minimum_distance_from_collisions=0.03,
            collision_detection_distance=0.08,
        )
        limits.append(lim)

    left_base_id = model.body("left_base").id
    right_base_id = model.body("right_base").id
    left_arm_geoms = mink.get_subtree_geom_ids(model, left_base_id)
    right_arm_geoms = mink.get_subtree_geom_ids(model, right_base_id)
    if left_arm_geoms and right_arm_geoms:
        cross_limit = mink.CollisionAvoidanceLimit(
            model,
            geom_pairs=[(left_arm_geoms, right_arm_geoms)],
            minimum_distance_from_collisions=0.03,
            collision_detection_distance=0.06,
        )
        limits.append(cross_limit)
        limits.append(mink.ConfigurationLimit(model))

    return limits


def joint_space_lerp(configuration, model, joint_names, target_values, steps=30):
    """Linearly interpolate specific joints from their CURRENT configuration
    values to target_values, holding every other DOF fixed. Bypasses mink's
    Cartesian IK (and its nullspace ambiguity) entirely for this segment --
    use it to route through a manually-verified, known-collision-free
    configuration instead of letting the solver pick an arbitrary elbow/wrist
    branch on its own.

    NOTE: this does not run CollisionAvoidanceLimit -- it trusts that the
    straight joint-space line from the current (already-safe) pose to the
    target (manually verified) pose stays safe. Verify with a contact check
    on your real meshes before trusting it in production.
    """
    q_start = configuration.q.copy()
    adrs = [model.jnt_qposadr[model.joint(jn).id] for jn in joint_names]
    start_vals = [q_start[adr] for adr in adrs]

    qpos_trace = []
    for i in range(steps):
        s = (i + 1) / steps
        q = q_start.copy() if i == 0 else qpos_trace[-1].copy()
        for adr, sv, tv in zip(adrs, start_vals, target_values):
            q[adr] = (1.0 - s) * sv + s * tv
        qpos_trace.append(q)

    configuration.update(qpos_trace[-1])
    return qpos_trace


def compute_gripper_closing_axis_local(model, data, gripper_body, moving_jaw_body, gripper_joint,
                                        reference_qpos_5, arm_joint_names, dtheta=0.02):
    """Measure the gripper's closing-motion direction, expressed in the
    gripper site's own local frame. This is a MECHANISM CONSTANT: it does
    not depend on the arm's shoulder/elbow/wrist_roll configuration (verified
    empirically -- recomputing it from two unrelated arm poses gives the same
    vector to ~1e-14). Only recompute this if the gripper geometry itself
    changes (different STL, different joint placement).

    Returns the unit vector in the `<gripper_body>frame` site's local frame.
    """
    for jn, val in zip(arm_joint_names, reference_qpos_5):
        data.qpos[model.jnt_qposadr[model.joint(jn).id]] = val
    g_adr = model.jnt_qposadr[model.joint(gripper_joint).id]
    data.qpos[g_adr] = 0.3
    mujoco.mj_forward(model, data)

    site_id = model.site(f"{gripper_body}frame").id
    tcp_world = data.site_xpos[site_id].copy()
    site_R = data.site_xmat[site_id].reshape(3, 3).copy()

    jaw_bid = model.body(moving_jaw_body).id
    p = data.xpos[jaw_bid]
    R = data.xmat[jaw_bid].reshape(3, 3)
    local_on_jaw = R.T @ (tcp_world - p)

    def world_from_local(local_pt):
        p = data.xpos[jaw_bid]
        R = data.xmat[jaw_bid].reshape(3, 3)
        return p + R @ local_pt

    data.qpos[g_adr] = 0.3 + dtheta
    mujoco.mj_forward(model, data)
    p_plus = world_from_local(local_on_jaw)
    data.qpos[g_adr] = 0.3 - dtheta
    mujoco.mj_forward(model, data)
    p_minus = world_from_local(local_on_jaw)

    tangent_world = p_plus - p_minus
    tangent_world /= np.linalg.norm(tangent_world)
    return site_R.T @ tangent_world


def _rotation_from_a_to_b(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s < 1e-8:
        return np.eye(3) if c > 0 else -np.eye(3) + 2 * np.outer(a, a)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))


def _plate_surface_z_range_at_xy(model, data, plate_bid, world_xy, xy_tolerance=0.004):
    """Return (z_min, z_max) of the plate's collision-mesh material within
    xy_tolerance (in world XY) of world_xy, or None if no mesh vertex is
    that close in XY.

    Used by compute_plate_rim_grasp to find the actual material surface at a
    rim point. This is required because the plate is NOT a uniform-thickness
    disc: measured radial profile shows material at r=0.9R sits at z_rel
    +2.95..+11.59 mm above the body origin, while material at r=0.5R sits at
    z_rel -9.79..-4.60 mm -- a ~16 mm vertical difference between rim and
    center. The plate body origin (== the z used by `plate_z = xpos.z`) sits
    between them, so positioning the gripper site at plate_z puts it under
    the rim material (top jaw inside the plate) and above the center
    material (both jaws in the air). Measured the hard way: robot touched
    the plate with the TIP of the bottom jaw while the gripper was still
    open, then the weld fired and the plate rode the tip. This helper gives
    the caller the mid-thickness of the actual material at the rim point, so
    the closing axis runs through the plate's substance.
    """
    plate_pos = data.xpos[plate_bid].copy()
    plate_R = np.zeros(9)
    mujoco.mju_quat2Mat(plate_R, data.xquat[plate_bid])
    plate_R = plate_R.reshape(3, 3)

    all_verts = []
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != plate_bid:
            continue
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[gid]
        v = model.mesh_vert[model.mesh_vertadr[mid]:
                            model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        gpos = model.geom_pos[gid]
        gquat = model.geom_quat[gid]
        gr = np.zeros(9)
        mujoco.mju_quat2Mat(gr, gquat)
        gr = gr.reshape(3, 3)
        all_verts.append((gr @ v.T).T + gpos)
    if not all_verts:
        return None
    V = np.concatenate(all_verts, axis=0)
    world = (plate_R @ V.T).T + plate_pos

    dx = world[:, 0] - world_xy[0]
    dy = world[:, 1] - world_xy[1]
    rr = np.sqrt(dx * dx + dy * dy)
    mask = rr < xy_tolerance
    if mask.sum() == 0:
        return None
    return float(world[mask, 2].min()), float(world[mask, 2].max())


def compute_plate_rim_grasp(model, data, configuration, plate_center_xy, left_base_xy,
                             plate_radius, plate_z, closing_axis_local, arm_joint_names,
                             gripper_frame_name, home_stage_xyz, rim_fraction=0.9,
                             outside_margin=0.06, sweep_steps_per_segment=15, phi_step_deg=15,
                             grasp_side="near", max_orientation_error_deg=25.0,
                             plate_body_name="plate"):
    """Find the plate-rim grasp point + orientation for THIS episode's actual
    plate position. Pinches the rim vertically (closing_axis_local -> world Z)
    so the jaws squeeze the plate's top/bottom surface at the rim (thin, well
    within the gripper's mouth) instead of the flat center (which slips under
    load).

    IMPORTANT, learned the hard way on the real robot mesh: the left arm is
    only 5-DOF (5 revolute joints + gripper). A fully-specified 6D target
    (3 position + 3 orientation) generally has NO exact solution for a 5-DOF
    chain. mink's FrameTask resolves this as a weighted least-squares
    compromise, and which compromise it lands on is EXTREMELY sensitive to
    step resolution -- a coarse check (sweep_steps_per_segment=6, an earlier
    version of this function) can report a converged, well-margined solution
    that a full-resolution replay (steps=15-30) reveals is actually a false
    positive: same quat, position error balloons from ~1cm to ~9cm because
    the coarse path skipped past the true (bad) equilibrium the fine path
    correctly settles into. Verified directly on this rig: steps=4/6/8 gave
    err < 5cm, steps=10 gave 8.6cm, steps>=15 all converged to the SAME
    9.4cm error (a real, reproducible attractor, not noise or slow
    convergence -- 50 steps gives an identical result to 15).

    Consequently this function ranks candidates on BOTH the achieved
    position error AND the achieved orientation error (angle between the
    resulting closing axis and world vertical) -- NOT on joint-limit margin,
    which does not correlate with whether the least-squares compromise is
    any good. And it evaluates every candidate at close to full resolution
    (sweep_steps_per_segment=15 default) because coarser checks are not a
    reliable proxy for the fine-resolution outcome, as shown above -- this
    makes the sweep slower (~30s for 48 candidates on the real meshes) but
    the 6-step version's speed was not real, it was reporting wrong answers
    fast.

    max_orientation_error_deg: candidates whose achieved closing axis is
    more than this many degrees from vertical are rejected even if their
    position error is excellent -- a badly tilted closing axis will not
    pinch the rim top/bottom the way this grasp is designed to. On the one
    real seed tested so far the best achievable compromise was ~20 degrees
    off vertical, not 0 -- this is a real, seed-dependent kinematic limit
    of the 5-DOF arm, not a bug; if every candidate gets rejected, that
    seed's plate position may need `grasp_side="far"` or a smaller
    `rim_fraction` (closer to center, shorter reach) instead.

    The grasp z is placed at the MID-THICKNESS of the plate material at the
    rim point, not at plate_z (which is the plate body's CoM z). These are
    NOT the same: the plate has a raised rim (r=0.9R material is ~7 mm above
    CoM, r=0.5R material is ~7 mm below), so using plate_z as the site z
    puts the closing axis through the wrong plane. Measured consequence:
    tip-of-jaw-only contact while the gripper was open, then weld fired and
    the plate rode the tip. `plate_z` is kept as a fallback for cases where
    the mesh query finds no nearby vertices (should not happen for a
    well-formed plate, but a wrong-but-close z is better than a crash).
    """
    plate_bid = model.body(plate_body_name).id

    to_base = left_base_xy - plate_center_xy
    to_base_dir = to_base / np.linalg.norm(to_base)
    if grasp_side == "far":
        to_base_dir = -to_base_dir
    elif grasp_side != "near":
        raise ValueError(f"grasp_side must be 'near' or 'far', got {grasp_side!r}")
    rim_xy = plate_center_xy + rim_fraction * plate_radius * to_base_dir

    surface = _plate_surface_z_range_at_xy(model, data, plate_bid, rim_xy)
    if surface is not None:
        z_lo, z_hi = surface
        rim_z = 0.5 * (z_lo + z_hi)
    else:
        # Mesh query found no vertices within xy_tolerance of the rim
        # point. Fall back to the CoM z passed in by the caller -- worse,
        # but still a valid position, and lets the sweep run rather than
        # crashing the whole collection pass.
        print(f"[compute_plate_rim_grasp] no plate mesh vertices within "
              f"xy_tolerance of rim point {rim_xy}; falling back to "
              f"plate_z={plate_z:.4f}")
        rim_z = float(plate_z)
    rim_xyz = np.array([rim_xy[0], rim_xy[1], rim_z])
    outside_xyz = rim_xyz + outside_margin * np.array([to_base_dir[0], to_base_dir[1], 0.0])

    q0 = configuration.q.copy()
    best = None
    for e1_sign in (1.0, -1.0):
        R0 = _rotation_from_a_to_b(closing_axis_local, np.array([0.0, 0.0, e1_sign]))
        for phi_deg in range(0, 360, phi_step_deg):
            phi = np.radians(phi_deg)
            Rz = np.array([[np.cos(phi), -np.sin(phi), 0],
                            [np.sin(phi), np.cos(phi), 0],
                            [0, 0, 1]])
            R = Rz @ R0
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, R.flatten())

            cfg = mink.Configuration(model)
            cfg.update(q0)
            waypoints = [
                (gripper_frame_name, home_stage_xyz, None, 1.2, None),
                (gripper_frame_name, outside_xyz, quat, 1.2, plate_body_name),
                (gripper_frame_name, rim_xyz, quat, 1.2, plate_body_name),
            ]
            try:
                waypoint_sequence(cfg, model, waypoints, steps_per_segment=sweep_steps_per_segment)
            except Exception:
                continue
            transform = cfg.get_transform_frame_to_world(gripper_frame_name, "site")
            final_site = transform.translation()
            final_R = transform.rotation().as_matrix()
            pos_err_cm = np.linalg.norm(final_site - rim_xyz) * 100
            closing_world = final_R @ closing_axis_local
            angle_from_vertical_deg = np.degrees(np.arccos(np.clip(abs(closing_world[2]), 0, 1)))

            if angle_from_vertical_deg > max_orientation_error_deg:
                continue
            score = pos_err_cm + 0.3 * angle_from_vertical_deg
            if best is None or score < best[0]:
                best = (score, pos_err_cm, angle_from_vertical_deg, quat.copy())

    if best is None:
        raise RuntimeError(
            "compute_plate_rim_grasp: no candidate satisfied max_orientation_error_deg "
            f"({max_orientation_error_deg}) for this plate position. Try grasp_side='far', "
            "a smaller rim_fraction, or relax max_orientation_error_deg."
        )
    _, pos_err_cm, angle_from_vertical_deg, quat = best

    # IMPORTANT: scripted_episode_ik.py's tail waypoints may pass
    # target_quat=None on the open-approach waypoints and plate_quat on the
    # closed-gripper waypoints (with a lower ori_cost so the 5-DOF arm does
    # not sacrifice position for orientation). The sweep above uses quat on
    # ALL its waypoints, which is a stronger constraint than what actually
    # executes -- so this function ALSO re-verifies with the SAME
    # target_quat=None path the trajectory uses for its approach, and
    # returns THAT number. It is the trustworthy signal for whether this
    # specific (randomized) plate position is reachable at all -- use it to
    # skip/flag episodes the same way SKIP_RETRIEVE already does for
    # cutlery, rather than silently shipping a "successful-looking"
    # trajectory where the gripper never touched anything.
    #
    # The verify pass is wrapped in try/except because mink's QP solver can
    # raise NoSolutionFound on genuinely infeasible seeds (e.g. plate placed
    # past the arm's reach under a full collision-avoidance limit set). A
    # single unreachable seed must not crash the whole dataset-collection
    # run -- treat it as "plate unreachable", return 999.0, and let the
    # caller (collect_demonstrations.py) skip this episode.
    verify_cfg = mink.Configuration(model)
    verify_cfg.update(q0)
    verify_waypoints = [
        (gripper_frame_name, home_stage_xyz, None, 1.2, None),
        (gripper_frame_name, outside_xyz, None, 1.2, plate_body_name),
        (gripper_frame_name, rim_xyz, None, 1.2, plate_body_name),
    ]
    try:
        waypoint_sequence(verify_cfg, model, verify_waypoints, steps_per_segment=20)
        verify_final = verify_cfg.get_transform_frame_to_world(
            gripper_frame_name, "site"
        ).translation()
        true_pos_err_cm = float(np.linalg.norm(verify_final - rim_xyz) * 100)
    except Exception as e:
        print(f"[compute_plate_rim_grasp] verify pass failed "
              f"({type(e).__name__}) -- treating plate as unreachable on this seed")
        true_pos_err_cm = 999.0

    return rim_xyz, outside_xyz, quat, true_pos_err_cm, angle_from_vertical_deg


def compute_plate_rim_grasp_adaptive(model, data, configuration, plate_center_xy, left_base_xy,
                                      plate_radius, plate_z, closing_axis_local, arm_joint_names,
                                      gripper_frame_name, home_stage_xyz,
                                      max_angle_deg=15.0, max_pos_err_cm=3.0, **kwargs):
    """Try several (rim_fraction, grasp_side) combinations and return the
    first that clears BOTH the position and angle thresholds.

    Measured directly on this rig (reachability grid, 150 cells over
    x in [-0.35,0.05], y in [-0.25,0.05]): the FIXED default
    (rim_fraction=0.9, grasp_side="near") achieves angle<10 in only 51/150
    cells overall, and only 1/30 cells inside the declared plate range
    x in [-0.32,-0.02], y in [-0.21,-0.089] -- 29/30 cells there sit in an
    18.7-21 degree dead zone, a real kinematic attractor of the 5-DOF chain
    at that specific rim point, not a bug (confirmed: bit-identical angle
    across dt/step-count/ori_cost variations in earlier testing). Re-scanning
    9 representative points spanning the declared range with a SMALL set of
    (rim_fraction, grasp_side) alternatives found <4 degrees at every single
    one -- the task is achievable, the fixed default parameterization was
    simply the wrong choice, not a hard limit of the arm.

    Order tried: (0.9,near) first (cheapest, matches old behavior when it
    happens to work), then rim_fraction 0.65/0.5 with both sides. Returns as
    soon as a combination clears both thresholds.
    """
    combos = [(0.9, "near"), (0.9, "far"), (0.65, "near"), (0.65, "far"),
              (0.5, "near"), (0.5, "far")]
    best = None
    for rim_frac, side in combos:
        cfg = mink.Configuration(model)
        cfg.update(configuration.q.copy())
        try:
            rim_xyz, outside_xyz, quat, pos_err_cm, angle_deg = compute_plate_rim_grasp(
                model, data, cfg,
                plate_center_xy=plate_center_xy, left_base_xy=left_base_xy,
                plate_radius=plate_radius, plate_z=plate_z,
                closing_axis_local=closing_axis_local, arm_joint_names=arm_joint_names,
                gripper_frame_name=gripper_frame_name, home_stage_xyz=home_stage_xyz,
                rim_fraction=rim_frac, grasp_side=side,
                max_orientation_error_deg=90.0, **kwargs,
            )
        except RuntimeError:
            continue
        if best is None or (pos_err_cm + 0.3 * angle_deg) < (best[3] + 0.3 * best[4]):
            best = (rim_xyz, outside_xyz, quat, pos_err_cm, angle_deg, rim_frac, side)
        if pos_err_cm <= max_pos_err_cm and angle_deg <= max_angle_deg:
            return rim_xyz, outside_xyz, quat, pos_err_cm, angle_deg, rim_frac, side

    if best is None:
        raise RuntimeError("compute_plate_rim_grasp_adaptive: every combination failed to converge.")
    return best


def activate_grasp_connect(model, data, eq_name, body1_name, body2_name, world_anchor_pt):
    """Activate a `connect` equality constraint, anchored at the CURRENT physical
    grasp point. Must be called AFTER the gripper has physically closed (contact
    settled) and BEFORE any pulling motion. Call deactivate_grasp_connect to release.

    This does not require correct mesh collision geometry -- it directly enforces
    "these two bodies stay joined at this point," which is what a closed gripper
    with sufficient friction is physically doing anyway. It replaces reliance on
    the moving jaw's convex-hull mesh contact for load-bearing grip.
    """
    eq_id = model.equality(eq_name).id
    b1 = model.body(body1_name).id
    b2 = model.body(body2_name).id

    def world_to_local(bid, world_pt):
        p = data.xpos[bid]
        R = data.xmat[bid].reshape(3, 3)
        return R.T @ (world_pt - p)

    model.eq_data[eq_id, 0:3] = world_to_local(b1, world_anchor_pt)
    model.eq_data[eq_id, 3:6] = world_to_local(b2, world_anchor_pt)
    data.eq_active[eq_id] = 1
    model.eq_active0[eq_id] = 1


def deactivate_grasp_connect(model, data, eq_name):
    eq_id = model.equality(eq_name).id
    data.eq_active[eq_id] = 0
    model.eq_active0[eq_id] = 0


def compute_cup_grasp_target(model, data, right_base_xy, jaw_half_width=0.010, grasp_depth=0.035):
    """Measure the cup's real collision-mesh geometry (CoACD decomposition,
    not a primitive) and return a wall-grasp target: a point on the cup
    wall, on the side facing right_base_xy, at a height below the (thicker,
    rolled) rim where the straight wall lets the jaws close.

    Ported directly from reach_cup.py's cup_top_geometry + grasp-target
    logic (verified: 10/10 seeds reach with 0.00cm error, 6-11 jaw-cup
    contacts each, using target_quat=None -- do not add an orientation
    target here, it costs 6-8cm of position error on this 5-DOF arm).

    Returns (grasp_xyz, approach_xyz, inner_r, outer_r).
    """
    cup_bid = model.body("cup").id
    cup_pos = data.xpos[cup_bid].copy()
    cup_R = np.zeros(9)
    mujoco.mju_quat2Mat(cup_R, data.xquat[cup_bid])
    cup_R = cup_R.reshape(3, 3)

    verts = []
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != cup_bid:
            continue
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[gid]
        v = model.mesh_vert[model.mesh_vertadr[mid]:
                            model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        gpos = model.geom_pos[gid]
        gr = np.zeros(9)
        mujoco.mju_quat2Mat(gr, model.geom_quat[gid])
        gr = gr.reshape(3, 3)
        verts.append((gr @ v.T).T + gpos)
    V = np.concatenate(verts, axis=0)
    Vw = (cup_R @ V.T).T + cup_pos

    top_z = float(Vw[:, 2].max())
    grasp_z = top_z - grasp_depth
    band = np.abs(Vw[:, 2] - grasp_z) < 0.005
    if band.sum() < 3:
        height = top_z - cup_pos[2]
        band = Vw[:, 2] > (top_z - 0.15 * height)
    r_xy = np.linalg.norm(Vw[band, :2] - cup_pos[:2], axis=1)
    inner_r = float(r_xy.min())
    outer_r = float(r_xy.max())
    wall_mid_r = 0.5 * (inner_r + outer_r)

    to_base = right_base_xy - cup_pos[:2]
    to_base_dir = to_base / np.linalg.norm(to_base)
    grasp_xy = cup_pos[:2] + wall_mid_r * to_base_dir
    grasp_xyz = np.array([grasp_xy[0], grasp_xy[1], grasp_z])
    approach_xyz = grasp_xyz + np.array([0.0, 0.0, 0.08])
    return grasp_xyz, approach_xyz, inner_r, outer_r


def activate_grasp_weld(model, data, eq_name, body1_name, body2_name):
    """Weld body2 rigidly to body1 at their CURRENT relative pose (6-DOF lock
    -- translation + orientation). Use for objects that must not rotate
    relative to the grasping hand once gripped (a free-floating plate on a
    connect/3-DOF-point constraint will swing like a pendulum around the
    anchor as the arm moves -- verified, ~7cm lateral drift over an 8cm lift).

    body1_name MUST be a body that does NOT itself rotate as the gripper
    hinge closes -- i.e. the FIXED wrist structure (e.g. "left_gripper"),
    NOT the moving jaw (e.g. "left_moving_jaw_so101_v1"). If you weld to the
    moving jaw and activate before the gripper has finished closing, the
    welded object is rigidly attached to a frame that keeps rotating for
    the remainder of the close motion, and gets flung with it. Verified
    directly: welding to the moving jaw mid-close gives ~21 degrees of
    rotation drift and ~3.4cm position drift by the time the gripper
    finishes closing; welding to the fixed wrist body under the exact same
    conditions gives ~1cm / ~0.5 degrees once settled -- delaying activation
    until the gripper is fully closed does NOT fix this on its own (tested:
    0.2 degree difference, i.e. no meaningful effect) if you're still
    welding to the moving jaw. The body choice is what matters, not timing.

    eq_data layout for MuJoCo's <weld>, confirmed empirically on this
    MuJoCo version (not assumed): eq_data[0:3] = anchor (body1 frame),
    eq_data[3:6] = relpose position (body1 frame), eq_data[6:10] = relpose
    quaternion wxyz, eq_data[10] = torquescale (default 1.0, left
    unmodified here).
    """
    eq_id = model.equality(eq_name).id
    b1 = model.body(body1_name).id
    b2 = model.body(body2_name).id
    p1 = data.xpos[b1].copy()
    R1 = data.xmat[b1].reshape(3, 3)
    p2 = data.xpos[b2].copy()
    R2 = data.xmat[b2].reshape(3, 3)
    rel_pos = R1.T @ (p2 - p1)
    R_rel = R1.T @ R2
    rel_quat = np.zeros(4)
    mujoco.mju_mat2Quat(rel_quat, R_rel.flatten())
    model.eq_data[eq_id, 0:3] = 0.0
    model.eq_data[eq_id, 3:6] = rel_pos
    model.eq_data[eq_id, 6:10] = rel_quat
    data.eq_active[eq_id] = 1
    model.eq_active0[eq_id] = 1


def deactivate_grasp_weld(model, data, eq_name):
    eq_id = model.equality(eq_name).id
    data.eq_active[eq_id] = 0
    model.eq_active0[eq_id] = 0


def solve_ik_step(configuration, model, frame_name, target_xyz, target_quat=None,
                  limits=None, n_iter=8, dt=0.01, pos_cost=1.0, ori_cost=0.15,
                  frame_type="site",
                  secondary_frame=None, secondary_xyz=None, secondary_quat=None,
                  secondary_pos_cost=1.0, secondary_ori_cost=0.15):
    """Single IK step. If secondary_* is given, a second FrameTask pins
    that frame at the secondary target in every solve -- used for bimanual
    ops where one arm moves and the other must hold its pose.

    ori_cost is a soft weight: when the chain cannot satisfy position AND
    orientation simultaneously (e.g. 5-DOF arm + 6-DOF target), a lower
    ori_cost tells the QP to sacrifice orientation first and keep position.
    Used by the plate-place phase to hold a vertical "sandwich" grasp while
    still hitting the rim accurately.
    """
    task = mink.FrameTask(
        frame_name=frame_name,
        frame_type=frame_type,
        position_cost=pos_cost,
        orientation_cost=ori_cost if target_quat is not None else 0.0,
    )
    if target_quat is not None:
        target = mink.SE3.from_rotation_and_translation(mink.SO3(target_quat), target_xyz)
    else:
        target = mink.SE3.from_translation(target_xyz)
    task.set_target(target)

    tasks = [task]
    if secondary_frame is not None and secondary_xyz is not None:
        sec_task = mink.FrameTask(
            frame_name=secondary_frame,
            frame_type=frame_type,
            position_cost=secondary_pos_cost,
            orientation_cost=secondary_ori_cost if secondary_quat is not None else 0.0,
        )
        if secondary_quat is not None:
            sec_target = mink.SE3.from_rotation_and_translation(
                mink.SO3(secondary_quat), secondary_xyz)
        else:
            sec_target = mink.SE3.from_translation(secondary_xyz)
        sec_task.set_target(sec_target)
        tasks.append(sec_task)

    if limits is None:
        limits = get_avoidance_limits(model)

    for _ in range(n_iter):
        vel = mink.solve_ik(configuration, tasks, dt=dt, solver="daqp", limits=limits)
        configuration.integrate_inplace(vel, dt=dt)

    err = np.linalg.norm(task.compute_error(configuration)[:3])
    return configuration.q.copy(), err


def waypoint_sequence(configuration, model, waypoints, steps_per_segment=20,
                      frame_type="site",
                      ori_cost=0.15,
                      secondary_frame=None, secondary_xyz=None, secondary_quat=None,
                      secondary_pos_cost=1.0, secondary_ori_cost=0.15):
    """
    Dense frame-by-frame IK over a waypoint list, min-jerk interpolation,
    CollisionAvoidanceLimit at every step.

    gripper_trace entries are ("left"|"right", value) tuples or None.
    Side is inferred from the waypoint's frame name -- avoids the old bug
    where every gripper_trace value was applied to right_gripper only,
    silently leaving the left gripper uncommanded.

    ori_cost is forwarded to solve_ik_step for every step in this call, so
    a caller can dial down the orientation weight for a whole sequence (used
    by the plate-place phase: hold the vertical sandwich grasp with
    ori_cost=0.1 so the 5-DOF arm doesn't sacrifice rim position for it).
    """
    qpos_trace = []
    gripper_trace = []
    current_poses = {}

    for wp in waypoints:
        frame_name = wp[0]
        target_xyz = wp[1]
        target_quat = wp[2]
        gripper_val = wp[3]
        exclude_body = wp[4] if len(wp) > 4 else None

        if frame_name not in current_poses:
            current_poses[frame_name] = configuration.get_transform_frame_to_world(
                frame_name, frame_type
            ).translation().copy()

        start_xyz = current_poses[frame_name]
        limits = get_avoidance_limits(model, exclude_body=exclude_body)

        t_vals = np.linspace(0, 1, steps_per_segment)
        s_vals = 10 * t_vals**3 - 15 * t_vals**4 + 6 * t_vals**5

        for s in s_vals:
            curr_target_xyz = (1.0 - s) * start_xyz + s * np.asarray(target_xyz)

            q_step, err = solve_ik_step(
                configuration, model, frame_name, curr_target_xyz,
                target_quat=target_quat, limits=limits,
                n_iter=8, dt=0.01, frame_type=frame_type,
                ori_cost=ori_cost,
                secondary_frame=secondary_frame,
                secondary_xyz=secondary_xyz,
                secondary_quat=secondary_quat,
                secondary_pos_cost=secondary_pos_cost,
                secondary_ori_cost=secondary_ori_cost,
            )
            qpos_trace.append(q_step)

            if gripper_val is None:
                gripper_trace.append(None)
            elif "left" in frame_name:
                gripper_trace.append(("left", gripper_val))
            elif "right" in frame_name:
                gripper_trace.append(("right", gripper_val))
            else:
                gripper_trace.append(None)

        current_poses[frame_name] = np.asarray(target_xyz).copy()

    return qpos_trace, gripper_trace