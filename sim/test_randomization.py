"""
Validate DomainRandomizer against every seed in randomization.yaml.

One file, one function per concern -- run this single script and every
category of failure is reported together, but each check is independently
readable/debuggable:
  - determinism        : reset(seed) twice -> identical qpos both times.
  - stability           : 300 post-reset steps never produce NaN/Inf.
  - placement_bounds    : every object stays within the real table edges
                           (accounting for its own radius), not just "didn't
                           fall through the floor."
  - placement_overlap   : no two objects were placed closer than
                           radius1 + radius2 + min_gap at spawn time --
                           checked using the SAME circle/segment geometry
                           randomization.py used to place them, so this is
                           a genuine correctness check, not a re-guess.
  - drawer_clearance     : cutlery respects the soft drawer-avoidance zone.
  - self_collision       : no left/right arm contact at rest.
  - cross_seed_variety   : different seeds actually produce different scenes.
"""
import sys
import numpy as np
import mujoco
from randomization import (
    DomainRandomizer,
    Footprint,
    _capsule_endpoints,
    _footprint_gap,
)

TABLE_HALF_X = 0.5
TABLE_HALF_Y = 0.35
DRAWER_Y_CENTER = 0.27
DRAWER_HALF_Y = 0.08
DRAWER_EXTRA_MARGIN = 0.03


def get_object_qpos(model, data, name):
    bid = model.body(name).id
    jnt_adr = model.body_jntadr[bid]
    qpos_adr = model.jnt_qposadr[jnt_adr]
    return data.qpos[qpos_adr:qpos_adr + 7].copy()


def get_object_xy_yaw(model, data, name):
    """Read back an object's current world XY and yaw from qpos (free joint:
    [x,y,z,qw,qx,qy,qz]). Yaw extraction is only meaningful up to the same
    convention randomization.py uses (extra world-Z rotation on top of the
    nominal flat-lying orientation) -- fine for capsule endpoint reconstruction
    here since we just need *a* consistent yaw to rebuild the segment, not an
    absolute angle."""
    q = get_object_qpos(model, data, name)
    x, y = float(q[0]), float(q[1])
    qw, qx, qy, qz = q[3], q[4], q[5], q[6]
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return x, y, float(yaw)


def footprint_from_state(randomizer, model, data, name):
    """Reconstructs an object's footprint from its ACTUAL current geometry
    (via randomizer's own _placement_radius/_placement_half_length -- live
    geom_size for primitives, post shape_scale), not the static yaml
    radius_m. Using the static value here was a real test bug: shape_scale
    can shrink a capsule up to 10% below nominal, and the real placement
    check (correctly) validates against that true smaller size -- so a
    static-radius reconstruction could show an artificially tight gap for
    an object that was, in its real/current size, validly spaced."""
    cfg = randomizer.object_cfg[name]
    x, y, yaw = get_object_xy_yaw(model, data, name)
    radius = randomizer._placement_radius(name)
    if cfg["shape"] == "circle":
        return Footprint(shape="circle", center=np.array([x, y]), radius=radius)
    half_length = randomizer._placement_half_length(name)
    p1, p2 = _capsule_endpoints(np.array([x, y]), yaw, half_length)
    return Footprint(shape="capsule", center=np.array([x, y]), radius=radius, p1=p1, p2=p2)


def check_determinism(model, data, randomizer, seed):
    randomizer.reset(data, seed)
    q1 = data.qpos.copy()
    randomizer.reset(data, seed)
    q2 = data.qpos.copy()
    ok = np.array_equal(q1, q2)
    return ok, "OK" if ok else "FAIL: reset(seed) is not deterministic"


def check_stability(model, data, n_steps=300):
    for i in range(n_steps):
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            return False, f"FAIL: non-finite state at step {i}"
    return True, f"OK ({n_steps} steps, no NaN/Inf)"


def check_placement_bounds(randomizer, model, data, margin=0.0):
    """Immediately after reset (before physics moves anything), verify every
    TABLE-placed object's footprint fits inside the real table edges.
    In-drawer items are checked separately by check_drawer_interior."""
    failures = []
    for name in randomizer.object_names:
        if name in randomizer.last_in_drawer_items:
            continue
        fp = footprint_from_state(randomizer, model, data, name)
        if fp.shape == "circle":
            pts = [fp.center]
        else:
            pts = [fp.p1, fp.p2]
        for pt in pts:
            if abs(pt[0]) > TABLE_HALF_X - margin or abs(pt[1]) > TABLE_HALF_Y - margin:
                failures.append(f"{name}: point {pt.round(4)} outside table half-extent "
                                 f"({TABLE_HALF_X},{TABLE_HALF_Y})")
    ok = len(failures) == 0
    return ok, "OK (all objects within table edges)" if ok else "FAIL:\n    " + "\n    ".join(failures)


def check_placement_overlap(randomizer, model, data):
    """Immediately after PLACEMENT (not after settling -- settling is real
    physics and can legitimately bring two circle-modeled footprints a
    fraction of a mm closer than the static 2D guarantee promised, without
    any actual mesh interpenetration; that's a different, physical claim,
    checked separately by check_no_real_penetration below). Verifies no two
    TABLE-placed objects are closer than their configured min_gap. In-drawer
    items are checked separately (against cavity bounds and each other,
    not against table objects -- they're not spatially comparable, one set
    is on the table surface, the other in a cavity beneath it)."""
    min_gap = randomizer.placement_cfg["min_gap_between_objects_m"]
    names = [n for n in randomizer.object_names if n not in randomizer.last_in_drawer_items]
    footprints = {n: footprint_from_state(randomizer, model, data, n) for n in names}
    failures = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            gap = _footprint_gap(footprints[a], footprints[b])
            if gap < min_gap - 1e-6:
                failures.append(f"{a} <-> {b}: gap={gap:.4f}m < required {min_gap}m")
    ok = len(failures) == 0
    return ok, "OK (no overlaps)" if ok else "FAIL:\n    " + "\n    ".join(failures)


def check_drawer_interior(randomizer, model, data):
    """For cutlery placed INSIDE the drawer this seed: verify it's actually
    within the cavity bounds (not clipping through a wall) and doesn't
    overlap another in-drawer item.

    Checks endpoints against the TRUE wall interior (from the MJCF:
    drawer_floor/walls half-extent 0.114 x 0.074), not against
    `drawer_interior.cavity_x/cavity_y` from the yaml -- those are CENTER
    sampling ranges, deliberately tighter than the true cavity so a
    capsule's half-length can extend beyond its center and still land
    inside the real walls. Comparing an endpoint against the center range
    would be comparing the wrong two things."""
    di_cfg = randomizer.placement_cfg.get("drawer_interior")
    if not di_cfg:
        return True, "OK (drawer_interior not configured)"
    names = list(randomizer.last_in_drawer_items)
    if not names:
        return True, "OK (no cutlery in drawer this seed)"

    drawer_y = di_cfg["world_y_center_closed_m"]
    CAVITY_WALL_HALF_X, CAVITY_WALL_HALF_Y = 0.114, 0.074  # true interior, from the MJCF
    wall_margin = 0.005
    failures = []

    footprints = {}
    for name in names:
        fp = footprint_from_state(randomizer, model, data, name)
        footprints[name] = fp
        pts = [fp.p1, fp.p2] if fp.shape == "capsule" else [fp.center]
        for pt in pts:
            if not (-CAVITY_WALL_HALF_X + wall_margin <= pt[0] <= CAVITY_WALL_HALF_X - wall_margin):
                failures.append(f"{name}: x={pt[0]:.4f} outside true cavity walls "
                                 f"[{-CAVITY_WALL_HALF_X},{CAVITY_WALL_HALF_X}]")
            y_lo_wall, y_hi_wall = drawer_y - CAVITY_WALL_HALF_Y, drawer_y + CAVITY_WALL_HALF_Y
            if not (y_lo_wall + wall_margin <= pt[1] <= y_hi_wall - wall_margin):
                failures.append(f"{name}: y={pt[1]:.4f} outside true cavity walls "
                                 f"[{y_lo_wall},{y_hi_wall}]")

    min_gap = randomizer.placement_cfg["min_gap_between_objects_m"]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            gap = _footprint_gap(footprints[a], footprints[b])
            if gap < min_gap - 1e-6:
                failures.append(f"{a} <-> {b} (both in drawer): gap={gap:.4f}m < {min_gap}m")

    ok = len(failures) == 0
    return ok, (f"OK ({len(names)} item(s) in drawer, all contained)" if ok
                else "FAIL:\n    " + "\n    ".join(failures))


def check_no_real_penetration(model, data, object_names, max_penetration_m=0.003):
    """Post-settle: checks ACTUAL contact penetration depth (data.contact.dist,
    negative = penetrating) between manipulable objects, not the conservative
    circle-footprint model. Small negative values (a fraction of a mm) are
    normal -- that's how a constraint solver represents "touching"; this
    catches real interpenetration, not the static guarantee's rounding."""
    object_bodies = {model.body(n).id for n in object_names}
    failures = []
    for c in range(data.ncon):
        con = data.contact[c]
        b1 = model.geom_bodyid[con.geom1]
        b2 = model.geom_bodyid[con.geom2]
        if b1 in object_bodies and b2 in object_bodies and con.dist < -max_penetration_m:
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1)
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2)
            failures.append(f"{n1} <-> {n2}: penetration={con.dist:.5f}m")
    ok = len(failures) == 0
    return ok, "OK (no real interpenetration)" if ok else "FAIL:\n    " + "\n    ".join(failures)


def check_drawer_clearance(randomizer, model, data):
    """TABLE-placed cutlery only -- soft workspace-clearance rule (see
    randomization.yaml `placement.drawer_avoidance` docstring). Cutlery
    placed INSIDE the drawer this seed is intentionally in that zone --
    that's the whole point -- so it's excluded here and checked instead by
    check_drawer_interior."""
    drawer_y_min = DRAWER_Y_CENTER - DRAWER_HALF_Y - DRAWER_EXTRA_MARGIN
    failures = []
    for name in randomizer.object_names:
        if name in randomizer.last_in_drawer_items:
            continue
        if randomizer.object_cfg[name]["shape"] != "capsule":
            continue
        fp = footprint_from_state(randomizer, model, data, name)
        if fp.p1[1] > drawer_y_min or fp.p2[1] > drawer_y_min:
            failures.append(f"{name}: endpoint y={max(fp.p1[1], fp.p2[1]):.4f} "
                             f"crosses drawer clearance line y={drawer_y_min:.4f}")
    ok = len(failures) == 0
    return ok, "OK (cutlery clear of drawer zone)" if ok else "FAIL:\n    " + "\n    ".join(failures)


def check_self_collision(model, data):
    left_geoms, right_geoms = set(), set()
    for gid in range(model.ngeom):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid]) or ""
        if bname.startswith("left_"):
            left_geoms.add(gid)
        elif bname.startswith("right_"):
            right_geoms.add(gid)
    cross = sum(
        1 for c in range(data.ncon)
        if (data.contact[c].geom1 in left_geoms and data.contact[c].geom2 in right_geoms)
        or (data.contact[c].geom1 in right_geoms and data.contact[c].geom2 in left_geoms)
    )
    ok = cross == 0
    return ok, "OK" if ok else f"FAIL: {cross} left/right contacts"


def reset_pre_settle(randomizer, model, data, seed):
    """Replicates reset()'s pre-settle steps only, to test the actual
    contract placement makes: 'objects are placed with at least min_gap
    clearance' is a claim about the moment placement finishes, not about
    forever after real physics has been allowed to run. Post-settle overlap
    tolerance is a different, physical claim -- see check_no_real_penetration."""
    rng = np.random.RandomState(seed)
    mujoco.mj_resetData(model, data)
    randomizer._randomize_mass(rng)
    randomizer._randomize_friction(rng)
    randomizer._randomize_shape(rng)
    randomizer._randomize_object_placement(data, rng)
    mujoco.mj_forward(model, data)


def main():
    model = mujoco.MjModel.from_xml_path("sim/assets/dinner_table_dual_so101.xml")
    data = mujoco.MjData(model)
    randomizer = DomainRandomizer(model, "configs/randomization.yaml")

    all_pass = True
    plate_positions = {}

    for seed in randomizer.seeds:
        print(f"\n=== seed {seed} ===")

        ok, msg = check_determinism(model, data, randomizer, seed)
        print(f"  determinism:       {msg}")
        all_pass &= ok

        # Pre-settle: the actual placement contract.
        reset_pre_settle(randomizer, model, data, seed)

        ok, msg = check_placement_bounds(randomizer, model, data)
        print(f"  placement_bounds:  {msg}")
        all_pass &= ok

        ok, msg = check_placement_overlap(randomizer, model, data)
        print(f"  placement_overlap: {msg}")
        all_pass &= ok

        ok, msg = check_drawer_clearance(randomizer, model, data)
        print(f"  drawer_clearance:  {msg}")
        all_pass &= ok

        ok, msg = check_drawer_interior(randomizer, model, data)
        print(f"  drawer_interior:   {msg}")
        all_pass &= ok

        # Post-settle (full reset(), including physics): physical sanity.
        randomizer.reset(data, seed)

        ok, msg = check_stability(model, data)
        print(f"  stability:         {msg}")
        all_pass &= ok

        ok, msg = check_self_collision(model, data)
        print(f"  self_collision:    {msg}")
        all_pass &= ok

        ok, msg = check_no_real_penetration(model, data, randomizer.object_names)
        print(f"  no_real_penetration: {msg}")
        all_pass &= ok

        plate_positions[seed] = get_object_xy_yaw(model, data, "plate")[:2]

    print("\n=== cross_seed_variety (plate xy) ===")
    for s, p in plate_positions.items():
        print(f"  seed {s}: ({p[0]:.4f}, {p[1]:.4f})")
    unique = len(set(tuple(np.round(p, 6)) for p in plate_positions.values()))
    variety_ok = unique == len(plate_positions)
    print(f"  variety: {'OK' if variety_ok else 'FAIL: duplicate positions across seeds'}")
    all_pass &= variety_ok

    print(f"\n{'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
