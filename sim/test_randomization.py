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

TABLE_HALF_X = 0.46
TABLE_HALF_Y = 0.32
DRAWER_Y_CENTER = 0.24
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
    """Reconstructs an object's footprint from its ACTUAL current geometry.
    For capsules (cutlery), uses randomizer.true_capsule_endpoints -- the
    single authoritative source, computed directly from the real applied
    quaternion and the real PCA long axis. An earlier version extracted an
    abstract 'yaw' from the quaternion and recomposed via _capsule_endpoints;
    that assumed the object's reference orientation was identity, which
    broke once cutlery's real reference became the PCA flat quat (itself a
    substantial rotation) -- the two reconstructions silently disagreed."""
    cfg = randomizer.object_cfg[name]
    x, y, _ = get_object_xy_yaw(model, data, name)
    radius = randomizer._placement_radius(name)
    if cfg["shape"] == "circle":
        return Footprint(shape="circle", center=np.array([x, y]), radius=radius)
    p1, p2 = randomizer.true_capsule_endpoints(data, name)
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
    contained (real collision check against the drawer walls -- the same
    proven mechanism _place_in_drawer itself uses to decide acceptance, not
    an idealized capsule-endpoint-vs-boundary comparison) and doesn't
    overlap another in-drawer item.

    An earlier version compared PCA-endpoint positions against the wall
    coordinates directly; that over-reports violations for an asymmetric,
    tapered mesh (a fork's tines don't fill a uniform capsule all the way to
    its PCA-measured tip) -- the real physics found no contact for cases
    the idealized model flagged as "outside". Checking real contacts is
    what actually matters (does it visually/physically clip through a
    wall), and is consistent with the mechanism already governing
    acceptance during placement."""
    di_cfg = randomizer.placement_cfg.get("drawer_interior")
    if not di_cfg:
        return True, "OK (drawer_interior not configured)"
    names = list(randomizer.last_in_drawer_items)
    if not names:
        return True, "OK (no cutlery in drawer this seed)"

    wall_names = ["drawer_wall_left", "drawer_wall_right", "drawer_wall_back", "drawer_wall_front"]
    wall_geoms = {model.geom(n).id for n in wall_names}
    failures = []
    for name in names:
        bid = model.body(name).id
        for c in range(data.ncon):
            con = data.contact[c]
            g1, g2 = con.geom1, con.geom2
            b1, b2 = model.geom_bodyid[g1], model.geom_bodyid[g2]
            if bid in (b1, b2) and (g1 in wall_geoms or g2 in wall_geoms):
                wname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1 if g1 in wall_geoms else g2)
                failures.append(f"{name}: real contact with {wname} (dist={con.dist:.4f})")

    footprints = {n: footprint_from_state(randomizer, model, data, n) for n in names}

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


def check_self_collision(model, data, max_penetration_m=0.001):
    """Real overlap only -- not the small sub-mm 'touching' every resting
    contact shows in MuJoCo's solver (that's the normal representation of
    contact, not instability). Verified directly: the real SO-101's
    equilibrium pose shows 8 left/right contacts, all under 0.3mm -- a
    zero-tolerance check flags this every time despite nothing being
    physically wrong."""
    left_geoms, right_geoms = set(), set()
    for gid in range(model.ngeom):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid]) or ""
        if bname.startswith("left_"):
            left_geoms.add(gid)
        elif bname.startswith("right_"):
            right_geoms.add(gid)
    real_hits = [
        data.contact[c].dist for c in range(data.ncon)
        if ((data.contact[c].geom1 in left_geoms and data.contact[c].geom2 in right_geoms)
            or (data.contact[c].geom1 in right_geoms and data.contact[c].geom2 in left_geoms))
        and data.contact[c].dist < -max_penetration_m
    ]
    ok = len(real_hits) == 0
    msg = "OK" if ok else f"FAIL: {len(real_hits)} contacts deeper than {max_penetration_m*1000:.1f}mm (worst: {min(real_hits)*1000:.2f}mm)"
    return ok, msg


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
