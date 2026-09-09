"""
Domain randomization for the dinner-table scene.

Owned by: feature/mujoco-scene
Consumed by: /eval (calls reset(seed) once per episode, before stepping)

Design contract with randomization.yaml:
  - `apply_at: episode_reset` -> reset() is the ONLY place any of this runs.
  - Every randomization is relative to NOMINAL values captured from the
    compiled model at construction time, never cumulative. reset(seed=3)
    twice in a row must produce the identical scene both times.
  - Every axis named in the brief (object placement, weight, friction, shape,
    lighting, background) is sampled from a config range, never hardcoded.

Placement / collision avoidance
--------------------------------
Each object is placed independently via rejection sampling: sample a
candidate (x, y, yaw) from its own range in randomization.yaml, check it
against every already-placed object this reset, accept if the gap is >=
min_gap_between_objects_m, else resample (up to `max_attempts`, then fall
back to the object's nominal pose -- per randomization.yaml `placement`).

This is pure 2D geometry, no MuJoCo collision calls -- see the PRD/README for
why (short version: mj_collision needs a forward-kinematics pass per
candidate, which is unnecessary cost and a chicken-and-egg problem when
you're trying to decide the pose that produces that forward pass in the
first place; closed-form circle/segment distance is exact for these shapes
and effectively free).

Two footprint types, matching what the objects actually are:
  - circle:  plate, cup, bottle -- round, footprint = center + radius.
  - capsule: cutlery -- 12cm-long, 6mm-radius. Modeled as an actual line
    segment, NOT a bounding circle of radius (half_length + radius) -- a
    circle that size would be wildly conservative for how thin these
    objects really are and would make four of them nearly impossible to
    place close together. Segment endpoints are derived from the object's
    (x, y, yaw): the capsule geom lies along local Z by default and is
    rotated -90deg about X in the MJCF to lie flat along world Y, so an
    additional yaw rotation about world Z (the same composition
    _apply_pose uses) points its axis along (-sin(yaw), cos(yaw)).

Placement order: objects are placed largest-effective-radius-first (plate,
then cup/bottle, then cutlery). This isn't required for correctness -- it
reduces how often later, smaller objects have to retry, since the biggest
gaps get claimed by the object least able to route around a conflict.
"""
from dataclasses import dataclass, field
from typing import Optional
import math
import numpy as np
import mujoco
import yaml


@dataclass
class NominalState:
    body_pos: dict = field(default_factory=dict)
    body_quat: dict = field(default_factory=dict)
    body_mass: dict = field(default_factory=dict)
    geom_size: dict = field(default_factory=dict)
    geom_friction: dict = field(default_factory=dict)
    geom_rgba: dict = field(default_factory=dict)
    light_pos: dict = field(default_factory=dict)
    cam_pos: dict = field(default_factory=dict)
    cam_quat: dict = field(default_factory=dict)
    up_axis_world: dict = field(default_factory=dict)  # object -> nominal local-Z expressed in world frame


# --------------------------------------------------------------------------- #
# 2D collision geometry: circles and line segments, in table-local XY.
# --------------------------------------------------------------------------- #
def _capsule_endpoints(center_xy, yaw, half_length):
    """World-XY endpoints of a cutlery capsule at (x, y, yaw)."""
    direction = np.array([-math.sin(yaw), math.cos(yaw)])
    center = np.asarray(center_xy)
    return center - half_length * direction, center + half_length * direction


def _point_segment_dist(p, a, b):
    ab = b - a
    denom = np.dot(ab, ab)
    if denom < 1e-12:
        return float(np.linalg.norm(p - a))
    t = np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0)
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))


def _segment_segment_dist(p1, p2, p3, p4):
    """Minimum distance between segments p1-p2 and p3-p4.
    Standard closest-point-between-two-segments algorithm
    (Ericson, Real-Time Collision Detection, ch. 5.1.9), clamped variant."""
    d1, d2, r = p2 - p1, p4 - p3, p1 - p3
    a, e, f = np.dot(d1, d1), np.dot(d2, d2), np.dot(d2, r)

    if a < 1e-12 and e < 1e-12:
        return float(np.linalg.norm(p1 - p3))
    if a < 1e-12:
        t = np.clip(f / e, 0.0, 1.0)
        return float(np.linalg.norm((p3 + t * d2) - p1))

    c = np.dot(d1, r)
    if e < 1e-12:
        s = np.clip(-c / a, 0.0, 1.0)
        return float(np.linalg.norm(p1 + s * d1 - p3))

    b = np.dot(d1, d2)
    denom = a * e - b * b
    s = np.clip((b * f - c * e) / denom, 0.0, 1.0) if abs(denom) > 1e-12 else 0.0
    t = (b * s + f) / e
    if t < 0.0:
        t = 0.0
        s = np.clip(-c / a, 0.0, 1.0)
    elif t > 1.0:
        t = 1.0
        s = np.clip((b - c) / a, 0.0, 1.0)

    closest1 = p1 + s * d1
    closest2 = p3 + t * d2
    return float(np.linalg.norm(closest1 - closest2))


@dataclass
class Footprint:
    """A placed (or candidate) object's 2D collision shape."""
    shape: str            # "circle" or "capsule"
    center: np.ndarray
    radius: float
    p1: Optional[np.ndarray] = None   # capsule only
    p2: Optional[np.ndarray] = None   # capsule only


def _footprint_gap(a: Footprint, b: Footprint) -> float:
    """Surface-to-surface gap between two footprints (negative = overlapping)."""
    if a.shape == "circle" and b.shape == "circle":
        center_dist = float(np.linalg.norm(a.center - b.center))
    elif a.shape == "circle" and b.shape == "capsule":
        center_dist = _point_segment_dist(a.center, b.p1, b.p2)
    elif a.shape == "capsule" and b.shape == "circle":
        center_dist = _point_segment_dist(b.center, a.p1, a.p2)
    else:  # capsule-capsule
        center_dist = _segment_segment_dist(a.p1, a.p2, b.p1, b.p2)
    return center_dist - a.radius - b.radius


# --------------------------------------------------------------------------- #
class DomainRandomizer:
    def __init__(self, model: mujoco.MjModel, randomization_cfg_path: str):
        self.model = model
        with open(randomization_cfg_path) as f:
            self.cfg = yaml.safe_load(f)
        self.seeds = self.cfg["seeds"]
        self.ranges = self.cfg["ranges"]
        self.object_cfg = self.cfg["objects"]
        self.placement_cfg = self.cfg["placement"]
        self.object_names = list(self.object_cfg.keys())
        self.last_in_drawer_items: set = set()  # updated each reset(); which
        # cutlery started inside the closed drawer this episode -- useful
        # for /eval logging or success-criteria checks (e.g. "did the
        # episode actually need to open the drawer"), not just internal.
        self._compute_robot_equilibrium_pose()
        self._capture_nominal()

    # ------------------------------------------------------------------ #
    def _capture_nominal(self):
        m = self.model
        self.nominal = NominalState()

        for name in self.object_names:
            bid = m.body(name).id
            jnt_adr = m.body_jntadr[bid]
            assert m.jnt_type[jnt_adr] == mujoco.mjtJoint.mjJNT_FREE, (
                f"{name} must have a free joint for pose randomization"
            )
            qpos_adr = m.jnt_qposadr[jnt_adr]
            self.nominal.body_pos[name] = m.qpos0[qpos_adr:qpos_adr + 3].copy()
            self.nominal.body_quat[name] = m.qpos0[qpos_adr + 3:qpos_adr + 7].copy()

            # Each object's own nominal "resting orientation" reference --
            # NOT assumed to be world +Z. Cutlery is intentionally rotated
            # to lie flat (local Z points roughly horizontal at rest), so a
            # hardcoded "must point up" check would wrongly flag every
            # correctly-placed spoon as tipped over. Comparing against each
            # object's own nominal local-Z-in-world instead makes the
            # tip-detection check below correct for both object families.
            nominal_quat = self.nominal.body_quat[name]
            rotmat = np.zeros(9)
            mujoco.mju_quat2Mat(rotmat, nominal_quat)
            self.nominal.up_axis_world[name] = rotmat.reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
            self.nominal.body_mass[name] = float(m.body_mass[bid])

            for gid in range(m.ngeom):
                if m.geom_bodyid[gid] == bid:
                    self.nominal.geom_size[gid] = m.geom_size[gid].copy()
                    self.nominal.geom_friction[gid] = m.geom_friction[gid].copy()

        table_bid = m.body("table").id
        for gid in range(m.ngeom):
            if m.geom_bodyid[gid] == table_bid:
                self.nominal.geom_rgba[gid] = m.geom_rgba[gid].copy()

        for lid in range(m.nlight):
            self.nominal.light_pos[lid] = m.light_pos[lid].copy()

        for cid in range(m.ncam):
            self.nominal.cam_pos[cid] = m.cam_pos[cid].copy()
            self.nominal.cam_quat[cid] = m.cam_quat[cid].copy()

    # ------------------------------------------------------------------ #
    def reset(self, data: mujoco.MjData, seed: int):
        """Reset to a randomized scene for the given seed.

        Accepts ANY integer seed -- both the 10 reserved eval seeds
        (randomization.yaml `seeds`) and training seeds from
        `training.seed_range`. This method intentionally does not enforce
        which seed-space a caller uses; that discipline belongs to the
        caller (see is_eval_seed below), because a hard restriction here
        would make it impossible to use this same class for both training
        collection and eval without duplicating it.
        """
        rng = np.random.RandomState(seed)

        # Full scene draw + settle is retried (deterministically -- the SAME
        # rng stream just keeps advancing, never reseeded, so a given seed's
        # outcome is still fully reproducible) if settling carries an object
        # into the robot. Placement's own check only sees the object's
        # PLACED position; settling can drift it those last few cm into the
        # arm afterward (verified: found this happening in ~1/500 seeds).
        # Retrying the whole draw is what actually fixes it -- rejecting or
        # nudging just the offending object post-hoc risks a second bad
        # settle with everything else already at rest around it.
        max_scene_attempts = 3
        for attempt in range(max_scene_attempts):
            mujoco.mj_resetData(self.model, data)
            # Robot starts at its measured gravity equilibrium, NOT qpos0 --
            # must happen before placement so the collision checks below
            # validate against the pose the arms actually hold. See
            # _compute_robot_equilibrium_pose for the full rationale.
            self._apply_robot_equilibrium_pose(data)

            self._randomize_mass(rng)
            self._randomize_friction(rng)
            self._randomize_shape(rng)
            self._randomize_object_placement(data, rng)
            self._randomize_lighting(rng)
            self._randomize_camera(rng)
            self._randomize_background(rng)

            mujoco.mj_forward(self.model, data)
            self._settle(data)

            if not self._any_object_touches_robot(data) and not self._any_object_tipped_over(data):
                return

            reason = ("robot contact" if self._any_object_touches_robot(data)
                      else "an object tipped over")
            print(f"[randomization] seed {seed}: {reason} after "
                  f"settle (attempt {attempt + 1}/{max_scene_attempts}), "
                  f"retrying full scene draw")

        print(f"[randomization] WARNING: seed {seed} still has robot contact "
              f"after {max_scene_attempts} full attempts -- accepting anyway "
              f"(batch eval shouldn't die on one seed), but this episode will "
              f"start with an unresolved collision.")

    def _any_object_touches_robot(self, data) -> bool:
        return any(self._collides_with_environment(data, self.model.body(name).id)
                   for name in self.object_names)

    def _any_object_tipped_over(self, data, max_up_z_deviation=0.3) -> bool:
        """True if any object's orientation has tilted significantly away
        from ITS OWN nominal resting orientation.

        Compares only the Z-COMPONENT of the local-up-axis, not the full 3D
        vector -- this is deliberate, not a simplification. Yaw rotation
        (intentional, randomized) only ever changes the X/Y components of a
        rotated vector, never its Z-component, regardless of the object's
        nominal orientation. So this check is exactly invariant to
        intentional yaw for BOTH object families:
          - bottle/cup/bowl/plate: nominal up-axis is ~[0,0,1] (z=1). If one
            tips onto its side, its up-axis z-component collapses toward 0
            -- correctly flagged.
          - cutlery: nominal up-axis lies flat (z=~0). Yaw rotation keeps
            z=~0 no matter which direction it points in the horizontal
            plane -- never flagged for its own intended placement. If it
            somehow ended up knocked upright instead, z would jump toward
            1, correctly flagged.
        (An earlier version of this check compared full 3D dot product
        against the nominal vector, which broke for cutlery: two equally
        "flat" but differently-yawed spoons can have a full-vector dot
        product near 0 despite neither being tipped, since yaw changes
        their in-plane direction. Z-component-only sidesteps that entirely.)
        """
        for name in self.object_names:
            bid = self.model.body(name).id
            quat = data.xquat[bid]
            rotmat = np.zeros(9)
            mujoco.mju_quat2Mat(rotmat, quat)
            current_up_z = (rotmat.reshape(3, 3) @ np.array([0.0, 0.0, 1.0]))[2]
            nominal_up_z = self.nominal.up_axis_world[name][2]
            if abs(current_up_z - nominal_up_z) > max_up_z_deviation:
                return True
        return False

    def _settle(self, data):
        """Step physics until objects reach real equilibrium, so reset()
        hands back an already-stable scene rather than a teleported pose
        that merely LOOKS placed correctly. Placement computes each
        object's resting z from an isolated single-object measurement (see
        randomization.yaml comments); with several objects now interacting
        on the same table, small settle motion is normal and expected --
        this was previously invisible because nothing called mj_step after
        placement inside reset() at all, only mj_forward. A caller stepping
        physics themselves right after reset() was the one discovering that
        transient, which reads as "wobbling."

        Stops early once every object's velocity drops below a small
        threshold, so well-behaved seeds don't pay for the full budget.
        `settle_max_steps` in randomization.yaml `placement` controls the
        ceiling; picked from measurement (worst observed seed settled by
        ~450 steps), with margin.
        """
        max_steps = self.placement_cfg.get("settle_max_steps", 500)
        vel_threshold = self.placement_cfg.get("settle_velocity_threshold", 0.01)

        for _ in range(max_steps):
            mujoco.mj_step(self.model, data)
            max_speed = 0.0
            for name in self.object_names:
                bid = self.model.body(name).id
                dof = self.model.body_dofadr[bid]
                speed = np.linalg.norm(data.qvel[dof:dof + 3])
                max_speed = max(max_speed, speed)
            if max_speed < vel_threshold:
                break

        # Zero residual velocity so the episode's first observation shows a
        # scene at rest, not one still carrying the last bit of settle
        # motion (matters more for RGB motion blur / policy input framing
        # than for physics -- the position is already converged either way).
        for name in self.object_names:
            bid = self.model.body(name).id
            dof = self.model.body_dofadr[bid]
            data.qvel[dof:dof + 6] = 0.0
        mujoco.mj_forward(self.model, data)

    # ------------------------------------------------------------------ #
    # Robot gravity-equilibrium pose
    # ------------------------------------------------------------------ #
    def _robot_joint_qpos_addrs(self):
        """qpos addresses of the ROBOT's joints, identified model-agnostically
        as 'joints driven by an actuator'. No name prefixes, no hardcoded
        joint list -- swapping the placeholder arms for real SO-101 (or
        ALOHA) needs no change here. The drawer slide and the objects' free
        joints are correctly excluded: neither has an actuator."""
        addrs = []
        for act_id in range(self.model.nu):
            joint_id = self.model.actuator_trnid[act_id, 0]
            if joint_id >= 0:
                addrs.append(int(self.model.jnt_qposadr[joint_id]))
        return sorted(set(addrs))

    def _compute_robot_equilibrium_pose(self):
        """Measure (not guess) the pose the robot actually SETTLES INTO under
        gravity, and cache it as the per-episode starting pose.

        Why this exists -- this was a real, score-costing bug:
        the arms' MJCF rest pose (qpos0, all zeros) is NOT a gravity
        equilibrium. Position actuators produce torque kp*(ctrl - qpos) with
        no gravity compensation, so from qpos0 the arms visibly droop for
        the first second or so until spring torque balances weight -- the
        gripper tip drops ~7.5cm. Placement was validating object positions
        against the un-drooped qpos0 pose, and then the arms swept downward
        through exactly that space, knocking over tall objects (the bottle).
        No amount of placement checking could fix that, because the pose
        being checked was one the robot never actually holds.

        Fix: settle the robot ALONE once (objects parked far off-scene so
        they can't perturb it), record the equilibrium qpos, and start every
        episode there. Since ctrl stays at 0 and qpos already satisfies
        kp*(0 - qpos) = gravity torque, the arms begin at rest and stay at
        rest -- no droop, no sweep, and placement checks now validate
        against the pose the robot genuinely holds.

        Model-agnostic: nothing here knows the robot's kinematics, joint
        names, or link geometry. Whatever robot is loaded, its own
        equilibrium is measured from its own physics.
        """
        data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, data)

        # Park objects far away so they can't collide with (and thereby
        # distort) the robot's free-settling pose.
        for name in self.object_names:
            bid = self.model.body(name).id
            qadr = self.model.jnt_qposadr[self.model.body_jntadr[bid]]
            data.qpos[qadr:qadr + 3] = [100.0, 100.0, 100.0]

        mujoco.mj_forward(self.model, data)

        # DOF addresses of the robot's own joints, so residual motion is
        # measured on the ROBOT -- not on the parked objects, which are
        # falling freely at x=100 forever and would otherwise dominate the
        # reading with a meaningless huge velocity.
        robot_dofs = []
        for act_id in range(self.model.nu):
            jid = self.model.actuator_trnid[act_id, 0]
            if jid >= 0:
                robot_dofs.append(int(self.model.jnt_dofadr[jid]))
        robot_dofs = sorted(set(robot_dofs))

        # Quasi-static relaxation: step, then periodically zero the robot's
        # velocity. Plain stepping converges slowly here because the arms
        # oscillate around equilibrium (low damping relative to the spring),
        # leaving gravity-neutral joints -- e.g. the vertical-axis base yaw,
        # where gravity exerts no torque at all -- drifting far from where
        # they should rest. Killing velocity repeatedly bleeds off that
        # oscillation and converges to the true static pose.
        settle_steps = self.placement_cfg.get("robot_settle_steps", 4000)
        for i in range(settle_steps):
            mujoco.mj_step(self.model, data)
            if i % 50 == 0:
                for dof in robot_dofs:
                    data.qvel[dof] = 0.0

        for dof in robot_dofs:
            data.qvel[dof] = 0.0
        mujoco.mj_forward(self.model, data)

        self._robot_qpos_addrs = self._robot_joint_qpos_addrs()
        self.robot_equilibrium_qpos = {
            adr: float(data.qpos[adr]) for adr in self._robot_qpos_addrs
        }

        # Verify convergence by stepping once more, undisturbed, and seeing
        # how fast the robot is actually moving from this pose.
        for _ in range(20):
            mujoco.mj_step(self.model, data)
        residual = max(abs(float(data.qvel[d])) for d in robot_dofs) if robot_dofs else 0.0
        if residual > 0.02:
            print(f"[randomization] NOTE: robot still moving at {residual:.4f} rad/s from the "
                  f"computed equilibrium -- pose may be approximate; consider raising "
                  f"placement.robot_settle_steps (currently {settle_steps}).")

    def _apply_robot_equilibrium_pose(self, data):
        """Start the robot at its measured gravity equilibrium instead of
        qpos0, so it does not droop into the scene once physics runs."""
        for adr, value in self.robot_equilibrium_qpos.items():
            data.qpos[adr] = value

    def is_eval_seed(self, seed: int) -> bool:
        """True if `seed` is one of the 10 reserved grading seeds. Training
        code should assert NOT this before recording an episode."""
        return seed in self.seeds

    def sample_training_seed(self, rng: np.random.RandomState) -> int:
        """Pick a seed from `training.seed_range` -- disjoint from the eval
        seeds by construction (the ranges don't overlap), not by convention.
        `rng` here is a run-level generator for CHOOSING which seed to use
        next; it's separate from the RandomState reset() builds internally
        from that chosen seed."""
        lo, hi = self.cfg["training"]["seed_range"]
        return int(rng.randint(lo, hi))

    # ------------------------------------------------------------------ #
    def _placement_radius(self, name) -> float:
        """Radius to use for placement checks -- reads the LIVE geom_size
        for primitive-geom objects (bottle: cylinder, cutlery: capsule),
        which (now that _randomize_shape runs BEFORE placement, see reset())
        already reflects this seed's actual sampled shape_scale. Exact, not
        a worst-case guess.

        For mesh-based objects (bowl/plate/cup): writing geom_size on a mesh
        geom has NO effect on its actual collision geometry (verified
        directly -- a 5x geom_size write left geom_rbound completely
        unchanged, because mesh collision comes from mesh_vert, not
        geom_size). shape_scale is therefore currently a no-op for these
        three objects regardless of execution order; the static yaml
        radius_m is already exact and there's nothing live to read.
        """
        cfg = self.object_cfg[name]
        bid = self.model.body(name).id
        for gid in range(self.model.ngeom):
            if self.model.geom_bodyid[gid] == bid and self.model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
                return float(self.model.geom_size[gid][0])
        return cfg["radius_m"]

    def _placement_half_length(self, name) -> float:
        """Capsule half-length counterpart to _placement_radius -- same
        live-vs-static reasoning."""
        cfg = self.object_cfg[name]
        bid = self.model.body(name).id
        for gid in range(self.model.ngeom):
            if self.model.geom_bodyid[gid] == bid and self.model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
                return float(self.model.geom_size[gid][1])
        return cfg["half_length_m"]

    def _make_footprint(self, name, x, y, yaw) -> Footprint:
        cfg = self.object_cfg[name]
        center = np.array([x, y])
        radius = self._placement_radius(name)
        if cfg["shape"] == "circle":
            return Footprint(shape="circle", center=center, radius=radius)
        half_length = self._placement_half_length(name)
        p1, p2 = _capsule_endpoints(center, yaw, half_length)
        return Footprint(shape="capsule", center=center, radius=radius, p1=p1, p2=p2)

    def _violates_drawer_avoidance(self, name, footprint: Footprint) -> bool:
        cfg = self.object_cfg[name]
        if cfg["shape"] != "capsule":
            return False  # soft rule applies to cutlery only, per config docstring
        da = self.placement_cfg.get("drawer_avoidance")
        if not da:
            return False
        drawer_y_min = da["drawer_y_center_m"] - da["drawer_half_extent_y_m"] - da["extra_margin_m"]
        # Check the capsule's endpoints, not just its center -- a segment can
        # poke into the exclusion zone even if its center doesn't.
        return footprint.p1[1] > drawer_y_min or footprint.p2[1] > drawer_y_min

    def _collides_with_environment(self, data, candidate_body_id) -> bool:
        """True if the candidate object (already written into data, with
        mj_forward already called) touches anything it shouldn't.

        Model-agnostic by construction: this asks MuJoCo's real collision
        detector what the actually-loaded geometry touches, rather than
        guessing from a hardcoded distance, a "tall object" list, or a known
        robot home pose. Swapping the placeholder arms for real SO-101 (or
        ALOHA, or anything else) needs zero changes here -- whatever
        geometry is in the model at reset time is what gets checked against.

        "Touching" is only a violation if the other body is neither the
        table (expected -- objects rest on it) nor another manipulable
        object (handled separately by the 2D footprint checks; a
        not-yet-placed object is still sitting at an irrelevant stale qpos
        right now, so its contacts must be ignored here, not flagged).
        """
        table_bid = self.model.body("table").id
        for c in range(data.ncon):
            con = data.contact[c]
            b1 = int(self.model.geom_bodyid[con.geom1])
            b2 = int(self.model.geom_bodyid[con.geom2])
            if candidate_body_id not in (b1, b2):
                continue
            other_bid = b2 if b1 == candidate_body_id else b1
            if other_bid == table_bid:
                continue
            other_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, other_bid) or ""
            if other_name in self.object_names:
                continue
            # An item deliberately placed INSIDE the drawer is supposed to
            # rest on the drawer floor -- that contact is the feature
            # working, not a collision. Only flag drawer contact for items
            # that were meant to be on the table.
            if other_name == "drawer":
                candidate_name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, candidate_body_id) or ""
                if candidate_name in self.last_in_drawer_items:
                    continue
            return True  # touching the robot, floor, wall, etc.
        return False

    def _cutlery_names(self):
        return [n for n in self.object_names if self.object_cfg[n]["shape"] == "capsule"]

    def _decide_drawer_interior(self, rng):
        """Per seed, decide which cutlery items start inside the closed
        drawer instead of on the table. Returns a set of names. Drawn from
        the SAME seeded rng as everything else -- consumes one rng.uniform
        per cutlery piece (fixed draw count/order regardless of the cap
        below, so adding/lowering max_items_in_drawer later doesn't change
        which seeds pick which items, only how many of those picks get
        used), deterministic and reproducible like every other
        randomization axis here.

        Applies max_items_in_drawer as a hard cap -- see randomization.yaml
        drawer_interior comment for why (real geometric packing limit, not
        an arbitrary choice)."""
        di_cfg = self.placement_cfg.get("drawer_interior")
        if not di_cfg or not di_cfg.get("enabled", False):
            return set()
        fraction = di_cfg["fraction_in_drawer"]
        cap = di_cfg.get("max_items_in_drawer", len(self._cutlery_names()))
        selected = [name for name in self._cutlery_names() if rng.uniform() < fraction]
        return set(selected[:cap])

    def _place_in_drawer(self, data, rng, names, max_attempts):
        """Rejection-sample cutlery positions inside the closed drawer's
        cavity, checking only against other in-drawer items (the drawer is
        closed and isolated from the table/robot at spawn time, so no
        table-neighbor or robot-collision check applies here -- those only
        matter for what's reachable right now, and a closed drawer's
        contents aren't). Position is computed in the drawer's CLOSED
        world-frame position; since these items are ordinary free-jointed
        bodies (not parented to the drawer), when the drawer later slides
        open, normal contact friction carries them along with its floor --
        no special "attached to drawer" logic needed or wanted."""
        di_cfg = self.placement_cfg["drawer_interior"]
        drawer_y = di_cfg["world_y_center_closed_m"]
        x_lo, x_hi = di_cfg["cavity_x"]
        y_lo, y_hi = di_cfg["cavity_y"]
        yaw_lo, yaw_hi = di_cfg["cavity_yaw"]
        local_z = di_cfg["cavity_floor_local_z"]
        world_z = self.model.body("drawer").pos[2] + local_z
        min_gap = self.placement_cfg["min_gap_between_objects_m"]
        # Tighter space than the open table -- items are long (12cm) relative
        # to the cavity, so give rejection sampling more room to find a fit
        # before falling back.
        drawer_max_attempts = max(max_attempts, 500)

        placed_in_drawer: list[Footprint] = []
        for name in names:
            accepted = None
            for _ in range(drawer_max_attempts):
                x = rng.uniform(x_lo, x_hi)
                y = rng.uniform(y_lo, y_hi) + drawer_y
                yaw = rng.uniform(yaw_lo, yaw_hi)
                fp = self._make_footprint(name, x, y, yaw)
                if any(_footprint_gap(fp, other) < min_gap for other in placed_in_drawer):
                    continue
                accepted = (x, y, yaw, fp)
                break
            if accepted is None:
                # Deterministic 2D grid search over (x, y), not just y at a
                # fixed x=0. The first item can land at any x/yaw from the
                # random pass, so the remaining free space for a second item
                # isn't necessarily a clean band along y at x=0 -- it can be
                # a diagonal sliver. A y-only search at fixed x sometimes
                # missed exactly that sliver (found empirically: 5/1000
                # seeds failed with gaps of 0.019-0.0200m, just under the
                # 0.02m requirement -- a 2D search covers those cases a 1D
                # search structurally cannot). Checks each candidate against
                # where items ACTUALLY landed, not an assumed layout.
                yaw_mid = (yaw_lo + yaw_hi) / 2
                grid_accepted = None
                for x_candidate in np.linspace(x_lo, x_hi, 15):
                    for y_candidate in np.linspace(y_lo, y_hi, 15) + drawer_y:
                        fp = self._make_footprint(name, x_candidate, y_candidate, yaw_mid)
                        if any(_footprint_gap(fp, other) < min_gap for other in placed_in_drawer):
                            continue
                        grid_accepted = (x_candidate, y_candidate, yaw_mid, fp)
                        break
                    if grid_accepted is not None:
                        break
                if grid_accepted is not None:
                    print(f"[randomization] drawer interior: '{name}' used "
                          f"2D grid-search fallback after {drawer_max_attempts} random attempts")
                    accepted = grid_accepted
                else:
                    # Cavity is genuinely full (more items than comfortably
                    # fit) -- no valid slot exists at all even on a 225-point
                    # grid. Loud, not silent: this means fewer items should
                    # be sent to the drawer, not that this item's overlap
                    # should be hidden.
                    print(f"[randomization] WARNING: drawer interior has no "
                          f"valid slot left for '{name}' -- cavity may be "
                          f"over-subscribed for this seed's item count")
                    y = drawer_y
                    fp = self._make_footprint(name, 0.0, y, yaw_mid)
                    accepted = (0.0, y, yaw_mid, fp)
            x, y, yaw, fp = accepted
            placed_in_drawer.append(fp)
            self._apply_pose(data, name, x, y, yaw)
            self._set_object_z(data, name, world_z)

    def _set_object_z(self, data, name, world_z):
        bid = self.model.body(name).id
        jnt_adr = self.model.body_jntadr[bid]
        qpos_adr = self.model.jnt_qposadr[jnt_adr]
        data.qpos[qpos_adr + 2] = world_z

    def _randomize_object_placement(self, data, rng):
        max_attempts = self.placement_cfg["max_attempts"]
        min_gap = self.placement_cfg["min_gap_between_objects_m"]

        in_drawer = self._decide_drawer_interior(rng)
        self.last_in_drawer_items = in_drawer
        table_names = [n for n in self.object_names if n not in in_drawer]

        # Largest effective radius first, so big objects (hardest to route
        # around) claim space before small ones have to dodge them.
        def effective_radius(name):
            cfg = self.object_cfg[name]
            r = self._placement_radius(name)
            return r + (self._placement_half_length(name) if cfg["shape"] == "capsule" else 0.0)

        order = sorted(table_names, key=effective_radius, reverse=True)

        # Objects not yet processed this call still sit at whatever qpos
        # mj_resetData left them at (their raw MJCF nominal pose) -- push
        # them far off-scene first so they can never spuriously register as
        # "environment" in the mj_forward collision checks below before
        # their own turn comes up. (Belt-and-suspenders on top of the
        # object_names exclusion in _collides_with_environment.) Includes
        # in-drawer items too, since they're placed in a separate pass below.
        park_pos = np.array([100.0, 100.0, 100.0])
        for name in self.object_names:
            bid = self.model.body(name).id
            jnt_adr = self.model.body_jntadr[bid]
            qpos_adr = self.model.jnt_qposadr[jnt_adr]
            data.qpos[qpos_adr:qpos_adr + 3] = park_pos

        placed: list[Footprint] = []
        fallback_count = 0
        unresolved_robot_collision = []

        for name in order:
            cfg = self.object_cfg[name]
            x_lo, x_hi = cfg["x"]
            y_lo, y_hi = cfg["y"]
            yaw_lo, yaw_hi = cfg["yaw"]
            bid = self.model.body(name).id

            accepted = None
            for _ in range(max_attempts):
                x = rng.uniform(x_lo, x_hi)
                y = rng.uniform(y_lo, y_hi)
                yaw = rng.uniform(yaw_lo, yaw_hi)
                fp = self._make_footprint(name, x, y, yaw)

                if self._violates_drawer_avoidance(name, fp):
                    continue
                if any(_footprint_gap(fp, other) < min_gap for other in placed):
                    continue

                # Cheap 2D checks passed -- now the real 3D check against
                # whatever robot/environment geometry is actually loaded.
                self._apply_pose(data, name, x, y, yaw)
                mujoco.mj_forward(self.model, data)
                if self._collides_with_environment(data, bid):
                    continue

                accepted = (x, y, yaw, fp)
                break

            if accepted is None:
                # Fall back to nominal pose -- but still verify it, rather
                # than blindly trusting it. If the robot's own default pose
                # happens to overlap this object's nominal spawn point too,
                # silently accepting it would produce exactly the violent
                # spawn-collision this whole check exists to prevent.
                fallback_count += 1
                nominal_pos = self.nominal.body_pos[name]
                x, y, yaw = float(nominal_pos[0]), float(nominal_pos[1]), 0.0
                fp = self._make_footprint(name, x, y, yaw)
                self._apply_pose(data, name, x, y, yaw)
                mujoco.mj_forward(self.model, data)
                if self._collides_with_environment(data, bid):
                    unresolved_robot_collision.append(name)
                accepted = (x, y, yaw, fp)

            x, y, yaw, fp = accepted
            placed.append(fp)
            self._apply_pose(data, name, x, y, yaw)

        if in_drawer:
            self._place_in_drawer(data, rng, list(in_drawer), max_attempts)

        if fallback_count:
            # Not necessarily an error -- degrading gracefully is the
            # design -- but worth surfacing if it happens often, since it
            # means the ranges/margins are tight relative to object count.
            print(f"[randomization] seed used {fallback_count} nominal-pose "
                  f"fallback(s) after {max_attempts} placement attempts each")
        if unresolved_robot_collision:
            # This IS worth shouting about: a fallback that's STILL
            # colliding with the robot means this seed will start with a
            # violent spawn-collision no matter what. Don't crash (a batch
            # eval run over many seeds shouldn't die on one bad seed), but
            # make it impossible to miss.
            print(f"[randomization] WARNING: seed's fallback pose(s) still "
                  f"collide with robot/environment: {unresolved_robot_collision} "
                  f"-- this seed will start with an unresolved collision.")

    def _apply_pose(self, data, name, x, y, dyaw):
        bid = self.model.body(name).id
        jnt_adr = self.model.body_jntadr[bid]
        qpos_adr = self.model.jnt_qposadr[jnt_adr]

        nominal_pos = self.nominal.body_pos[name]
        nominal_quat = self.nominal.body_quat[name]

        new_pos = np.array([x, y, nominal_pos[2]])

        yaw_quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(yaw_quat, np.array([0.0, 0.0, 1.0]), dyaw)
        new_quat = np.zeros(4)
        mujoco.mju_mulQuat(new_quat, yaw_quat, nominal_quat)

        data.qpos[qpos_adr:qpos_adr + 3] = new_pos
        data.qpos[qpos_adr + 3:qpos_adr + 7] = new_quat
        dof_adr = self.model.jnt_dofadr[jnt_adr]
        data.qvel[dof_adr:dof_adr + 6] = 0.0

    # ------------------------------------------------------------------ #
    # Non-spatial randomization axes -- unchanged in behavior from before.
    # ------------------------------------------------------------------ #
    def _randomize_mass(self, rng):
        lo, hi = self.ranges["mass_scale"]
        for name in self.object_names:
            bid = self.model.body(name).id
            scale = rng.uniform(lo, hi)
            self.model.body_mass[bid] = self.nominal.body_mass[name] * scale

    def _randomize_friction(self, rng):
        lo, hi = self.ranges["friction_scale"]
        for name in self.object_names:
            bid = self.model.body(name).id
            scale = rng.uniform(lo, hi)
            for gid, nominal_fric in self.nominal.geom_friction.items():
                if self.model.geom_bodyid[gid] == bid:
                    self.model.geom_friction[gid] = nominal_fric * scale

    def _randomize_shape(self, rng):
        # KNOWN LIMITATION, verified empirically: writing geom_size on a
        # MESH geom (bowl, plate, cup) has zero effect on its actual
        # collision geometry -- mesh collision comes from mesh_vert, not
        # geom_size (confirmed: a 5x geom_size write left geom_rbound
        # completely unchanged). This loop still runs for those objects
        # (harmless no-op, not worth special-casing away) but shape_scale
        # currently only has real effect on primitive-geom objects: bottle
        # (cylinder) and the four cutlery pieces (capsule). Real mesh
        # scaling would require rescaling mesh_vert relative to a stored
        # nominal copy, per-instance (meshes are shared assets across
        # potential future duplicate objects) -- a real task, not a one-line
        # fix; flagging rather than silently pretending this covers meshes.
        lo, hi = self.ranges["shape_scale"]
        for name in self.object_names:
            bid = self.model.body(name).id
            scale = rng.uniform(lo, hi)
            for gid, nominal_size in self.nominal.geom_size.items():
                if self.model.geom_bodyid[gid] == bid:
                    self.model.geom_size[gid] = nominal_size * scale

    def _randomize_lighting(self, rng):
        amb_lo, amb_hi = self.ranges["lighting"]["ambient"]
        dif_lo, dif_hi = self.ranges["lighting"]["diffuse"]
        jit_lo, jit_hi = self.ranges["lighting"]["position_jitter_m"]

        for lid in range(self.model.nlight):
            amb = rng.uniform(amb_lo, amb_hi)
            dif = rng.uniform(dif_lo, dif_hi)
            jitter = rng.uniform(jit_lo, jit_hi, size=3)

            self.model.light_ambient[lid] = np.array([amb, amb, amb])
            self.model.light_diffuse[lid] = np.array([dif, dif, dif])
            self.model.light_pos[lid] = self.nominal.light_pos[lid] + jitter

    def _randomize_camera(self, rng):
        """Small pose jitter on every camera, relative to its nominal
        (sim.yaml-defined) pose -- never cumulative, same pattern as
        lighting/objects above. See randomization.yaml `ranges.camera` for
        the rationale (sim-to-real: don't let a vision backbone memorize one
        exact static viewpoint)."""
        cam_cfg = self.ranges.get("camera")
        if not cam_cfg:
            return
        pos_lo, pos_hi = cam_cfg["pos_jitter_m"]
        rot_lo, rot_hi = cam_cfg["rot_jitter_deg"]

        for cid in range(self.model.ncam):
            pos_jitter = rng.uniform(pos_lo, pos_hi, size=3)
            self.model.cam_pos[cid] = self.nominal.cam_pos[cid] + pos_jitter

            # Small rotation about a random axis -- not just a single Euler
            # axis -- so the jitter looks like plausible small camera
            # wobble/mounting tolerance, not a suspiciously clean yaw-only
            # perturbation.
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            angle_deg = rng.uniform(rot_lo, rot_hi)
            jitter_quat = np.zeros(4)
            mujoco.mju_axisAngle2Quat(jitter_quat, axis, np.deg2rad(angle_deg))
            new_quat = np.zeros(4)
            mujoco.mju_mulQuat(new_quat, jitter_quat, self.nominal.cam_quat[cid])
            self.model.cam_quat[cid] = new_quat

    def _randomize_background(self, rng):
        # PLACEHOLDER: no real background texture set exists on disk yet.
        # Approximated as a table color tint until real textures land.
        if not self.ranges["background"]["randomize_table_material"]:
            return
        for gid, nominal_rgba in self.nominal.geom_rgba.items():
            tint = rng.uniform(0.7, 1.0, size=3)
            new_rgba = nominal_rgba.copy()
            new_rgba[:3] = nominal_rgba[:3] * tint
            self.model.geom_rgba[gid] = new_rgba
