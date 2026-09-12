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
    """World-XY endpoints at (x, y, yaw). Flat orientation maps the mesh's
    long axis to world +X at yaw=0 (see compute_flat_quat), so direction
    rotates from there."""
    direction = np.array([math.cos(yaw), math.sin(yaw)])
    center = np.asarray(center_xy)
    return center - half_length * direction, center + half_length * direction


def _compute_flat_quat(verts: np.ndarray) -> np.ndarray:
    """PCA on mesh vertices: longest axis -> world X, thinnest -> world Z.
    Lays an elongated mesh flat regardless of the mesh's own local frame."""
    c = verts - verts.mean(axis=0)
    _, evecs = np.linalg.eigh(c.T @ c)  # ascending eigenvalues
    thin, mid, long = evecs[:, 0], evecs[:, 1], evecs[:, 2]
    R = np.column_stack([long, mid, thin]).T
    if np.linalg.det(R) < 0:
        R[2, :] *= -1
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q


def _flip_about_long_axis(q: np.ndarray) -> np.ndarray:
    """180deg about the long axis, applied AFTER q establishes world-X
    alignment (flip is the outer/second-applied rotation) -- not before,
    which would flip about the mesh's raw unaligned local axis instead."""
    flip = np.array([0.0, 1.0, 0.0, 0.0])
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, flip, q)
    return out


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
        self._compute_flat_orientations()

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
        """Radius to use for placement checks. Priority:
        1. Measured from the actual mesh (self._mesh_radius, computed once
           via PCA in _compute_flat_orientations) -- for cutlery. This is
           the source of truth; a hand-measured/yaml value can silently
           drift out of sync with the real asset (verified: the fork's
           real PCA length is 0.198m, nearly double a stale 0.105m yaml
           value, which caused real wall penetration in the drawer).
        2. LIVE geom_size for primitive-geom objects (bottle: cylinder),
           which reflects this seed's actual shape_scale.
        3. Static yaml radius_m -- for mesh objects with no live geom_size
           (bowl/plate/cup; shape_scale is a no-op for these, see below).
        """
        if hasattr(self, "_mesh_radius") and name in self._mesh_radius:
            return self._mesh_radius[name]
        cfg = self.object_cfg[name]
        bid = self.model.body(name).id
        for gid in range(self.model.ngeom):
            if self.model.geom_bodyid[gid] == bid and self.model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
                return float(self.model.geom_size[gid][0])
        return cfg["radius_m"]

    def _placement_half_length(self, name) -> float:
        """Capsule half-length counterpart to _placement_radius -- same
        measured-mesh-first priority."""
        if hasattr(self, "_mesh_half_length") and name in self._mesh_half_length:
            return self._mesh_half_length[name]
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

    def _robot_base_positions(self):
        """XY positions of every robot base in the model, found by name
        pattern ('*_base' bodies that are actual robot mounts, identified
        via the left/right arm prefixes already used elsewhere) -- not
        hardcoded coordinates, so this stays correct through any future
        base repositioning without a code change."""
        positions = []
        for bid in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if name.endswith("_base") and (name.startswith("left_") or name.startswith("right_")):
                positions.append(self.model.body_pos[bid][:2].copy())
        return positions

    def _violates_base_keepout(self, footprint: Footprint) -> bool:
        """True if this footprint's circle (or capsule) comes within the
        robot base's real physical footprint of any arm base.

        This is NOT the same thing check as _collides_with_environment --
        that only catches literal mesh overlap after the fact (mj_forward +
        contact check); this is a proactive, cheap 2D distance check used
        during the SAME rejection-sampling loop as the object-object gap
        check, so a bad candidate is rejected before ever calling
        mj_forward. Found necessary the hard way: a plate placed 7cm from a
        base center, with its own 8.9cm radius, visually oversat the base
        (verified: plate's own edge extended 1.9cm past the base's center
        point) without the existing collision check ever flagging it --
        real mesh geometry doesn't fill its own bounding sphere, so
        "no literal mesh contact" and "doesn't visually sit on the base"
        are different claims. This checks the second one directly.

        Base footprint radius (0.1075m) measured from the actual base
        geom's bounding sphere, not guessed -- see conversation record.
        """
        BASE_FOOTPRINT_RADIUS = 0.1075
        margin = self.placement_cfg["min_gap_between_objects_m"]
        for base_xy in self._robot_base_positions():
            if footprint.shape == "circle":
                dist = float(np.linalg.norm(footprint.center - base_xy))
            else:
                dist = _point_segment_dist(base_xy, footprint.p1, footprint.p2)
            if dist - footprint.radius - BASE_FOOTPRINT_RADIUS < margin:
                return True
        return False

    def _violates_drawer_avoidance(self, name, footprint: Footprint) -> bool:
        """Applies to EVERY object shape, not just cutlery. This rule used
        to be capsule-only because the drawer sat recessed under the
        tabletop -- no table-surface object, of any shape, could physically
        be in its sweep path. That's no longer true: the drawer now sits ON
        the table surface (verified the hard way -- 26/50 randomized seeds
        had the drawer blocked from opening, and checking found bowl/plate/
        cup/bottle sitting in the sweep path, since only capsules were ever
        checked here)."""
        da = self.placement_cfg.get("drawer_avoidance")
        if not da:
            return False
        drawer_y_min = da["drawer_y_center_m"] - da["drawer_half_extent_y_m"] - da["extra_margin_m"]
        if footprint.shape == "capsule":
            # Check the capsule's endpoints, not just its center -- a
            # segment can poke into the exclusion zone even if its center
            # doesn't.
            return footprint.p1[1] > drawer_y_min or footprint.p2[1] > drawer_y_min
        return footprint.center[1] + footprint.radius > drawer_y_min

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

    def _mesh_verts_for_body(self, name, collision_only=True):
        bid = self.model.body(name).id
        verts = []
        for gid in range(self.model.ngeom):
            if self.model.geom_bodyid[gid] != bid or self.model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            if collision_only and self.model.geom_contype[gid] == 0:
                continue  # skip visual-only geoms -- they never generate contact,
                          # so including them overstates the collision-relevant extent
            mid = self.model.geom_dataid[gid]
            v = self.model.mesh_vert[self.model.mesh_vertadr[mid]:
                                      self.model.mesh_vertadr[mid] + self.model.mesh_vertnum[mid]]
            gpos = self.model.geom_pos[gid]
            gquat = self.model.geom_quat[gid]
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, gquat)
            v_in_body = (rot.reshape(3, 3) @ v.T).T + gpos
            verts.append(v_in_body)
        return np.concatenate(verts, axis=0)

    def _settle_upright_z(self, name, quat, steps=1500):
        """Drop `name` alone at a safe spot with the given base quat, return
        |z-component of its long axis| after settling. Lower = flatter."""
        data = mujoco.MjData(self.model)
        bid = self.model.body(name).id
        jnt_adr = self.model.body_jntadr[bid]
        qpos_adr = self.model.jnt_qposadr[jnt_adr]
        for other in self.object_names:
            if other == name:
                continue
            oadr = self.model.jnt_qposadr[self.model.body_jntadr[self.model.body(other).id]]
            data.qpos[oadr:oadr + 3] = [100.0, 100.0, 100.0]
        data.qpos[qpos_adr:qpos_adr + 3] = [0.0, -0.28, 0.85]
        data.qpos[qpos_adr + 3:qpos_adr + 7] = quat
        mujoco.mj_forward(self.model, data)
        for _ in range(steps):
            mujoco.mj_step(self.model, data)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, data.qpos[qpos_adr + 3:qpos_adr + 7])
        long_axis_world = rot.reshape(3, 3) @ np.array([1.0, 0.0, 0.0])
        return abs(long_axis_world[2])

    def _compute_flat_orientations(self):
        self._flat_quat = {}
        self._mesh_half_length = {}
        self._mesh_radius = {}
        self._long_eigvec = {}
        for name in self._cutlery_names():
            verts = self._mesh_verts_for_body(name)
            c = verts - verts.mean(axis=0)
            _, evecs = np.linalg.eigh(c.T @ c)
            thin, mid, long = evecs[:, 0], evecs[:, 1], evecs[:, 2]
            long_proj = c @ long
            thin_proj = c @ thin
            self._mesh_half_length[name] = float((long_proj.max() - long_proj.min()) / 2)
            self._mesh_radius[name] = float((thin_proj.max() - thin_proj.min()) / 2)
            self._long_eigvec[name] = long

            base = _compute_flat_quat(verts)
            flipped = _flip_about_long_axis(base)
            z_base = self._settle_upright_z(name, base)
            z_flip = self._settle_upright_z(name, flipped)
            chosen = base if z_base <= z_flip else flipped
            self._flat_quat[name] = chosen
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, chosen)
            self.nominal.up_axis_world[name] = rot.reshape(3, 3) @ np.array([0.0, 0.0, 1.0])

    def true_capsule_endpoints(self, data, name):
        """Authoritative world-space endpoints of a cutlery item, computed
        from its ACTUAL current qpos quaternion and the real PCA long axis
        -- not by extracting an abstract 'yaw' and recomposing (that broke:
        the object's true reference orientation is the PCA flat quat, not
        identity, so a generic quaternion-to-yaw formula doesn't recover
        anything meaningful). Single source of truth, used internally and
        by tests, so the two can never silently disagree again."""
        bid = self.model.body(name).id
        qadr = self.model.jnt_qposadr[self.model.body_jntadr[bid]]
        pos = data.qpos[qadr:qadr + 3]
        quat = data.qpos[qadr + 3:qadr + 7]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        world_long = rot.reshape(3, 3) @ self._long_eigvec[name]
        half_length = self._mesh_half_length[name]
        p1 = np.array(pos[:2]) - half_length * world_long[:2]
        p2 = np.array(pos[:2]) + half_length * world_long[:2]
        return p1, p2

    def _decide_drawer_interior(self, rng):
        """Per seed, decide which cutlery items start inside the closed
        drawer instead of on the table. Returns a set of names.

        Draws exactly one rng.uniform per cutlery piece (fixed order), so
        changing min_items_in_drawer later does NOT shift the rng stream
        for other randomization axes (lighting, camera, background) on
        already-used seeds -- only the drawer contents for that seed can
        change, and only upward (never empties a previously non-empty draw).

        After the random draw, enforces:
          - min_items_in_drawer: if fewer than this many were picked by
            chance, force-fill from the LOWEST uniform draws (they were
            closest to being picked anyway, so it still reads as seeded).
          - max_items_in_drawer: hard cap (real geometric packing limit).

        Why min_items_in_drawer exists: with fraction_in_drawer=0.4 and 4
        cutlery items, the probability all four land on the table is
        0.6^4 = 13%. Those seeds are unusable for the retrieve task (the
        drawer opens to an empty cavity, no retrieve waypoints exist, and
        the whole bimanual phase is skipped). Seed 10001 was one of them.
        Setting min_items_in_drawer=1 eliminates the dead-seed class
        entirely, at the cost of biasing the distribution slightly toward
        one-in-drawer -- which is fine, because the task is not "measure
        the frequency of empty drawers", it is "always exercise retrieve".
        """
        di_cfg = self.placement_cfg.get("drawer_interior")
        if not di_cfg or not di_cfg.get("enabled", False):
            return set()

        fraction = di_cfg["fraction_in_drawer"]
        min_items = di_cfg.get("min_items_in_drawer", 0)
        cap = di_cfg.get("max_items_in_drawer", len(self._cutlery_names()))

        names = self._cutlery_names()
        scores = [(name, rng.uniform()) for name in names]
        selected = [name for name, u in scores if u < fraction]

        # Force-fill if the random draw came up short.
        if len(selected) < min_items and names:
            remaining = sorted(
                ((name, u) for name, u in scores if name not in selected),
                key=lambda t: t[1],
            )
            for name, _ in remaining[: min_items - len(selected)]:
                selected.append(name)

        return set(selected[:cap])

    def _place_in_drawer(self, data, rng, names, max_attempts):
        di_cfg = self.placement_cfg["drawer_interior"]
        drawer_y = di_cfg["world_y_center_closed_m"]
        yaw_lo, yaw_hi = di_cfg["cavity_yaw"]
        local_z = di_cfg["cavity_floor_local_z"]
        world_z = self.model.body("drawer").pos[2] + local_z
        min_gap = self.placement_cfg["min_gap_between_objects_m"]
        drawer_max_attempts = max(max_attempts, 500)

        # Sampling range comes from config (cavity_x/cavity_y), NOT hardcoded
        # constants -- this was a real, previously-silent bug: the hardcoded
        # values here ignored cavity_x/cavity_y entirely, so tightening
        # those in randomization.yaml (e.g. to keep cutlery within IK reach)
        # had zero effect. ACCEPTANCE is still real collision detection
        # against the drawer walls (physical fit), independent of this --
        # this range only controls where sampling is ATTEMPTED.
        x_lo, x_hi = di_cfg["cavity_x"]
        y_lo, y_hi = di_cfg["cavity_y"]

        wall_names = ["drawer_wall_left", "drawer_wall_right",
                      "drawer_wall_back", "drawer_wall_front"]
        wall_geoms = {self.model.geom(n).id for n in wall_names}

        def hits_wall(body_id):
            for c in range(data.ncon):
                con = data.contact[c]
                g1, g2 = con.geom1, con.geom2
                b1, b2 = self.model.geom_bodyid[g1], self.model.geom_bodyid[g2]
                if body_id in (b1, b2) and (g1 in wall_geoms or g2 in wall_geoms):
                    return True
            return False

        # Split the drawer's x-range into N equal slots -- one per item --
        # so two cutlery pieces are laid out SIDE-BY-SIDE along x, never
        # both trying to fit into the same half. Each item still randomizes
        # position/yaw WITHIN its slot; only the slot assignment itself is
        # structured. Without this, two ~20cm capsules could not be placed
        # in an 8.6cm-wide drawer cavity without overlap, and disabling the
        # overlap check just let one of them get pushed outside the drawer
        # entirely (measured: z=0.007 -> floor, not drawer floor 0.78).
        #
        # The slot order is shuffled by the same rng that drives every other
        # randomized axis, so "which item is on the left" varies from seed
        # to seed. Deterministic per seed (same rng stream), not per run.
        names = list(names)
        rng.shuffle(names)
        n_slots = max(len(names), 1)
        slot_edges = np.linspace(x_lo, x_hi, n_slots + 1)

        for slot_idx, name in enumerate(names):
            slot_lo = float(slot_edges[slot_idx])
            slot_hi = float(slot_edges[slot_idx + 1])
            bid = self.model.body(name).id
            accepted = None
            for _ in range(drawer_max_attempts):
                x = rng.uniform(slot_lo, slot_hi)
                y = rng.uniform(y_lo, y_hi) + drawer_y
                yaw = rng.uniform(yaw_lo, yaw_hi)
                fp = self._make_footprint(name, x, y, yaw)
                # Object-object overlap inside the drawer is allowed: items
                # are in disjoint x-slots, and within a slot overlap does not
                # hurt (they are not being retrieved). Only wall penetration
                # is a real failure mode.
                self._apply_pose(data, name, x, y, yaw)
                self._set_object_z(data, name, world_z)
                mujoco.mj_forward(self.model, data)
                if hits_wall(bid):
                    continue
                accepted = (x, y, yaw, fp)
                break
            if accepted is None:
                yaw_mid = (yaw_lo + yaw_hi) / 2
                grid_accepted = None
                for x_candidate in np.linspace(slot_lo, slot_hi, 15):
                    for y_candidate in np.linspace(y_lo, y_hi, 15) + drawer_y:
                        fp = self._make_footprint(name, x_candidate, y_candidate, yaw_mid)
                        self._apply_pose(data, name, x_candidate, y_candidate, yaw_mid)
                        self._set_object_z(data, name, world_z)
                        mujoco.mj_forward(self.model, data)
                        if hits_wall(bid):
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
                    print(f"[randomization] WARNING: drawer interior has no "
                          f"valid slot left for '{name}' -- cavity may be "
                          f"over-subscribed for this seed's item count")
                    y = drawer_y
                    fp = self._make_footprint(name, 0.0, y, yaw_mid)
                    self._apply_pose(data, name, 0.0, y, yaw_mid)
                    self._set_object_z(data, name, world_z)
                    accepted = (0.0, y, yaw_mid, fp)
            x, y, yaw, fp = accepted
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

        def range_area(name):
            cfg = self.object_cfg[name]
            x_lo, x_hi = cfg["x"]
            y_lo, y_hi = cfg["y"]
            return (x_hi - x_lo) * (y_hi - y_lo)

        def placement_difficulty(name):
            """Combines both ways an object can be 'hard to place first':
            being physically large (original heuristic), or having an
            unusually small allowed range relative to its own size (a
            frozen/pinned object's tiny zone gets crowded by other objects'
            much wider ranges before its own turn). Pure size-first
            de-prioritizes small-range objects and lets them get crowded
            out (verified: 1/100 real failure, bottle's tiny zone invaded).
            Pure range-first over-corrects and de-prioritizes the biggest
            objects instead, since their ranges are naturally the largest.

            Also weights in shape rigidity: a circle's footprint is
            identical at every yaw, so it has no orientation to rotate into
            whatever gap is left -- a capsule of similar size can angle
            itself to fit a gap a circle of the same effective radius
            cannot. Verified this matters: without the rigidity weight,
            round objects (plate, both circles) were placed after all
            four cutlery (capsules, more forgiving) and lost the placement
            race in 3/150 seeds despite having reasonable size/range
            scores individually."""
            cfg = self.object_cfg[name]
            r = effective_radius(name)
            own_area = np.pi * r * r
            rigidity = 3.0 if cfg["shape"] == "circle" else 1.0
            base_difficulty = rigidity * own_area / max(range_area(name), 1e-9)
            # An object whose OWN range is a statistical outlier -- much
            # smaller than every other object's range (e.g. a frozen/pinned
            # object) -- needs to go essentially first regardless of shape
            # or size, since nothing else can be trusted not to randomly
            # land in its narrow zone first. Verified: even after the shape
            # and size weighting above, the frozen bottle (range_area=0.056
            # vs 0.25-0.44 for everything else -- a clear outlier) still
            # lost its own space in 2/300 seeds to plate placed
            # earlier under the general formula.
            all_range_areas = [range_area(n) for n in table_names]
            median_range = float(np.median(all_range_areas))
            if range_area(name) < 0.5 * median_range:
                return base_difficulty + 1000.0
            return base_difficulty

        order = sorted(table_names, key=placement_difficulty, reverse=True)

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
                if self._violates_base_keepout(fp):
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
                # Deterministic grid search over this object's own valid
                # range, same proven pattern as the drawer-interior fallback
                # (see _place_in_drawer) -- a single nominal-pose fallback
                # can't satisfy the base keep-out constraint on a crowded
                # table (verified: 29% overlap rate before this fix, since
                # one fixed fallback point per object has no way to route
                # around whatever's already placed). A search over many
                # candidates can.
                yaw_mid = (yaw_lo + yaw_hi) / 2
                grid_accepted = None
                grid_safety_buffer = 0.004  # avoids landing exactly at the
                # min_gap threshold, where grid discretization can produce a
                # gap a millimeter or two under the requirement even though
                # a nearby, unsampled point would clear it comfortably.
                for x_c in np.linspace(x_lo, x_hi, 20):
                    for y_c in np.linspace(y_lo, y_hi, 20):
                        fp = self._make_footprint(name, x_c, y_c, yaw_mid)
                        if self._violates_drawer_avoidance(name, fp):
                            continue
                        if self._violates_base_keepout(fp):
                            continue
                        if any(_footprint_gap(fp, other) < min_gap + grid_safety_buffer for other in placed):
                            continue
                        self._apply_pose(data, name, x_c, y_c, yaw_mid)
                        mujoco.mj_forward(self.model, data)
                        if self._collides_with_environment(data, bid):
                            continue
                        grid_accepted = (x_c, y_c, yaw_mid, fp)
                        break
                    if grid_accepted is not None:
                        break

                if grid_accepted is not None:
                    fallback_count += 1
                    accepted = grid_accepted
                else:
                    # Even the grid search found nothing -- fall back to
                    # nominal pose as a last resort, still verified rather
                    # than blindly trusted.
                    fallback_count += 1
                    nominal_pos = self.nominal.body_pos[name]
                    x, y, yaw = float(nominal_pos[0]), float(nominal_pos[1]), 0.0
                    fp = self._make_footprint(name, x, y, yaw)
                    self._apply_pose(data, name, x, y, yaw)
                    mujoco.mj_forward(self.model, data)
                    if self._collides_with_environment(data, bid) or self._violates_base_keepout(fp):
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
        base_quat = self._flat_quat[name] if name in self._flat_quat else self.nominal.body_quat[name]

        new_pos = np.array([x, y, nominal_pos[2]])

        yaw_quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(yaw_quat, np.array([0.0, 0.0, 1.0]), dyaw)
        new_quat = np.zeros(4)
        mujoco.mju_mulQuat(new_quat, yaw_quat, base_quat)

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
        # MESH geom ( plate, cup) has zero effect on its actual
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