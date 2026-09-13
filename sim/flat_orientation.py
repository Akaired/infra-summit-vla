"""
Generic "lay an elongated mesh flat on the table" orientation helper.

Why this exists
---------------
Swapping primitive capsule cutlery for YCB spoon/fork meshes broke the old
`euler="-1.5708 0 0"` trick: that assumed the object's long axis was local Z,
which is NOT guaranteed across meshes. Worse, even with the long axis
correctly horizontalized, these thin meshes would settle back toward VERTICAL
during physics, because the placeholder isotropic inertia
(`diaginertia="0.0001 0.0001 0.0001"`) told the solver a thin fork resists
tipping equally in every direction -- so balancing on its end was a valid
equilibrium.

Verified empirically (real YCB spoon + fork, CoACD collision meshes):
  - Dropping from 5 different start orientations -> always settled near
    VERTICAL (long-axis |Z-component| ~0.9-0.99). So "settle and record"
    (a naive Option A) bakes in the WRONG pose.
  - Replacing the placeholder inertia with geometry-inferred inertia (via
    `density` on the collision geoms, no explicit <inertial>) let the spoon
    settle genuinely flat (|Zcomp| 0.28 -> 0.034).
  - PCA orientation (longest axis -> world X, thinnest -> world Z), auto-
    flipped 180deg about the long axis to whichever rests flatter, brought
    the fork to |Zcomp| 0.23 (~13deg, a physically honest lean for a tined
    fork), and kept the spoon at 0.034.

So the robust recipe is: (1) infer inertia from geometry, (2) spawn at a
PCA-computed flat quaternion, (3) still apply per-seed yaw on top for
randomization.

This module is the orientation half (steps 2-3). The inertia half is an XML
change: give each cutlery collision geom a `density="700"` and REMOVE the
`<inertial ...>` line so MuJoCo computes the real tensor.
"""
import numpy as np
import mujoco


def compute_flat_quat_from_vertices(verts: np.ndarray, model, coll_meshes=None) -> np.ndarray:
    """Return a wxyz quaternion that lays an elongated mesh flat.

    PCA on the vertices:
      - largest principal axis  -> world X (long dimension, in table plane)
      - smallest principal axis -> world Z (thinnest dimension, vertical)
    Independent of the mesh's internal coordinate convention.

    `verts`: (N,3) vertex array in the mesh's own frame.
    Returns the base flat quaternion (no yaw yet -- caller composes yaw).
    """
    centered = verts - verts.mean(axis=0)
    cov = centered.T @ centered
    _, evecs = np.linalg.eigh(cov)              # ascending eigenvalues
    thin, mid, long = evecs[:, 0], evecs[:, 1], evecs[:, 2]

    # Rotation taking the mesh frame -> world so long->X, mid->Y, thin->Z.
    R = np.column_stack([long, mid, thin]).T
    if np.linalg.det(R) < 0:                     # ensure proper rotation
        R[2, :] *= -1

    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q


def flip_about_long_axis(q_flat: np.ndarray) -> np.ndarray:
    """Return the quaternion rotated 180deg about its own long (local-X) axis
    -- the 'other way up' resting pose. For asymmetric cutlery (a fork's
    tines, a spoon's scoop) one of the two rests flatter; the caller settles
    both once at init and keeps the flatter."""
    flip = np.array([1.0, 0.0, 0.0, 0.0])       # identity placeholder
    # 180deg about X in wxyz is (0,1,0,0):
    flip = np.array([0.0, 1.0, 0.0, 0.0])
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, q_flat, flip)
    return out


def compose_yaw(q_flat: np.ndarray, yaw_rad: float) -> np.ndarray:
    """Apply a world-Z yaw on top of the flat orientation -- this is the
    per-seed randomization, applied AFTER laying flat so the object still
    faces a random direction in the table plane."""
    yaw_q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(yaw_q, np.array([0.0, 0.0, 1.0]), yaw_rad)
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, yaw_q, q_flat)
    return out


# --------------------------------------------------------------------------
# Integration sketch for randomization.py (compute once per distinct mesh at
# DomainRandomizer.__init__, cache the flatter-settling variant per object):
#
#   def _compute_flat_orientations(self):
#       self._flat_quat = {}
#       for name in self._cutlery_names():
#           verts = self._mesh_vertices_for_body(name)   # from model.mesh_vert
#           base = compute_flat_quat_from_vertices(verts, self.model)
#           flipped = flip_about_long_axis(base)
#           # settle both once on a scratch plane, keep whichever ends flatter
#           self._flat_quat[name] = self._pick_flatter(name, base, flipped)
#
# Then in _apply_pose, for cutlery, replace the nominal-quat composition with:
#       new_quat = compose_yaw(self._flat_quat[name], dyaw)
#
# And in the MJCF, for each cutlery body:
#   - REMOVE:  <inertial pos="0 0 0" mass="0.03" diaginertia="0.0001 0.0001 0.0001"/>
#   - ADD density to each collision geom: <geom ... density="700"/>
#   so MuJoCo infers the real (anisotropic) inertia tensor from the mesh.
# --------------------------------------------------------------------------
