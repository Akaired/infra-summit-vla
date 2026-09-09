"""Per-subtask completion detection, driven by configs/eval.yaml:detection.

Field practice (Bifrost, "How to evaluate a VLA policy"): always report the
per-subtask breakdown, not just the aggregate pass/fail -- the aggregate hides
*where* the pipeline breaks. :class:`SubtaskTracker` therefore keeps a status
for every subtask in ``configs/eval.yaml:subtasks`` and the step index at which
each first completed.

Each status is one of:

    "completed"        -- physics confirmed it, this step or earlier (latching)
    "incomplete"       -- implemented, not yet satisfied
    "not_implemented"  -- no physical state exists to check it (never a success)

A subtask never counts as done on the policy's say-so; only the MJCF state,
read back through :class:`eval.scene.EpisodeScene`, can flip it to "completed".
"""
from __future__ import annotations

import math
from typing import Any

STATUS_COMPLETED = "completed"
STATUS_INCOMPLETE = "incomplete"
STATUS_NOT_IMPLEMENTED = "not_implemented"


class SubtaskTracker:
    def __init__(
        self,
        scene: Any,
        eval_cfg: dict[str, Any],
        sim_cfg: dict[str, Any],
    ) -> None:
        self._scene = scene
        self._order: list[str] = list(eval_cfg["subtasks"])
        self._det: dict[str, Any] = eval_cfg["detection"]
        self._objects: dict[str, Any] = sim_cfg["objects"]
        self._left_prefix = sim_cfg["robot"]["left_arm_prefix"]
        self._right_prefix = sim_cfg["robot"]["right_arm_prefix"]

        self.status: dict[str, str] = {name: STATUS_INCOMPLETE for name in self._order}
        self.first_completion_step: dict[str, int] = {}

        # pour has no simulated liquid to check -- declare that up front so it
        # can never be reported as anything but not_implemented.
        if "pour_completed" in self.status and not self._det.get("pour_enabled", False):
            self.status["pour_completed"] = STATUS_NOT_IMPLEMENTED

        self._spawn_xyz: dict[str, tuple[float, float, float]] = {}
        self._handoff_arms_seen: set[str] = set()

    # ------------------------------------------------------------------ #
    def bind_spawn_reference(self) -> None:
        """Snapshot the post-settle pose of the placed objects.

        Call once after ``scene.reset(seed)`` and before the first step: the
        *_placed detectors measure displacement from here, so it must be the
        scene the episode actually starts from.
        """
        for role in ("plate", "cup"):
            body = self._objects[role]
            pos = self._scene.body_xpos(body)
            self._spawn_xyz[role] = (float(pos[0]), float(pos[1]), float(pos[2]))
        # Per-cutlery start height. Body origins sit at a mesh-dependent offset
        # from the contact point, so "retrieved" is judged as a rise relative
        # to where *this* piece started (table or drawer floor), not an
        # absolute Z -- the offset cancels out.
        self._cutlery_spawn_z: dict[str, float] = {
            name: float(self._scene.body_xpos(name)[2])
            for name in self._objects["cutlery"]
        }

    def update(self, step_index: int) -> None:
        for name in self._order:
            if self.status[name] in (STATUS_COMPLETED, STATUS_NOT_IMPLEMENTED):
                continue
            if self._detect(name):
                self.status[name] = STATUS_COMPLETED
                self.first_completion_step[name] = step_index

    def all_terminal(self) -> bool:
        """True once no status can change again -- lets an episode stop early."""
        return all(
            s in (STATUS_COMPLETED, STATUS_NOT_IMPLEMENTED) for s in self.status.values()
        )

    def report(self, require_all_subtasks: bool) -> dict[str, Any]:
        completed = sum(1 for s in self.status.values() if s == STATUS_COMPLETED)
        implemented = [n for n, s in self.status.items() if s != STATUS_NOT_IMPLEMENTED]
        not_implemented = [n for n, s in self.status.items() if s == STATUS_NOT_IMPLEMENTED]

        if require_all_subtasks:
            success = all(s == STATUS_COMPLETED for s in self.status.values())
        else:
            success = all(self.status[n] == STATUS_COMPLETED for n in implemented)

        return {
            "subtasks": dict(self.status),
            "subtasks_completed": completed,
            "subtasks_total": len(self._order),
            "subtasks_implemented": len(implemented),
            "not_implemented_subtasks": not_implemented,
            "first_completion_step": dict(self.first_completion_step),
            "success": success,
        }

    # ------------------------------------------------------------------ #
    def _detect(self, name: str) -> bool:
        detector = getattr(self, f"_detect_{name}", None)
        if detector is None:
            raise KeyError(
                f"subtask {name!r} from configs/eval.yaml:subtasks has no detector "
                f"in eval/subtasks.py"
            )
        return detector()

    def _detect_drawer_opened(self) -> bool:
        joint = self._det["drawer_joint"]
        low, high = self._scene.joint_range(joint)
        travel = self._scene.joint_qpos(joint) - low
        return travel >= self._det["drawer_opened_travel_fraction"] * (high - low)

    def _detect_cutlery_retrieved(self) -> bool:
        lift = self._det["cutlery_retrieved_lift_m"]
        return any(
            self._scene.body_xpos(name)[2] - start_z >= lift
            for name, start_z in self._cutlery_spawn_z.items()
        )

    def _detect_plate_placed(self) -> bool:
        return self._placed("plate")

    def _detect_cup_placed(self) -> bool:
        return self._placed("cup")

    def _placed(self, role: str) -> bool:
        body = self._objects[role]
        pos = self._scene.body_xpos(body)
        sx, sy, sz = self._spawn_xyz[role]
        moved_xy = math.hypot(pos[0] - sx, pos[1] - sy)
        at_surface_height = abs(pos[2] - sz) <= self._det["on_surface_tolerance_m"]
        at_rest = self._scene.body_linear_speed(body) <= self._det["resting_linear_speed_mps"]
        return (
            moved_xy >= self._det["placed_min_xy_displacement_m"]
            and at_surface_height
            and at_rest
        )

    def _detect_handoff_completed(self) -> bool:
        target = self._objects[self._det["handoff_object"]]
        touching = self._scene.bodies_touching(target)
        if any(b.startswith(self._left_prefix) for b in touching):
            self._handoff_arms_seen.add("left")
        if any(b.startswith(self._right_prefix) for b in touching):
            self._handoff_arms_seen.add("right")
        return {"left", "right"} <= self._handoff_arms_seen

    def _detect_pour_completed(self) -> bool:
        # STUB -- unreachable: __init__ pins this to not_implemented while
        # detection.pour_enabled is false. No simulated liquid exists in
        # sim/assets/dinner_table_dual_so101.xml, so there is nothing to
        # measure. Real logic goes here when a particle/liquid model lands.
        return False
