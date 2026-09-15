from __future__ import annotations

from typing import Any

import numpy as np

from ik_demo_utils import (
    activate_grasp_connect,
    activate_grasp_weld,
    deactivate_grasp_connect,
    deactivate_grasp_weld,
)

class GraspAssist:
    def __init__(
        self,
        model: Any,
        data: Any,
        cfg: dict[str, Any],
    ) -> None:
        self.model = model
        self.data = data
        self.cfg = cfg
        self.enabled = bool(cfg["enabled"])

        self.drawer_active = False
        self.drawer_release_requested_at: int | None = None
        
        self.drawer_was_opened = False
        self.drawer_cycle_complete = False
        self.drawer_closed_hold_steps = 0

        self.plate_active = False
        self.plate_release_armed = False
        self.plate_release_requested_at: int | None = None
        self.plate_open_hold_steps = 0
        
        self.plate_activation_z: float | None = None
        self.plate_was_lifted = False
        self.plate_cycle_complete = False

    def _log(self, message: str) -> None:
        if bool(self.cfg["log_events"]):
            print(f"[grasp] {message}")

    def reset(self) -> None:
        self.drawer_active = False
        self.drawer_release_requested_at = None
        
        self.drawer_was_opened = False
        self.drawer_cycle_complete = False
        self.drawer_closed_hold_steps = 0

        self.plate_active = False
        self.plate_release_armed = False
        self.plate_release_requested_at = None
        self.plate_open_hold_steps = 0
        
        self.plate_activation_z: float | None = None
        self.plate_was_lifted = False
        self.plate_cycle_complete = False

        if not self.enabled:
            return

        drawer_cfg = self.cfg["drawer"]
        plate_cfg = self.cfg["plate"]

        deactivate_grasp_connect(
            self.model,
            self.data,
            drawer_cfg["equality_name"],
        )
        deactivate_grasp_weld(
            self.model,
            self.data,
            plate_cfg["equality_name"],
        )

    def _bodies_in_contact(
        self,
        object_body: str,
        candidate_bodies: list[str],
    ) -> bool:
        object_id = self.model.body(object_body).id
        candidate_ids = {
            self.model.body(name).id
            for name in candidate_bodies
        }

        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            body_1 = int(
                self.model.geom_bodyid[contact.geom1]
            )
            body_2 = int(
                self.model.geom_bodyid[contact.geom2]
            )

            if (
                body_1 == object_id
                and body_2 in candidate_ids
            ) or (
                body_2 == object_id
                and body_1 in candidate_ids
            ):
                return True

        return False
    
    def _geom_touches_bodies(
        self,
        target_geom_name: str,
        candidate_body_names: list[str],
    ) -> bool:
        target_geom_id = self.model.geom(target_geom_name).id
        candidate_body_ids = {
            self.model.body(name).id
            for name in candidate_body_names
        }

        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            geom_1 = int(contact.geom1)
            geom_2 = int(contact.geom2)

            if geom_1 == target_geom_id:
                other_body_id = int(
                    self.model.geom_bodyid[geom_2]
                )
                if other_body_id in candidate_body_ids:
                    return True

            if geom_2 == target_geom_id:
                other_body_id = int(
                    self.model.geom_bodyid[geom_1]
                )
                if other_body_id in candidate_body_ids:
                    return True

        return False

    def _drawer_fraction(self, cfg: dict[str, Any]) -> float:
        joint = self.model.joint(cfg["joint_name"])
        qpos_address = int(
            self.model.jnt_qposadr[joint.id]
        )
        position = float(self.data.qpos[qpos_address])
        low, high = self.model.jnt_range[joint.id]
        travel = float(high - low)

        if travel <= 0:
            raise ValueError(
                f"Joint {cfg['joint_name']!r} has invalid range"
            )

        return float((position - low) / travel)

    def _site_to_body_distance(
        self,
        site_name: str,
        body_name: str,
    ) -> float:
        site_position = self.data.site_xpos[
            self.model.site(site_name).id
        ]
        body_position = self.data.xpos[
            self.model.body(body_name).id
        ]

        return float(
            np.linalg.norm(site_position - body_position)
        )

    def update(
        self,
        action: np.ndarray,
        policy_step: int,
    ) -> None:
        if not self.enabled:
            return

        self._update_drawer(action, policy_step)
        self._update_plate(action, policy_step)

    def _update_drawer(
        self,
        action: np.ndarray,
        policy_step: int,
        ) -> None:
        
        cfg = self.cfg["drawer"]
        drawer_fraction = self._drawer_fraction(cfg)

        if (
            not self.drawer_was_opened
            and drawer_fraction
            >= float(cfg["opened_travel_fraction"])
        ):
            self.drawer_was_opened = True
            self._log(
                f"drawer opening confirmed at step={policy_step}, "
                f"fraction={drawer_fraction:.4f}"
            )

        if self.drawer_cycle_complete:
            return

        gripper_index = int(cfg["gripper_action_index"])
        gripper_value = float(action[gripper_index])

        jaw_position = self.data.xpos[
            self.model.body(cfg["gripper_body"]).id
        ]
        handle_position = self.data.geom_xpos[
            self.model.geom(cfg["handle_geom"]).id
        ]
        distance = float(
            np.linalg.norm(jaw_position - handle_position)
        )

        if not self.drawer_active:
            touching_handle = self._geom_touches_bodies(
                cfg["handle_geom"],
                list(cfg["contact_bodies"]),
            )

            gripper_is_closed = (
                gripper_value <= float(cfg["close_threshold"])
            )

            should_activate = (
                touching_handle
                and (
                    gripper_is_closed
                    or not bool(
                        cfg["require_closed_for_activation"]
                    )
                )
            )

            if should_activate:
                activate_grasp_connect(
                    self.model,
                    self.data,
                    cfg["equality_name"],
                    cfg["gripper_body"],
                    cfg["object_body"],
                    handle_position.copy(),
                )
                self.drawer_active = True
                self.drawer_release_requested_at = None
                self.drawer_closed_hold_steps = 0

                self._log(
                    f"drawer activated at step={policy_step}, "
                    f"source=contact, distance={distance:.4f}"
                )

            return

        if (
            self.drawer_release_requested_at is None
            and gripper_value >= float(cfg["open_threshold"])
        ):
            self.drawer_release_requested_at = policy_step
            self.drawer_closed_hold_steps = 0

            self._log(
                f"drawer release requested at step={policy_step}, "
                f"fraction={drawer_fraction:.4f}"
            )

        if self.drawer_release_requested_at is None:
            return

        closed_enough = (
            drawer_fraction
            <= float(cfg["release_max_travel_fraction"])
        )

        if closed_enough:
            self.drawer_closed_hold_steps += 1
        else:
            self.drawer_closed_hold_steps = 0

        closed_and_held = (
            self.drawer_closed_hold_steps
            >= int(cfg["release_closed_hold_policy_steps"])
        )

        elapsed = (
            policy_step - self.drawer_release_requested_at
        )
        timed_out = (
            elapsed
            >= int(cfg["release_timeout_policy_steps"])
        )

        if not closed_and_held and not timed_out:
            return

        reason = (
            "closed_and_held"
            if closed_and_held
            else "timeout"
        )

        deactivate_grasp_connect(
            self.model,
            self.data,
            cfg["equality_name"],
        )
        self.drawer_active = False
        self.drawer_release_requested_at = None
        self.drawer_closed_hold_steps = 0

        if (
            closed_and_held
            and self.drawer_was_opened
            and bool(
                cfg["prevent_reactivation_after_completed_cycle"]
            )
        ):
            self.drawer_cycle_complete = True

        self._log(
            f"drawer released at step={policy_step}, "
            f"fraction={drawer_fraction:.4f}, "
            f"reason={reason}, "
            f"cycle_complete={self.drawer_cycle_complete}"
        )
    
    
    def _update_plate(
        self,
        action: np.ndarray,
        policy_step: int,
    ) -> None:
        
        cfg = self.cfg["plate"]
        
        if (
            self.plate_cycle_complete
            and bool(cfg["prevent_reactivation_after_release"])
        ):
            return
        
        gripper_index = int(cfg["gripper_action_index"])
        gripper_value = float(action[gripper_index])

        if not self.plate_active:
            touching = self._bodies_in_contact(
                cfg["object_body"],
                list(cfg["contact_bodies"]),
            )
            distance = self._site_to_body_distance(
                cfg["activation_site"],
                cfg["object_body"],
            )

            mode = cfg["activation_mode"]
            proximity = (
                distance
                <= float(cfg["activation_distance_m"])
            )

            if mode == "contact":
                position_matches = touching
            elif mode == "proximity":
                position_matches = proximity
            elif mode == "contact_or_proximity":
                position_matches = touching or proximity
            else:
                raise ValueError(
                    f"Unknown plate activation_mode: {mode!r}"
                )

            closed_enough = (
                gripper_value <= float(cfg["close_threshold"])
            )
            gripper_matches = (
                closed_enough
                or not bool(
                    cfg["require_closed_for_activation"]
                )
            )

            if position_matches and gripper_matches:
                activate_grasp_weld(
                    self.model,
                    self.data,
                    cfg["equality_name"],
                    cfg["gripper_body"],
                    cfg["object_body"],
                )
                self.plate_active = True
                plate_position = self.data.xpos[
                    self.model.body(cfg["object_body"]).id
                ]

                self.plate_activation_z = float(plate_position[2])
                self.plate_was_lifted = False
                self.plate_release_armed = False
                self.plate_release_requested_at = None
                self.plate_open_hold_steps = 0

                source = (
                    "contact"
                    if touching
                    else "proximity"
                )
                self._log(
                    f"plate activated at step={policy_step}, "
                    f"source={source}, distance={distance:.4f}"
                )

        if not self.plate_active:
            return
        
        plate_z = float(
            self.data.xpos[
                self.model.body(cfg["object_body"]).id
             ][2]
        )

        if self.plate_activation_z is None:
            raise RuntimeError(
                "Plate activation height is not initialized"
            )

        lift_delta = plate_z - self.plate_activation_z

        if (
            not self.plate_was_lifted
            and lift_delta
            >= float(cfg["minimum_lift_before_release_m"])
        ):
            self.plate_was_lifted = True
            print(
            f"[grasp] plate lifting confirmed "
            f"at step={policy_step}, "
            f"delta_z={lift_delta:.4f}"
        )

        
        if (
            self.plate_was_lifted
            and gripper_value <= float(cfg["close_threshold"])
        ):
            self.plate_release_armed = True

        if gripper_value >= float(cfg["open_threshold"]):
            self.plate_open_hold_steps += 1
        else:
            self.plate_open_hold_steps = 0

        if (
            self.plate_was_lifted
            and self.plate_release_armed
            and self.plate_release_requested_at is None
            and self.plate_open_hold_steps
            >= int(cfg["release_open_hold_policy_steps"])
        ):
            self.plate_release_requested_at = policy_step
            print(
                f"[grasp] plate release requested "
                f"at step={policy_step}"
            )

        if self.plate_release_requested_at is None:
            return

        elapsed = (
            policy_step - self.plate_release_requested_at
        )
        site_z = float(
            self.data.site_xpos[
                self.model.site(cfg["gripper_site"]).id
            ][2]
        )

        low_enough = (
            site_z <= float(cfg["release_site_z_m"])
        )
        timed_out = (
            elapsed
            >= int(cfg["release_timeout_policy_steps"])
        )

        if low_enough or timed_out:
            reason = (
                "site_height"
                if low_enough
                else "timeout"
            )

            deactivate_grasp_weld(
                self.model,
                self.data,
                cfg["equality_name"],
            )

            self.plate_cycle_complete = bool(
                cfg["prevent_reactivation_after_release"]
            )
            self.plate_active = False
            self.plate_activation_z = None
            self.plate_was_lifted = False
            self.plate_release_armed = False
            self.plate_release_requested_at = None
            self.plate_open_hold_steps = 0

            self._log(
                f"plate released at step={policy_step}, "
                f"site_z={site_z:.4f}, reason={reason}, "
                f"cycle_complete={self.plate_cycle_complete}"
            )
