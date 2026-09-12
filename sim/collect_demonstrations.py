"""
Collect demonstration episodes from the MuJoCo scene into LeRobot dataset
format, ready for `lerobot-train --policy.path=lerobot/smolvla_base ...`.
"""
import os
os.environ["SVT_LOG"] = "0"
import sys
import numpy as np
import mujoco
from lerobot.datasets.lerobot_dataset import LeRobotDataset


sys.path.insert(0, "sim")
sys.path.insert(0, ".")
from randomization import DomainRandomizer  # noqa: E402
from scripted_episode_ik import build_drawer_and_cutlery_episode  # noqa: E402
from ik_demo_utils import (  # noqa: E402
    activate_grasp_connect, deactivate_grasp_connect,
    activate_grasp_weld, deactivate_grasp_weld,
)

MODEL_PATH = "sim/assets/dinner_table_dual_so101.xml"
RANDOMIZATION_CFG = "configs/randomization.yaml"
FPS = 30
RENDER_EVERY_N_STEPS = 2

# Hard timeout for the plate weld release, in control steps past the
# commanded-open step. No site_z gate anymore -- the old site_z<0.80-only
# gate could leave the weld active forever if the arm never got that low
# (observed: dragged the plate back home for the rest of the episode).
RELEASE_TIMEOUT_STEPS = 40
RELEASE_SITE_Z = 0.85

ACTUATOR_NAMES = [
    "left_shoulder_pan", "left_shoulder_lift", "left_elbow_flex",
    "left_wrist_flex", "left_wrist_roll", "left_gripper",
    "right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex",
    "right_wrist_flex", "right_wrist_roll", "right_gripper",
]

CAMERA_SPECS = {
    "overhead": (320, 240),
    "wrist_left": (160, 120),
    "wrist_right": (160, 120),
}


def build_features():
    features = {
        "action": {"dtype": "float32", "shape": (len(ACTUATOR_NAMES),), "names": ACTUATOR_NAMES},
        "observation.state": {"dtype": "float32", "shape": (len(ACTUATOR_NAMES),), "names": ACTUATOR_NAMES},
    }
    for cam_name, (w, h) in CAMERA_SPECS.items():
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"],
        }
    return features


def get_joint_state(model, data):
    state = np.zeros(len(ACTUATOR_NAMES), dtype=np.float32)
    for i, act_name in enumerate(ACTUATOR_NAMES):
        act_id = model.actuator(act_name).id
        joint_id = model.actuator_trnid[act_id, 0]
        qpos_adr = model.jnt_qposadr[joint_id]
        state[i] = data.qpos[qpos_adr]
    return state


def render_all_cameras(renderers, data):
    frames = {}
    for cam_name, renderer in renderers.items():
        renderer.update_scene(data, camera=cam_name)
        frames[cam_name] = renderer.render()
    return frames


def _apply_qpos_to_ctrl(model, data, qpos):
    for act_name in ACTUATOR_NAMES:
        if act_name.endswith("_gripper"):
            continue
        act_id = model.actuator(act_name).id
        joint_id = model.actuator_trnid[act_id, 0]
        qpos_adr = model.jnt_qposadr[joint_id]
        data.ctrl[act_id] = qpos[qpos_adr]


def _find_plate_release_step(gripper_trace):
    prev_left_val = None
    for i, g in enumerate(gripper_trace):
        if g is None:
            continue
        side, val = g
        if side != "left":
            continue
        if prev_left_val is not None and prev_left_val < 0.0 and val > 0.0:
            return i
        prev_left_val = val
    return None


def collect(n_episodes, repo_id, root, instruction, control_decimation=10, seed_offset=0):
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    randomizer = DomainRandomizer(model, RANDOMIZATION_CFG)

    renderers = {name: mujoco.Renderer(model, height=h, width=w)
                 for name, (w, h) in CAMERA_SPECS.items()}

    dataset = LeRobotDataset.create(
        repo_id=repo_id, fps=FPS, features=build_features(), root=root,
        robot_type="dual_so101", use_videos=True,
    )

    left_gripper_bodies = {
        model.body("left_gripper").id,
        model.body("left_moving_jaw_so101_v1").id,
    }
    plate_bid = model.body("plate").id
    left_site_id = model.site("left_gripperframe").id

    seed_rng = np.random.RandomState(seed_offset)
    used_seeds = []

    for ep in range(n_episodes):
        seed = randomizer.sample_training_seed(seed_rng)
        assert not randomizer.is_eval_seed(seed), (
            f"sampled seed {seed} collides with a reserved eval seed"
        )
        used_seeds.append(seed)

        randomizer.reset(data, seed)
        mujoco.mj_forward(model, data)

        print(f"[collect] building IK trajectory for seed={seed}...")
        qpos_trace, gripper_trace, grasp_events = build_drawer_and_cutlery_episode(
            model, data, randomizer
        )

        plate_open_step = _find_plate_release_step(gripper_trace)
        plate_weld_armed = True
        plate_weld_released = False

        print(f"[collect] trajectory: {len(qpos_trace)} steps, "
              f"cutlery-in-drawer={sorted(randomizer.last_in_drawer_items)}, "
              f"plate_release_step={plate_open_step}")

        for step_idx in range(len(qpos_trace)):
            _apply_qpos_to_ctrl(model, data, qpos_trace[step_idx])

            gval = gripper_trace[step_idx]
            if gval is not None:
                side, val = gval
                data.ctrl[model.actuator(f"{side}_gripper").id] = val

            # --- Right arm: drawer handle connect (unchanged, frozen) ----
            if step_idx == grasp_events["activate_step"]:
                handle_world_now = data.geom_xpos[model.geom("drawer_handle").id].copy()
                activate_grasp_connect(model, data, grasp_events["eq_name"],
                                        grasp_events["body1"], grasp_events["body2"],
                                        handle_world_now)
            if step_idx == grasp_events["deactivate_step"]:
                deactivate_grasp_connect(model, data, grasp_events["eq_name"])

            # --- Left arm: plate WELD, NO GATES. Fires on any contact
            # between any left-gripper body and the plate, at any gripper
            # position. Simplified spec: not a clean rim grasp, glue-and-carry.
            if plate_weld_armed:
                for c in range(data.ncon):
                    con = data.contact[c]
                    b1 = int(model.geom_bodyid[con.geom1])
                    b2 = int(model.geom_bodyid[con.geom2])
                    if ((b1 in left_gripper_bodies and b2 == plate_bid)
                            or (b2 in left_gripper_bodies and b1 == plate_bid)):
                        activate_grasp_weld(model, data, "left_grasp_weld",
                                             "left_gripper", "plate")
                        plate_weld_armed = False
                        break

            # --- Left arm: plate WELD release. Commanded-open + hard
            # timeout (RELEASE_TIMEOUT_STEPS), OR site_z < RELEASE_SITE_Z,
            # whichever first. No indefinite wait.
            if (plate_open_step is not None
                    and step_idx >= plate_open_step
                    and not plate_weld_released):
                steps_since_open = step_idx - plate_open_step
                site_z = data.site_xpos[left_site_id][2]
                if steps_since_open >= RELEASE_TIMEOUT_STEPS or site_z < RELEASE_SITE_Z:
                    deactivate_grasp_weld(model, data, "left_grasp_weld")
                    plate_weld_released = True

            for _ in range(control_decimation):
                mujoco.mj_step(model, data)

            if step_idx % RENDER_EVERY_N_STEPS != 0:
                continue

            state = get_joint_state(model, data)
            frames = render_all_cameras(renderers, data)
            frame = {
                "action": data.ctrl.copy()[: len(ACTUATOR_NAMES)].astype(np.float32),
                "observation.state": state,
                "task": instruction,
            }
            for cam_name, img in frames.items():
                frame[f"observation.images.{cam_name}"] = img
            dataset.add_frame(frame)

        # Verify the release actually happened -- never crash, warn instead.
        if not plate_weld_released:
            print(f"[collect] WARNING: weld never released on seed {seed} -- forcing release now.")
            deactivate_grasp_weld(model, data, "left_grasp_weld")

        dataset.save_episode()
        print(f"[collect] episode {ep+1}/{n_episodes} recorded (seed={seed})")

    dataset.finalize()
    print(f"\nWrote {n_episodes} episodes to {root}")
    print(f"Training seeds used (first 10 shown): {used_seeds[:10]}"
          + (" ..." if len(used_seeds) > 10 else ""))
    return dataset


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--repo-id", type=str, default="local/dinner_table_ik")
    parser.add_argument("--root", type=str, default="./data/demo_dataset")
    parser.add_argument("--instruction", type=str,
                         default="open the drawer, retrieve the cutlery, close the drawer, set the table")
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()

    collect(args.episodes, args.repo_id, args.root, args.instruction,
             seed_offset=args.seed_offset)
