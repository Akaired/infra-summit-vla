"""
Interactive seed viewer.

Launches MuJoCo's normal interactive viewer (all of MuJoCo's own controls --
pause, camera, reload, etc. -- keep working exactly as usual) with one
addition: `[` and `]` cycle through the seeds in randomization.yaml and
re-apply DomainRandomizer.reset() live, so you can see each seed's scene
without restarting anything.

Run:
    MUJOCO_GL=egl python3 sim/viewer_seeds.py     # if running headless/SSH
    python3 sim/viewer_seeds.py                    # normal desktop use

Controls:
    ]   next seed
    [   previous seed
    (everything else is untouched stock MuJoCo viewer behavior)

Why `[` / `]`:
  Every key in the brief's reserved list (Space, arrows, Backspace, Tab,
  Shift+Tab, R, S, T, M, G, X, W, F1-F5, Esc, Ctrl+A, Ctrl+L, Page Up/Down,
  Enter, N, P) is left completely alone. `[` (GLFW keycode 91) and `]`
  (GLFW keycode 93) are not bound to anything in MuJoCo's stock viewer.
  IMPORTANT: verify this on your actual machine the first time you run it --
  I confirmed the GLFW keycodes match ord('[')/ord(']') exactly (91/93) but
  could not grep MuJoCo's C++ viewer source directly in this environment, so
  press `[`/`]` once on startup and confirm nothing unexpected happens
  before relying on it for a demo recording.
"""
import sys
import mujoco
import mujoco.viewer

MODEL_PATH = "sim/assets/dinner_table_dual_so101.xml"
RANDOMIZATION_CFG = "configs/randomization.yaml"

sys.path.insert(0, "sim")
from randomization import DomainRandomizer  # noqa: E402


class SeedCycler:
    """GUI-independent seed-cycling logic, kept separate from the viewer loop
    so it can be unit-tested without a display (see test_viewer_seeds.py)."""

    def __init__(self, model, data, randomizer):
        self.model = model
        self.data = data
        self.randomizer = randomizer
        self.idx = 0
        self._viewer = None  # bound after mujoco.viewer.launch_passive() returns

    def set_viewer(self, viewer):
        """Called once, right after launch_passive() returns, so apply()
        can clear the viewer's selection/perturbation state on every seed
        switch -- see _clear_perturbation."""
        self._viewer = viewer

    def _clear_perturbation(self):
        """Programmatically do what pressing Esc does (per the reserved-key
        list, Esc's only job is 'clear selection'), by resetting the same
        MjvPerturb fields Esc clears: select/active/active2. This targets the
        reported 'mouse locks until Esc' symptom: teleporting objects via
        reset() while the viewer still holds an active mouse-drag/selection
        referencing old object state is the most likely cause. Best-effort --
        wrapped in try/except so a viewer-internals change on some MuJoCo
        version can never crash the callback."""
        if self._viewer is None:
            return
        try:
            self._viewer.perturb.select = 0
            self._viewer.perturb.active = 0
            self._viewer.perturb.active2 = 0
        except Exception:
            pass

    def print_banner(self, seed):
        line = f"  ACTIVE SEED: {seed}  ( [ = prev | ] = next )  "
        print("\n" + "=" * len(line))
        print(line)
        print("=" * len(line))

    def apply(self, idx):
        idx = idx % len(self.randomizer.seeds)
        self.idx = idx
        seed = self.randomizer.seeds[idx]
        self.randomizer.reset(self.data, seed)
        self._clear_perturbation()
        self.print_banner(seed)
        return seed

    def next(self):
        return self.apply(self.idx + 1)

    def prev(self):
        return self.apply(self.idx - 1)

    def key_callback(self, keycode):
        if keycode == ord('['):
            self.prev()
        elif keycode == ord(']'):
            self.next()


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    randomizer = DomainRandomizer(model, RANDOMIZATION_CFG)
    cycler = SeedCycler(model, data, randomizer)
    cycler.apply(0)  # start on the first seed, not raw MJCF defaults

    with mujoco.viewer.launch_passive(model, data, key_callback=cycler.key_callback) as viewer:
        cycler.set_viewer(viewer)
        print("\nViewer running. Press ] / [ to cycle seeds. Close the window to exit.")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
