"""
Headless test of SeedCycler's logic: wraparound, determinism, and that
key_callback dispatches to the right seed. Does NOT open a window (this
environment has no display) -- run viewer_seeds.py itself on a machine with
a display to confirm the GUI/keyboard side.
"""
import sys
import mujoco

sys.path.insert(0, "sim")
from randomization import DomainRandomizer  # noqa: E402
from viewer_seeds import SeedCycler, MODEL_PATH, RANDOMIZATION_CFG  # noqa: E402


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    randomizer = DomainRandomizer(model, RANDOMIZATION_CFG)
    cycler = SeedCycler(model, data, randomizer)

    n = len(randomizer.seeds)
    ok = True

    # Forward wraparound: next() n times should return to seed index 0.
    cycler.apply(0)
    seen = [cycler.idx]
    for _ in range(n):
        cycler.next()
        seen.append(cycler.idx)
    if seen[-1] != seen[0]:
        print(f"FAIL: forward wraparound broken, {seen}")
        ok = False
    else:
        print(f"OK: forward wraparound after {n} next() calls -> back to idx {seen[-1]}")

    # Backward wraparound.
    cycler.apply(0)
    cycler.prev()
    if cycler.idx != n - 1:
        print(f"FAIL: prev() from idx 0 should wrap to {n-1}, got {cycler.idx}")
        ok = False
    else:
        print(f"OK: prev() from idx 0 wraps to {n - 1}")

    # key_callback dispatch matches direct method calls.
    cycler.apply(0)
    cycler.key_callback(ord(']'))
    expected = 1
    if cycler.idx != expected:
        print(f"FAIL: key_callback(']') expected idx {expected}, got {cycler.idx}")
        ok = False
    else:
        print("OK: key_callback(']') advances one seed")

    cycler.key_callback(ord('['))
    if cycler.idx != 0:
        print(f"FAIL: key_callback('[') expected idx 0, got {cycler.idx}")
        ok = False
    else:
        print("OK: key_callback('[') goes back one seed")

    # Unrelated keys must not change state (e.g. Space's GLFW code is 32).
    before = cycler.idx
    cycler.key_callback(32)  # Space
    if cycler.idx != before:
        print("FAIL: an unrelated keycode changed the seed index")
        ok = False
    else:
        print("OK: unrelated keycodes (e.g. Space) are ignored by this callback")

    # Determinism: re-applying the same seed twice gives identical qpos.
    cycler.apply(3)
    qpos_a = data.qpos.copy()
    cycler.apply(3)
    qpos_b = data.qpos.copy()
    import numpy as np
    if not np.array_equal(qpos_a, qpos_b):
        print("FAIL: re-applying the same seed produced different qpos")
        ok = False
    else:
        print("OK: re-applying the same seed is deterministic")

    print("\n" + ("ALL VIEWER-LOGIC CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
