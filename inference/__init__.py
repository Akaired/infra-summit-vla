"""OpenVINO inference runtime for the bimanual VLA policy.

One module, one job: turn ``(instruction, observation, robot_state)`` into an
action, on whatever Intel device ``configs/inference.yaml`` selects. Timing and
reporting live in ``/eval``; this package never imports ``/sim`` or ``/eval``.

Status: STUB. The real OpenVINO IR is exported by ``/policy`` later; until then
:class:`inference.runtime.InferenceRuntime` delegates to the model-free
``policy.dummy_policy.DummyPolicy`` so that ``/eval`` and ``/inference``
integrate today. See the TODO in :func:`inference.runtime._build_backend`.
"""
