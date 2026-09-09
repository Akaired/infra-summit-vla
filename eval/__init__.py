"""Evaluation harness for the bimanual dinner-table task.

Two independent entry points share this package (see eval/README.md):

  * ``eval/run_episodes.py`` -- the 10-seed episode harness (steps MuJoCo).
  * the standalone Intel benchmark -- not yet landed.

The harness talks to the policy only through :class:`eval.policy_interface.Policy`
and to the scene only through :class:`eval.scene.EpisodeScene`; nothing here
reaches into ``/policy`` or ``/inference`` internals.
"""
