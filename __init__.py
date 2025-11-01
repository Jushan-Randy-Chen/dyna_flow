"""
DynaFlow: Dynamics-embedded Flow Matching for Physically Consistent Motion Generation
JAX/MuJoCo implementation based on https://arxiv.org/html/2509.19804v2

This package implements the DynaFlow framework using JAX and MuJoCo MJX for
differentiable physics simulation, enabling end-to-end training of flow matching
models that generate physically consistent robot trajectories.

Includes complete RL training pipeline (PPO) for collecting demonstration data.
"""

__version__ = "0.1.0"

from . import model
from . import rollout
from . import losses
from . import data
from . import utils

# RL components (env_wrapper contains the environment for PPO training)
try:
    from . import env_wrapper
    __all__ = ["model", "rollout", "losses", "data", "utils", "env_wrapper"]
except ImportError:
    # env_wrapper might not be importable if torch is not installed
    __all__ = ["model", "rollout", "losses", "data", "utils"]
