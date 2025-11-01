"""
MuJoCo MJX-based differentiable rollout operator for the Unitree Go2 robot.

This module implements the rollout operator Φ: (x₀, U) → X₁ that maps an initial
state and action sequence to a full state trajectory through differentiable physics.
"""

import jax
import jax.numpy as jnp
from jax import lax
import mujoco
from mujoco import mjx
from typing import Optional, Tuple
import os


class MuJoCoGo2Rollout:
    """
    Differentiable rollout using MuJoCo MJX for the Unitree Go2 quadruped.
    
    This implements the rollout operator Φ from the DynaFlow paper, mapping
    (x₀, U) → X₁ where:
    - x₀: initial state (batch, state_dim=37)
    - U: action sequence (batch, H, action_dim=12)
    - X₁: full state trajectory (batch, H+1, state_dim)
    
    State vector (37D):
        [base_pos(3), base_quat(4), joint_pos(12), base_linvel(3), 
         base_angvel(3), joint_vel(12)]
    
    Action vector (12D):
        PD position targets as residuals from default pose
    """
    
    def __init__(
        self,
        xml_path: str,
        dt: float = 0.02,
        action_scale: float = 0.3,
        default_pose: Optional[jnp.ndarray] = None,
        joint_lowers: Optional[jnp.ndarray] = None,
        joint_uppers: Optional[jnp.ndarray] = None,
    ):
        """
        Args:
            xml_path: Path to MuJoCo XML model file
            dt: Simulation timestep (should match XML)
            action_scale: Scale factor for action residuals
            default_pose: Default joint positions (12,)
            joint_lowers: Joint angle lower bounds (12,)
            joint_uppers: Joint angle upper bounds (12,)
        """
        # Load MuJoCo model
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mjx_model = mjx.put_model(self.mj_model)
        
        self.dt = dt
        self.action_scale = action_scale
        
        # Default configuration
        if default_pose is None:
            default_pose = jnp.array([0.0, 0.9, -1.8] * 4)
        self.default_pose = default_pose
        
        if joint_lowers is None:
            joint_lowers = jnp.array([-0.7, -1.0, -2.2] * 4)
        self.joint_lowers = joint_lowers
        
        if joint_uppers is None:
            joint_uppers = jnp.array([0.52, 2.1, -0.4] * 4)
        self.joint_uppers = joint_uppers
        
        # State dimensions
        self.state_dim = 37  # [pos(3), quat(4), joints(12), linvel(3), angvel(3), joint_vel(12)]
        self.action_dim = 12
        
        # Indices for extracting state components from MJX data
        self.qpos_base_idx = jnp.arange(7)  # base pos + quat
        self.qpos_joint_idx = jnp.arange(7, 19)  # 12 joints
        self.qvel_base_idx = jnp.arange(6)  # base linear + angular vel
        self.qvel_joint_idx = jnp.arange(6, 18)  # 12 joint velocities
    
    def _state_from_mjx_data(self, data: mjx.Data) -> jnp.ndarray:
        """
        Extract state vector from MJX data.
        
        Args:
            data: MJX data object (batched)
        Returns:
            State tensor (batch, 37)
        """
        base_pos = data.qpos[:, :3]  # (batch, 3)
        base_quat = data.qpos[:, 3:7]  # (batch, 4)
        joint_pos = data.qpos[:, 7:19]  # (batch, 12)
        base_linvel = data.qvel[:, :3]  # (batch, 3)
        base_angvel = data.qvel[:, 3:6]  # (batch, 3)
        joint_vel = data.qvel[:, 6:18]  # (batch, 12)
        
        return jnp.concatenate([
            base_pos, base_quat, joint_pos,
            base_linvel, base_angvel, joint_vel
        ], axis=-1)
    
    def _set_mjx_state(self, data: mjx.Data, state: jnp.ndarray) -> mjx.Data:
        """
        Set MJX data from state vector.
        
        Args:
            data: MJX data object to modify
            state: State tensor (batch, 37)
        Returns:
            Updated MJX data
        """
        # Unpack state
        base_pos = state[:, :3]
        base_quat = state[:, 3:7]
        joint_pos = state[:, 7:19]
        base_linvel = state[:, 19:22]
        base_angvel = state[:, 22:25]
        joint_vel = state[:, 25:37]
        
        # Set qpos and qvel
        qpos = jnp.concatenate([base_pos, base_quat, joint_pos], axis=-1)
        qvel = jnp.concatenate([base_linvel, base_angvel, joint_vel], axis=-1)
        
        data = data.replace(qpos=qpos, qvel=qvel)
        return data
    
    def _step(self, data: mjx.Data, action: jnp.ndarray) -> mjx.Data:
        """
        Single simulation step with PD control.
        
        Args:
            data: Current MJX data (batched)
            action: Actions (batch, 12) - residuals from default pose
        Returns:
            Updated MJX data
        """
        # Compute motor targets from action residuals
        motor_targets = self.default_pose + action * self.action_scale
        motor_targets = jnp.clip(motor_targets, self.joint_lowers, self.joint_uppers)
        
        # Set control inputs (MJX uses ctrl field for actuators)
        data = data.replace(ctrl=motor_targets)
        
        # Step physics
        data = mjx.step(self.mjx_model, data)
        
        return data
    
    def rollout_single(self, x0: jnp.ndarray, U: jnp.ndarray) -> jnp.ndarray:
        """
        Rollout a single trajectory (non-batched version for clarity).
        
        Args:
            x0: Initial state (state_dim,)
            U: Action sequence (H, action_dim)
        Returns:
            X: State trajectory (H+1, state_dim)
        """
        # Add batch dimension
        x0_batch = x0[None, :]  # (1, state_dim)
        U_batch = U[None, :, :]  # (1, H, action_dim)
        
        # Call batched version
        X_batch = self(x0_batch, U_batch)  # (1, H+1, state_dim)
        
        # Remove batch dimension
        return X_batch[0]
    
    def __call__(self, x0: jnp.ndarray, U: jnp.ndarray) -> jnp.ndarray:
        """
        Batched rollout operator Φ: (x₀, U) → X₁
        
        Args:
            x0: Initial states (batch, state_dim)
            U: Action sequences (batch, H, action_dim)
        Returns:
            X: State trajectories (batch, H+1, state_dim)
        """
        batch_size, H, action_dim = U.shape
        assert action_dim == self.action_dim
        assert x0.shape == (batch_size, self.state_dim)
        
        # Initialize MJX data with initial states
        data = mjx.make_data(self.mjx_model)
        # Replicate for batch
        data = jax.tree_map(lambda x: jnp.tile(x, (batch_size,) + (1,) * (x.ndim - 1)), data)
        data = self._set_mjx_state(data, x0)
        
        # Storage for trajectory
        X = jnp.zeros((batch_size, H + 1, self.state_dim))
        X = X.at[:, 0, :].set(x0)
        
        # Scan over time to build trajectory
        def scan_fn(data, action_t):
            # Step simulation
            data = self._step(data, action_t)
            # Extract state
            state = self._state_from_mjx_data(data)
            return data, state
        
        # Run scan over horizon
        _, states = lax.scan(scan_fn, data, U.transpose(1, 0, 2))  # (H, batch, state_dim)
        
        # Reshape and concatenate with initial state
        states = states.transpose(1, 0, 2)  # (batch, H, state_dim)
        X = X.at[:, 1:, :].set(states)
        
        return X


def create_go2_rollout(
    xml_path: Optional[str] = None,
    dt: float = 0.02,
    action_scale: float = 0.3
) -> MuJoCoGo2Rollout:
    """
    Factory function to create a Go2 rollout operator.
    
    Args:
        xml_path: Path to MuJoCo XML file (default: use DDAT GO2 model)
        dt: Simulation timestep
        action_scale: Action scaling factor
    Returns:
        MuJoCoGo2Rollout instance
    """
    if xml_path is None:
        # Default to local Unitree_go2 folder
        xml_path = os.path.join(
            os.path.dirname(__file__), "Unitree_go2", "scene_mjx_gym.xml"
        )
        if not os.path.exists(xml_path):
            raise FileNotFoundError(
                f"Default GO2 XML not found at {xml_path}. "
                "Please provide xml_path explicitly."
            )
    
    return MuJoCoGo2Rollout(
        xml_path=xml_path,
        dt=dt,
        action_scale=action_scale
    )
