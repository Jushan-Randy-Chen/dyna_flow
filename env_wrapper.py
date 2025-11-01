"""
MuJoCo environment wrapper for rsl_rl PPO training.

Implements the VecEnv interface required by rsl_rl for the Unitree Go2 robot.
"""

import torch
import numpy as np
from typing import Tuple, Union, Dict
import mujoco
import mujoco.viewer


def quat_rotate_inverse_np(q, v):
    """
    Rotate vector by inverse of quaternion (numpy version).
    
    Args:
        q: quaternion (4,) [w, x, y, z]
        v: vector (3,)
    
    Returns:
        rotated vector (3,)
    """
    q_w = q[0]
    q_vec = q[1:]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


class Go2MuJoCoEnv:
    """
    Vectorized environment for Unitree Go2 using MuJoCo.
    Compatible with rsl_rl's VecEnv interface.
    """
    
    def __init__(
        self,
        num_envs: int,
        env_cfg: dict,
        obs_cfg: dict,
        reward_cfg: dict,
        command_cfg: dict,
        device: str = 'cpu',
        xml_path: str = None,
    ):
        self.num_envs = num_envs
        self.device = torch.device(device)
        
        # Configuration
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        
        # Load MuJoCo model
        if xml_path is None:
            try:
                from dyna_flow.utils import get_default_xml_path
                xml_path = get_default_xml_path()
            except (ModuleNotFoundError, ImportError):
                # Running locally, use relative import
                try:
                    from utils import get_default_xml_path
                    xml_path = get_default_xml_path()
                except (ImportError, FileNotFoundError):
                    # Fallback to default path
                    import os
                    xml_path = os.path.join(os.path.dirname(__file__), "Unitree_go2", "go2_mjx_gym.xml")
                    if not os.path.exists(xml_path):
                        raise FileNotFoundError(
                            f"Could not find Go2 XML file. Please provide --xml-path argument.\n"
                            f"Expected location: {xml_path}"
                        )
        
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.dt = self.mj_model.opt.timestep
        
        # Action and observation dimensions
        self.num_actions = env_cfg["num_actions"]
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        
        # Episode management
        self.max_episode_length = int(env_cfg["episode_length_s"] / self.dt)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        
        # Default joint positions
        self.default_dof_pos = torch.zeros(self.num_actions, dtype=torch.float, device=self.device)
        for name, angle in env_cfg["default_joint_angles"].items():
            if name in env_cfg["dof_names"]:
                idx = env_cfg["dof_names"].index(name)
                self.default_dof_pos[idx] = angle
        
        # PD controller parameters
        self.kp = env_cfg["kp"]
        self.kd = env_cfg["kd"]
        self.action_scale = env_cfg["action_scale"]
        
        # Initialize MuJoCo data for each environment
        self.mj_datas = [mujoco.MjData(self.mj_model) for _ in range(num_envs)]
        
        # State buffers
        self.obs_buf = torch.zeros(num_envs, self.num_obs, dtype=torch.float, device=self.device)
        self.privileged_obs_buf = None
        self.rew_buf = torch.zeros(num_envs, dtype=torch.float, device=self.device)
        self.reset_buf = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        
        # Robot state
        self.base_pos = torch.zeros(num_envs, 3, dtype=torch.float, device=self.device)
        self.base_quat = torch.zeros(num_envs, 4, dtype=torch.float, device=self.device)
        self.base_lin_vel = torch.zeros(num_envs, 3, dtype=torch.float, device=self.device)
        self.base_ang_vel = torch.zeros(num_envs, 3, dtype=torch.float, device=self.device)
        self.dof_pos = torch.zeros(num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.dof_vel = torch.zeros(num_envs, self.num_actions, dtype=torch.float, device=self.device)
        
        # Commands
        self.num_commands = command_cfg["num_commands"]
        self.commands = torch.zeros(num_envs, self.num_commands, dtype=torch.float, device=self.device)
        
        # Previous actions for action rate penalty
        self.last_actions = torch.zeros(num_envs, self.num_actions, dtype=torch.float, device=self.device)
        
        # Extras
        self.extras = {}
        
        # Initialize all environments
        self.reset(torch.arange(num_envs, device=self.device))
    
    def reset(self, env_ids: Union[list, torch.Tensor, None] = None) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Reset specified environments. If env_ids is None, reset all environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, device=self.device)
        
        for idx in env_ids:
            i = idx.item()
            # Reset MuJoCo state
            mujoco.mj_resetData(self.mj_model, self.mj_datas[i])
            
            # Set initial base pose
            self.mj_datas[i].qpos[:3] = self.env_cfg["base_init_pos"]
            self.mj_datas[i].qpos[3:7] = self.env_cfg["base_init_quat"]
            
            # Set initial joint positions
            self.mj_datas[i].qpos[7:19] = self.default_dof_pos.cpu().numpy()
            
            # Forward kinematics
            mujoco.mj_forward(self.mj_model, self.mj_datas[i])
        
        # Update state buffers
        self._update_state_buffers(env_ids)
        
        # Reset episode tracking
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = False
        self.last_actions[env_ids] = 0.0
        
        # Resample commands
        self._resample_commands(env_ids)
        
        # Update observations
        self._update_observations()
        
        return self.obs_buf, self.privileged_obs_buf
    
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, None], torch.Tensor, torch.Tensor, Dict]:
        """Execute one step of the environment."""
        # Clip actions
        actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        
        # Apply actions to all environments
        for i in range(self.num_envs):
            # Compute PD control
            target_pos = self.default_dof_pos + actions[i] * self.action_scale
            tau = self.kp * (target_pos.cpu().numpy() - self.mj_datas[i].qpos[7:19]) - \
                  self.kd * self.mj_datas[i].qvel[6:18]
            
            # Apply control
            self.mj_datas[i].ctrl[:] = tau
            
            # Step simulation
            mujoco.mj_step(self.mj_model, self.mj_datas[i])
        
        # Update state buffers
        self._update_state_buffers(torch.arange(self.num_envs, device=self.device))
        
        # Compute rewards
        self._compute_rewards(actions)
        
        # Update episode tracking
        self.episode_length_buf += 1
        
        # Check for terminations
        self._check_termination()
        
        # Auto-reset terminated environments
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(reset_env_ids) > 0:
            self.extras['episode'] = {}
            for key in self.episode_sums.keys():
                self.extras['episode'][key] = self.episode_sums[key][reset_env_ids] / self.episode_length_buf[reset_env_ids].float()
            self.reset(reset_env_ids)
        
        # Update observations
        self._update_observations()
        
        # Store actions for next step
        self.last_actions[:] = actions[:]
        
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def get_observations(self) -> torch.Tensor:
        """Return current observations."""
        return self.obs_buf
    
    def get_privileged_observations(self) -> Union[torch.Tensor, None]:
        """Return privileged observations (None for this environment)."""
        return self.privileged_obs_buf
    
    def _update_state_buffers(self, env_ids: torch.Tensor):
        """Update state buffers from MuJoCo data."""
        for idx in env_ids:
            i = idx.item()
            self.base_pos[i] = torch.tensor(self.mj_datas[i].qpos[:3], device=self.device)
            self.base_quat[i] = torch.tensor(self.mj_datas[i].qpos[3:7], device=self.device)
            self.base_lin_vel[i] = torch.tensor(self.mj_datas[i].qvel[:3], device=self.device)
            self.base_ang_vel[i] = torch.tensor(self.mj_datas[i].qvel[3:6], device=self.device)
            self.dof_pos[i] = torch.tensor(self.mj_datas[i].qpos[7:19], device=self.device)
            self.dof_vel[i] = torch.tensor(self.mj_datas[i].qvel[6:18], device=self.device)
    
    def _update_observations(self):
        """Construct observation vector - exact same as go2_env.py."""
        # Observation structure from go2_env.py:
        # [ang_vel(3), projected_gravity(3), commands(5), dof_pos(12), dof_vel(12), actions(12), jump_toggled(1)]
        # Total: 3 + 3 + 5 + 12 + 12 + 12 + 1 = 48
        
        # Get gravity vector in base frame (projected_gravity)
        projected_gravity = self._quat_rotate_inverse(self.base_quat, torch.tensor([0., 0., -1.], device=self.device).repeat(self.num_envs, 1))
        
        # Commands scaling (same as go2_env.py)
        commands_scale = torch.tensor(
            [self.obs_cfg["obs_scales"]["lin_vel"], 
             self.obs_cfg["obs_scales"]["lin_vel"], 
             self.obs_cfg["obs_scales"]["ang_vel"], 
             self.obs_cfg["obs_scales"]["lin_vel"], 
             self.obs_cfg["obs_scales"]["lin_vel"]],
            device=self.device,
            dtype=torch.float,
        )
        
        # Jump toggled buffer (0 for now, can be extended later)
        jump_toggled_buf = torch.zeros((self.num_envs,), device=self.device)
        
        # Concatenate observations exactly as in go2_env.py
        self.obs_buf = torch.cat([
            self.base_ang_vel * self.obs_cfg["obs_scales"]["ang_vel"],  # 3
            projected_gravity,  # 3
            self.commands * commands_scale,  # 5
            (self.dof_pos - self.default_dof_pos) * self.obs_cfg["obs_scales"]["dof_pos"],  # 12
            self.dof_vel * self.obs_cfg["obs_scales"]["dof_vel"],  # 12
            self.last_actions,  # 12 (actions applied)
            (jump_toggled_buf / max(self.reward_cfg.get("jump_reward_steps", 50), 1)).unsqueeze(-1),  # 1
        ], dim=-1)
    
    def _resample_commands(self, env_ids: torch.Tensor):
        """Resample velocity commands for specified environments."""
        # Random commands in specified ranges
        self.commands[env_ids, 0] = torch.rand(len(env_ids), device=self.device) * \
            (self.command_cfg["lin_vel_x_range"][1] - self.command_cfg["lin_vel_x_range"][0]) + \
            self.command_cfg["lin_vel_x_range"][0]
        
        self.commands[env_ids, 1] = torch.rand(len(env_ids), device=self.device) * \
            (self.command_cfg["lin_vel_y_range"][1] - self.command_cfg["lin_vel_y_range"][0]) + \
            self.command_cfg["lin_vel_y_range"][0]
        
        self.commands[env_ids, 2] = torch.rand(len(env_ids), device=self.device) * \
            (self.command_cfg["ang_vel_range"][1] - self.command_cfg["ang_vel_range"][0]) + \
            self.command_cfg["ang_vel_range"][0]
        
        self.commands[env_ids, 3] = torch.rand(len(env_ids), device=self.device) * \
            (self.command_cfg["height_range"][1] - self.command_cfg["height_range"][0]) + \
            self.command_cfg["height_range"][0]
        
        if self.num_commands > 4:
            self.commands[env_ids, 4] = torch.rand(len(env_ids), device=self.device) * \
                (self.command_cfg["jump_range"][1] - self.command_cfg["jump_range"][0]) + \
                self.command_cfg["jump_range"][0]
    
    def _compute_rewards(self, actions: torch.Tensor):
        """Compute reward for current step."""
        # Initialize episode sums if needed
        if not hasattr(self, 'episode_sums'):
            self.episode_sums = {}
        
        # Tracking rewards
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        lin_vel_reward = torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])
        
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_reward = torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])
        
        # Penalties
        lin_vel_z_penalty = torch.square(self.base_lin_vel[:, 2])
        base_height_penalty = torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])
        action_rate_penalty = torch.sum(torch.square(actions - self.last_actions), dim=1)
        similar_to_default_penalty = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        
        # Total reward
        self.rew_buf = (
            self.reward_cfg["reward_scales"]["tracking_lin_vel"] * lin_vel_reward +
            self.reward_cfg["reward_scales"]["tracking_ang_vel"] * ang_vel_reward +
            self.reward_cfg["reward_scales"]["lin_vel_z"] * lin_vel_z_penalty +
            self.reward_cfg["reward_scales"]["base_height"] * base_height_penalty +
            self.reward_cfg["reward_scales"]["action_rate"] * action_rate_penalty +
            self.reward_cfg["reward_scales"]["similar_to_default"] * similar_to_default_penalty
        )
    
    def _check_termination(self):
        """Check termination conditions."""
        # Roll and pitch termination
        roll, pitch, _ = self._get_euler_from_quat(self.base_quat)
        
        roll_term = torch.abs(roll) > np.deg2rad(self.env_cfg["termination_if_roll_greater_than"])
        pitch_term = torch.abs(pitch) > np.deg2rad(self.env_cfg["termination_if_pitch_greater_than"])
        
        # Time limit
        time_out = self.episode_length_buf >= self.max_episode_length
        
        self.reset_buf = roll_term | pitch_term | time_out
    
    @staticmethod
    def _quat_rotate_inverse(q, v):
        """Rotate vector by inverse of quaternion."""
        q_w = q[:, 0]
        q_vec = q[:, 1:]
        a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(v.shape[0], 3, 1)).squeeze(-1) * 2.0
        return a - b + c
    
    @staticmethod
    def _get_euler_from_quat(quat):
        """Convert quaternion to Euler angles (roll, pitch, yaw)."""
        # quat: (N, 4) [w, x, y, z]
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = torch.asin(torch.clamp(sinp, -1, 1))
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
