"""
Author: Randy Chen
chenj72@rpi.edu

Brax-based Go2 environment using MJX.

Adapted from the Kaggle notebook: https://www.kaggle.com/code/alexeiplatzer/unitree-go2-mjx-ppo
Adapted from Google DeepMind's Barkour tutorial for Unitree Go2.

This implementation trained a PPO policy for controlling the Unitree Go2 quadruped,
assuming a "joystick" control mode. The command signal consists of [vel_x, vel_y, ang_vel_yaw].
"""

import os
from typing import Any, Dict, List, Sequence
from etils import epath
from ml_collections import config_dict

import jax
from jax import numpy as jp
import numpy as np
import mujoco
from mujoco import mjx

from brax import base
from brax import envs
from brax import math
from brax.base import Motion, Transform
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf



def get_config():
    def get_default_rewards_config():
        default_config = config_dict.ConfigDict(
            dict(
                # The coefficients for all reward terms used for training. All
                # physical quantities are in SI units, if no otherwise specified,
                # i.e. joint positions are in rad, positions are measured in meters,
                # torques in Nm, and time in seconds, and forces in Newtons.
                scales=config_dict.ConfigDict(
                    dict(
                        # Tracking rewards are computed using exp(-delta^2/sigma)
                        # sigma can be a hyperparameters to tune.
                        # Track the base x-y velocity (no z-velocity tracking.)
                        tracking_lin_vel=1.5,
                        # Track the angular velocity along z-axis, i.e. yaw rate.
                        tracking_ang_vel=0.8,
                        # Below are regularization terms, we roughly divide the
                        # terms to base state regularizations, joint
                        # regularizations, and other behavior regularizations.
                        # Penalize the base velocity in z direction, L2 penalty.
                        lin_vel_z=-2.0,
                        # Penalize the base roll and pitch rate. L2 penalty.
                        ang_vel_xy=-0.05,
                        # Penalize non-zero roll and pitch angles. L2 penalty.
                        orientation=-5.0,
                        # L2 regularization of joint torques, |tau|^2.
                        torques=-0.0002,
                        # Penalize the change in the action and encourage smooth
                        # actions. L2 regularization |action - last_action|^2
                        action_rate=-0.01,
                        # Encourage long swing steps.  However, it does not
                        # encourage high clearances.
                        feet_air_time=0.2,
                        # Encourage no motion at zero command, L2 regularization
                        # |q - q_default|^2.
                        stand_still=-0.5,
                        # Early termination penalty.
                        termination=-1.0,
                        # Penalizing foot slipping on the ground.
                        foot_slip=-0.1,
                    )
                ),
                # Tracking reward = exp(-error^2/sigma).
                tracking_sigma=0.25,
            )
        )
        return default_config

    default_config = config_dict.ConfigDict(
        dict(
            rewards=get_default_rewards_config(),
        )
    )

    return default_config


class Go2Env(PipelineEnv):
    """
    Unitree Go2 quadruped environment for Brax/MJX.
    
    Based on Kaggle implementation with DynaFlow's reward structure.
    Trains a joystick policy to control forward/backward/turning velocity.


    This environment describes the Unitree GO 2 robot MuJoCo menagerie from
    https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go2

    ## Action Space
    The action space is a `Box(-1, 1, (12,), float32)`. An action represents the torques applied at the hinge joints.

    | Num | Action                                 | Min  | Max  | Name in xml | Joint | Unit         |
    | --- | ---------------------------------------| ---- | ---- | ----------- | ----- | ------------ |
    | 0   | Desired angle of the front right hip   | -0.7 | 0.52 | FR_hip      | hinge | angle (rad) |
    | 1   | Desired angle of the front right thigh | -1.0 | 2.1  | FR_thigh    | hinge | angle (rad) |
    | 2   | Desired angle of the front right calf  | -2.2 | -0.4 | FR_calf     | hinge | angle (rad) |
    | 3   | Desired angle of the front left hip    | -0.7 | 0.52 | FL_hip      | hinge | angle (rad) |
    | 4   | Desired angle of the front left thigh  | -1.0 | 2.1  | FL_thigh    | hinge | angle (rad) |
    | 5   | Desired angle of the front left calf   | -2.2 | -0.4 | FL_calf     | hinge | angle (rad) |
    | 6   | Desired angle of the rear right hip    | -0.7 | 0.52 | RR_hip      | hinge | angle (rad) |
    | 7   | Desired angle of the rear right thigh  | -1.0 | 2.1  | RR_thigh    | hinge | angle (rad) |
    | 8   | Desired angle of the rear right calf   | -2.2 | -0.4 | RR_calf     | hinge | angle (rad) |
    | 9   | Desired angle of the rear left hip     | -0.7 | 0.52 | RL_hip      | hinge | angle (rad) |
    | 10  | Desired angle of the rear left thigh   | -1.0 | 2.1  | RL_thigh    | hinge | angle (rad) |
    | 11  | Desired angle of the rear left calf    | -2.2 | -0.4 | RL_calf     | hinge | angle (rad) |
   
    
    ## Observation Space

    | Num | Observation                                | Min    | Max    | Name in xml | Joint | Unit                     |
    |-----|--------------------------------------------|--------|--------|-------------|-------|--------------------------|
    | 0   | x-coordinate of the trunk                  | -Inf   | Inf    | trunk       | free  | position (m)             |
    | 1   | y-coordinate of the trunk                  | -Inf   | Inf    | trunk       | free  | position (m)             |
    | 2   | z-coordinate of the trunk                  | -Inf   | Inf    | trunk       | free  | position (m)             |
    | 3   | w-orientation of the trunk (quaternion)    | -Inf   | Inf    | trunk       | free  | angle (rad)              |
    | 4   | x-orientation of the trunk (quaternion)    | -Inf   | Inf    | trunk       | free  | angle (rad)              |
    | 5   | y-orientation of the trunk (quaternion)    | -Inf   | Inf    | trunk       | free  | angle (rad)              |
    | 6   | z-orientation of the trunk (quaternion)    | -Inf   | Inf    | trunk       | free  | angle (rad)              |
    | 7   | angle of the front right hip               | -Inf   | Inf    | FR_hip      | hinge | angle (rad)              |
    | 8   | angle of the front right thigh             | -Inf   | Inf    | FR_thigh    | hinge | angle (rad)              |
    | 9   | angle of the front right calf              | -Inf   | Inf    | FR_calf     | hinge | angle (rad)              |
    | 10  | angle of the front left hip                | -Inf   | Inf    | FL_hip      | hinge | angle (rad)              |
    | 11  | angle of the front left thigh              | -Inf   | Inf    | FL_thigh    | hinge | angle (rad)              |
    | 12  | angle of the front left calf               | -Inf   | Inf    | FL_calf     | hinge | angle (rad)              |
    | 13  | angle of the rear right hip                | -Inf   | Inf    | RR_hip      | hinge | angle (rad)              |
    | 14  | angle of the rear right thigh              | -Inf   | Inf    | RR_thigh    | hinge | angle (rad)              |
    | 15  | angle of the rear right calf               | -Inf   | Inf    | RR_calf     | hinge | angle (rad)              |
    | 16  | angle of the rear left hip                 | -Inf   | Inf    | RL_hip      | hinge | angle (rad)              |
    | 17  | angle of the rear left thigh               | -Inf   | Inf    | RL_thigh    | hinge | angle (rad)              |
    | 18  | angle of the rear left calf                | -Inf   | Inf    | RL_calf     | hinge | angle (rad)              |
    
    | 19  | x-coordinate velocity of the trunk         | -Inf   | Inf    | trunk       | free  | velocity (m/s)           |
    | 20  | y-coordinate velocity of the trunk         | -Inf   | Inf    | trunk       | free  | velocity (m/s)           |
    | 21  | z-coordinate velocity of the trunk         | -Inf   | Inf    | trunk       | free  | velocity (m/s)           |
    | 22  | x-coordinate angular velocity of the trunk | -Inf   | Inf    | trunk       | free  | angular velocity (rad/s) |
    | 23  | y-coordinate angular velocity of the trunk | -Inf   | Inf    | trunk       | free  | angular velocity (rad/s) |
    | 24  | z-coordinate angular velocity of the trunk | -Inf   | Inf    | trunk       | free  | angular velocity (rad/s) |
    | 25  | angular velocity of the front right hip    | -Inf   | Inf    | FR_hip      | hinge | angular velocity (rad/s) |
    | 26  | angular velocity of the front right thigh  | -Inf   | Inf    | FR_thigh    | hinge | angular velocity (rad/s) |
    | 27  | angular velocity of the front right calf   | -Inf   | Inf    | FR_calf     | hinge | angular velocity (rad/s) |
    | 28  | angular velocity of the front left hip     | -Inf   | Inf    | FL_hip      | hinge | angular velocity (rad/s) |
    | 29  | angular velocity of the front left thigh   | -Inf   | Inf    | FL_thigh    | hinge | angular velocity (rad/s) |
    | 30  | angular velocity of the front left calf    | -Inf   | Inf    | FL_calf     | hinge | angular velocity (rad/s) |
    | 31  | angular velocity of the rear right hip     | -Inf   | Inf    | RR_hip      | hinge | angular velocity (rad/s) |
    | 32  | angular velocity of the rear right thigh   | -Inf   | Inf    | RR_thigh    | hinge | angular velocity (rad/s) |
    | 33  | angular velocity of the rear right calf    | -Inf   | Inf    | RR_calf     | hinge | angular velocity (rad/s) |
    | 34  | angular velocity of the rear left hip      | -Inf   | Inf    | RL_hip      | hinge | angular velocity (rad/s) |
    | 35  | angular velocity of the rear left thigh    | -Inf   | Inf    | RL_thigh    | hinge | angular velocity (rad/s) |
    | 36  | angular velocity of the rear left calf     | -Inf   | Inf    | RL_calf     | hinge | angular velocity (rad/s) |
    


    """
    
    def __init__(
        self,
        xml_path: str = None,
        obs_noise: float = 0.05,
        action_scale: float = 0.3,
        kick_vel: float = 0.05,
        scene_file: str = 'go2_mjx_gym.xml',
        **kwargs,
    ):
        # Load MuJoCo model
        if xml_path is None:
            try:
                from dyna_flow.utils import get_default_xml_path
                xml_path = get_default_xml_path()
            except (ModuleNotFoundError, ImportError):
                try:
                    from utils import get_default_xml_path
                    xml_path = get_default_xml_path()
                except (ImportError, FileNotFoundError):
                    import os
                    xml_path = os.path.join(
                        os.path.dirname(__file__), "Unitree_go2", scene_file
                    )
        
        sys = mjcf.load(xml_path)
        self._dt = 0.02  # 50 fps
        sys = sys.tree_replace({'opt.timestep': 0.004})
        
        # Override params for smoother policy (from Kaggle)
        sys = sys.replace(
            dof_damping=sys.dof_damping.at[6:].set(0.5239),
            actuator_gainprm=sys.actuator_gainprm.at[:, 0].set(35.0),
            actuator_biasprm=sys.actuator_biasprm.at[:, 1].set(-35.0),
        )
        
        n_frames = kwargs.pop('n_frames', int(self._dt / sys.opt.timestep))
        super().__init__(sys, backend='mjx', n_frames=n_frames)
        
        self.reward_config = get_config()
        # Allow custom reward scales from kwargs
        for k, v in kwargs.items():
            if k.endswith('_scale'):
                self.reward_config.rewards.scales[k[:-6]] = v
        
        self._torso_idx = mujoco.mj_name2id(
            sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'base'
        )
        self._action_scale = action_scale
        self._obs_noise = obs_noise
        self._kick_vel = kick_vel
        self._init_q = jp.array(sys.mj_model.keyframe('home').qpos)
        self._default_pose = sys.mj_model.keyframe('home').qpos[7:]

        # self._init_q = np.array([0, 0, 0.27, 1, 0, 0, 0, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])
        # self._default_pose = np.array([0.0, 0.9, -1.8] * 4)
        self.lowers = jp.array([-0.7, -1.0, -2.2] * 4)
        self.uppers = jp.array([0.52, 2.1, -0.4] * 4)
        
        # Foot sites
        feet_site = ['FL_foot', 'RL_foot', 'FR_foot', 'RR_foot']
        feet_site_id = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f)
            for f in feet_site
        ]
        assert not any(id_ == -1 for id_ in feet_site_id), 'Site not found.'
        self._feet_site_id = np.array(feet_site_id)
        
        # Lower leg bodies
        lower_leg_body = ['FL_calf', 'RL_calf', 'FR_calf', 'RR_calf']
        lower_leg_body_id = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, l)
            for l in lower_leg_body
        ]
        assert not any(id_ == -1 for id_ in lower_leg_body_id), 'Body not found.'
        self._lower_leg_body_id = np.array(lower_leg_body_id)
        self._foot_radius = 0.0175
        self._nv = sys.nv
    
    def sample_command(self, rng: jax.Array) -> jax.Array:
        """Sample random command: [lin_vel_x, lin_vel_y, ang_vel_yaw]"""
        lin_vel_x = [-0.5, 1.5]  # min max [m/s]
        lin_vel_y = [-0.8, 0.8]  # min max [m/s]
        ang_vel_yaw = [-0.7, 0.7]  # min max [rad/s]
        
        _, key1, key2, key3 = jax.random.split(rng, 4)
        lin_vel_x = jax.random.uniform(
            key1, (1,), minval=lin_vel_x[0], maxval=lin_vel_x[1]
        )
        lin_vel_y = jax.random.uniform(
            key2, (1,), minval=lin_vel_y[0], maxval=lin_vel_y[1]
        )
        ang_vel_yaw = jax.random.uniform(
            key3, (1,), minval=ang_vel_yaw[0], maxval=ang_vel_yaw[1]
        )
        new_cmd = jp.array([lin_vel_x[0], lin_vel_y[0], ang_vel_yaw[0]])
        return new_cmd
    
    def reset(self, rng: jax.Array) -> State:
        """Reset environment to initial state."""
        rng, key = jax.random.split(rng)
        
        pipeline_state = self.pipeline_init(self._init_q, jp.zeros(self._nv))
        
        state_info = {
            'rng': rng,
            'last_act': jp.zeros(12),
            'last_vel': jp.zeros(12),
            'command': self.sample_command(key),
            'last_contact': jp.zeros(4, dtype=bool),
            'feet_air_time': jp.zeros(4),
            'rewards': {k: 0.0 for k in self.reward_config.rewards.scales.keys()},
            'kick': jp.array([0.0, 0.0]),
            'step': 0,
        }
        
        obs = self._get_obs(pipeline_state, state_info)
        reward, done = jp.zeros(2)
        metrics = {'total_dist': 0.0}
        for k in state_info['rewards']:
            metrics[k] = state_info['rewards'][k]
        state = State(pipeline_state, obs, reward, done, metrics, state_info)
        return state
    
    def step(self, state: State, action: jax.Array) -> State:  # pytype: disable=signature-mismatch
        rng, cmd_rng, kick_noise_2 = jax.random.split(state.info['rng'], 3)
        
        # kick
        push_interval = 10
        kick_theta = jax.random.uniform(kick_noise_2, maxval=2 * jp.pi)
        kick = jp.array([jp.cos(kick_theta), jp.sin(kick_theta)])
        kick *= jp.mod(state.info['step'], push_interval) == 0
        qvel = state.pipeline_state.qvel  # pytype: disable=attribute-error
        qvel = qvel.at[:2].set(kick * self._kick_vel + qvel[:2])
        state = state.tree_replace({'pipeline_state.qvel': qvel})

        # physics step
        motor_targets = self._default_pose + action * self._action_scale
        motor_targets = jp.clip(motor_targets, self.lowers, self.uppers)
        pipeline_state = self.pipeline_step(state.pipeline_state, motor_targets)
        x, xd = pipeline_state.x, pipeline_state.xd
        
        # observation data
        obs = self._get_obs(pipeline_state, state.info)
        joint_angles = pipeline_state.q[7:]
        joint_vel = pipeline_state.qd[6:]

        # foot contact data based on z-position
        foot_pos = pipeline_state.site_xpos[self._feet_site_id]  # pytype: disable=attribute-error
        foot_contact_z = foot_pos[:, 2] - self._foot_radius
        contact = foot_contact_z < 1e-3  # a mm or less off the floor
        contact_filt_mm = contact | state.info['last_contact']
        contact_filt_cm = (foot_contact_z < 3e-2) | state.info['last_contact']
        first_contact = (state.info['feet_air_time'] > 0) * contact_filt_mm
        state.info['feet_air_time'] += self.dt

        # done if joint limits are reached or robot is falling
        base_quat = x.rot[self._torso_idx - 1]
        # roll/pitch derived from quaternion (yaw ignored for fall condition)
        w = base_quat[0]
        xq = base_quat[1]
        yq = base_quat[2]
        zq = base_quat[3]

        sinr_cosp = 2.0 * (w * xq + yq * zq)
        cosr_cosp = 1.0 - 2.0 * (xq * xq + yq * yq)
        roll = jp.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * yq - zq * xq)
        pitch = jp.arcsin(jp.clip(sinp, -1.0, 1.0))

        angle_threshold = jp.pi / 18.0  # 10 degrees in radians
        done = (jp.abs(roll) > angle_threshold) | (jp.abs(pitch) > angle_threshold)
        done |= jp.any(joint_angles < self.lowers)
        done |= jp.any(joint_angles > self.uppers)
        done |= pipeline_state.x.pos[self._torso_idx - 1, 2] < 0.18

        # reward
        rewards = {
            'tracking_lin_vel': (
                self._reward_tracking_lin_vel(state.info['command'], x, xd)
            ),
            'tracking_ang_vel': (
                self._reward_tracking_ang_vel(state.info['command'], x, xd)
            ),
            'lin_vel_z': self._reward_lin_vel_z(xd),
            'ang_vel_xy': self._reward_ang_vel_xy(xd),
            'orientation': self._reward_orientation(x),
            'torques': self._reward_torques(pipeline_state.qfrc_actuator),  # pytype: disable=attribute-error
            'action_rate': self._reward_action_rate(action, state.info['last_act']),
            'stand_still': self._reward_stand_still(
                state.info['command'], joint_angles,
            ),
            'feet_air_time': self._reward_feet_air_time(
                state.info['feet_air_time'],
                first_contact,
                state.info['command'],
            ),
            'foot_slip': self._reward_foot_slip(pipeline_state, contact_filt_cm),
            'termination': self._reward_termination(done, state.info['step']),
        }
        rewards = {
            k: v * self.reward_config.rewards.scales[k] for k, v in rewards.items()
        }
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        # state management
        state.info['kick'] = kick
        state.info['last_act'] = action
        state.info['last_vel'] = joint_vel
        state.info['feet_air_time'] *= ~contact_filt_mm
        state.info['last_contact'] = contact
        state.info['rewards'] = rewards
        state.info['step'] += 1
        state.info['rng'] = rng

        # sample new command if more than 500 timesteps achieved
        state.info['command'] = jp.where(
            state.info['step'] > 500,
            self.sample_command(cmd_rng),
            state.info['command'],
        )
        # reset the step counter when done
        state.info['step'] = jp.where(
            done | (state.info['step'] > 500), 0, state.info['step']
        )

        # log total displacement as a proxy metric
        state.metrics['total_dist'] = math.normalize(x.pos[self._torso_idx - 1])[1]
        state.metrics.update(state.info['rewards'])

        done = jp.float32(done)
        state = state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )
        return state

    def _get_obs(
        self,
        pipeline_state: base.State,
        state_info: dict[str, Any],
    ) -> jax.Array:
        inv_torso_rot = math.quat_inv(pipeline_state.x.rot[0])
        
        # Get base velocities in local frame
        local_lin_vel = math.rotate(pipeline_state.xd.vel[0], inv_torso_rot)
        local_ang_vel = math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)

        """
        Full observation space from Unitree RL official repo (48-dim):
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel, #3
                                self.base_ang_vel  * self.obs_scales.ang_vel, #3
                                self.projected_gravity, #3
                                self.commands[:, :3] * self.commands_scale, #3
                                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, #12
                                self.dof_vel * self.obs_scales.dof_vel, #12 
                                self.actions #12
                                ),dim=-1)
        Total: 3 + 3 + 3 + 3 + 12 + 12 + 12 = 48
        """
     
        obs = jp.concatenate([
            local_lin_vel * 2.0,                                  # base linear velocity (3)
            local_ang_vel * 0.25,                                 # base angular velocity (3)
            math.rotate(jp.array([0, 0, -1]), inv_torso_rot),    # projected gravity (3)
            state_info['command'] * jp.array([2.0, 2.0, 0.25]),  # command (3)
            (pipeline_state.q[7:] - self._default_pose) * 1.0,   # motor joint angles (12)
            pipeline_state.qd[6:] * 0.05,                        # motor joint velocities (12)
            state_info['last_act'],                              # last action (12)
        ])
        
        # clip, noise
        obs = jp.clip(obs, -100.0, 100.0) + self._obs_noise * jax.random.uniform(
            state_info['rng'], obs.shape, minval=-1, maxval=1
        )
        
        return obs
    
    def get_full_state(self, pipeline_state: base.State) -> jax.Array:
        """
        Extract the full 37-dimensional state vector.
        
        Returns:
            state (37,): [pos (3), quat (4), joint_pos (12), 
                         lin_vel (3), ang_vel (3), joint_vel (12)]
        """
        # Generalized positions: pos (3) + quat (4) + joint_pos (12) = 19
        pos = pipeline_state.q[:3]          # base position (x, y, z)
        quat = pipeline_state.q[3:7]        # base orientation (quaternion w, x, y, z)
        joint_pos = pipeline_state.q[7:]    # 12 joint positions
        
        # Generalized velocities: lin_vel (3) + ang_vel (3) + joint_vel (12) = 18
        lin_vel = pipeline_state.qd[:3]     # base linear velocity
        ang_vel = pipeline_state.qd[3:6]    # base angular velocity
        joint_vel = pipeline_state.qd[6:]   # 12 joint velocities
        
        # Concatenate all components
        full_state = jp.concatenate([
            pos, quat, joint_pos,      # 3 + 4 + 12 = 19 (positions)
            lin_vel, ang_vel, joint_vel # 3 + 3 + 12 = 18 (velocities)
        ])
        
        return full_state
    
    def get_conditioning_vector(
        self, 
        pipeline_state: base.State, 
        command: jax.Array
    ) -> jax.Array:
        """
        Extract conditioning vector c = [g, v, ω, q, q̇, vcmd_x, vcmd_y, ωcmd_z].
        
        Args:
            pipeline_state: Current pipeline state
            command: Command vector [vcmd_x, vcmd_y, ωcmd_z]
        
        Returns:
            conditioning (33,): [g (3), v (3), ω (3), q (12), q̇ (12)]
                                where all velocities are in local frame
        """
        # Get projected gravity in local frame (3,)
        inv_torso_rot = math.quat_inv(pipeline_state.x.rot[0])
        g = math.rotate(jp.array([0, 0, -1]), inv_torso_rot)
        
        # Get base linear velocity in local frame (3,)
        v = math.rotate(pipeline_state.xd.vel[0], inv_torso_rot)
        
        # Get base angular velocity in local frame (3,)
        ω = math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)
        
        # Get joint positions (12,)
        q = pipeline_state.q[7:]
        
        # Get joint velocities (12,)
        q_dot = pipeline_state.qd[6:]
        
        # Concatenate: g (3) + v (3) + ω (3) + q (12) + q̇ (12) + cmd (3) = 36 
        conditioning = jp.concatenate([
            g,              # projected gravity (3)
            v,              # base linear velocity (3)
            ω,              # base angular velocity (3)
            q,              # joint positions (12)
            q_dot,          # joint velocities (12)
            command,        # velocity commands [vcmd_x, vcmd_y, ωcmd_z] (3)
        ])
        
        return conditioning

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self, xd: Motion) -> jax.Array:
        # Penalize z axis base linear velocity
        return jp.square(xd.vel[0, 2])

    def _reward_ang_vel_xy(self, xd: Motion) -> jax.Array:
        # Penalize xy axes base angular velocity
        return jp.sum(jp.square(xd.ang[0, :2]))

    def _reward_orientation(self, x: Transform) -> jax.Array:
        # Penalize non flat base orientation
        up = jp.array([0.0, 0.0, 1.0])
        rot_up = math.rotate(up, x.rot[0])
        return jp.sum(jp.square(rot_up[:2]))

    def _reward_torques(self, torques: jax.Array) -> jax.Array:
        # Penalize torques
        return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

    def _reward_action_rate(
        self, act: jax.Array, last_act: jax.Array
    ) -> jax.Array:
        # Penalize changes in actions
        return jp.sum(jp.square(act - last_act))

    def _reward_tracking_lin_vel(
        self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of linear velocity commands (xy axes)
        local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
        lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
        lin_vel_reward = jp.exp(
            -lin_vel_error / self.reward_config.rewards.tracking_sigma
        )
        return lin_vel_reward

    def _reward_tracking_ang_vel(
        self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of angular velocity commands (yaw)
        base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
        ang_vel_error = jp.square(commands[2] - base_ang_vel[2])
        return jp.exp(-ang_vel_error / self.reward_config.rewards.tracking_sigma)

    def _reward_feet_air_time(
        self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array
    ) -> jax.Array:
        # Reward air time.
        rew_air_time = jp.sum((air_time - 0.1) * first_contact)
        rew_air_time *= (
            math.normalize(commands[:2])[1] > 0.05
        )  # no reward for zero command
        return rew_air_time

    def _reward_stand_still(
        self,
        commands: jax.Array,
        joint_angles: jax.Array,
    ) -> jax.Array:
        # Penalize motion at zero commands
        return jp.sum(jp.abs(joint_angles - self._default_pose)) * (
            math.normalize(commands[:2])[1] < 0.1
        )

    def _reward_foot_slip(
        self, pipeline_state: base.State, contact_filt: jax.Array
    ) -> jax.Array:
        # get velocities at feet which are offset from lower legs
        # pytype: disable=attribute-error
        pos = pipeline_state.site_xpos[self._feet_site_id]  # feet position
        feet_offset = pos - pipeline_state.xpos[self._lower_leg_body_id]
        # pytype: enable=attribute-error
        offset = base.Transform.create(pos=feet_offset)
        foot_indices = self._lower_leg_body_id - 1  # we got rid of the world body
        foot_vel = offset.vmap().do(pipeline_state.xd.take(foot_indices)).vel

        # Penalize large feet velocity for feet that are in contact with the ground.
        return jp.sum(jp.square(foot_vel[:, :2]) * contact_filt.reshape((-1, 1)))

    def _reward_termination(self, done: jax.Array, step: jax.Array) -> jax.Array:
        return done & (step < 500)

    def render(
        self, trajectory: List[base.State], camera: str | None = None,
        width: int = 240, height: int = 320,
    ) -> Sequence[np.ndarray]:
        camera = camera or 'track'
        return super().render(trajectory, camera=camera, width=width, height=height)


# Register environment
# envs.register_environment('go2', Go2Env)

