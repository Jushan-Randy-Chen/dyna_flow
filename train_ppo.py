#!/usr/bin/env python3
"""
PPO training script for DynaFlow using rsl_rl.

Uses the same PPO parameters and training configuration as go2_train.py
from the quadrupeds_locomotion project.
"""

import os
import sys
import argparse
import pickle
import shutil

from rsl_rl.runners import OnPolicyRunner

from env_wrapper import Go2MuJoCoEnv


def get_train_cfg(exp_name, max_iterations, num_learning_epochs=5, num_steps_per_env=24):
    """
    Get training configuration - exact same as go2_train.py
    
    Args:
        exp_name: Experiment name
        max_iterations: Number of training iterations
        num_learning_epochs: Number of epochs to train on each batch (default: 5)
        num_steps_per_env: Steps to collect per environment per iteration (default: 24)
    """
    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": num_learning_epochs,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": num_steps_per_env,
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 100,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    """
    Get environment configurations - exact same as go2_train.py
    """
    env_cfg = {
        "num_actions": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "dof_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        # PD
        "kp": 20.0,
        "kd": 0.5,
        # termination
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_pitch_greater_than": 10,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 10.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.3,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 48,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "jump_upward_velocity": 1.2,  
        "jump_reward_steps": 50,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
            # "jump": 4.0,
            "jump_height_tracking": 0.5,
            "jump_height_achievement": 10,
            "jump_speed": 1.0,
            "jump_landing": 0.08,
        },
    }
    command_cfg = {
        "num_commands": 5,  # [lin_vel_x, lin_vel_y, ang_vel, height, jump]
        "lin_vel_x_range": [-1.0, 2.0],
        "lin_vel_y_range": [-0.5, 0.5],
        "ang_vel_range": [-0.6, 0.6],
        "height_range": [0.2, 0.4],
        "jump_range": [0.5, 1.5],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-ppo-dynaflow")
    parser.add_argument("-B", "--num_envs", type=int, default=2048)
    parser.add_argument("--max_iterations", type=int, default=100)
    parser.add_argument("--num_learning_epochs", type=int, default=5, 
                        help="Number of epochs to train on each batch (reduce to 3 for faster training)")
    parser.add_argument("--num_steps_per_env", type=int, default=24,
                        help="Steps to collect per environment per iteration (increase to 48 for better sample efficiency)")
    parser.add_argument("--device", type=str, default="cuda:0", help="device to use: 'cpu' or 'cuda:0'")
    parser.add_argument("--xml-path", type=str, default=None, help="Path to MuJoCo XML file")
    args = parser.parse_args()
    
    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations, 
                               args.num_learning_epochs, args.num_steps_per_env)

    # Clean up old logs if they exist
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Create environment
    print(f"Creating {args.num_envs} environments...")
    env = Go2MuJoCoEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        device=args.device,
        xml_path=args.xml_path,
    )

    # Create PPO runner
    print("Creating PPO runner...")
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=args.device)

    # Save configuration
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    # Train
    print(f"Starting training for {args.max_iterations} iterations...")
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    
    print(f"\nTraining complete! Checkpoints saved to {log_dir}")


if __name__ == "__main__":
    main()


"""
Usage examples:

# Basic training with default settings
python train_ppo.py

# Faster training (recommended for RTX 4080 - ~3-4 hours instead of 14 hours):
python train_ppo.py --num_envs 2048 --num_learning_epochs 3 --num_steps_per_env 48 --max_iterations 500

# Very fast training for testing/debugging (~1 hour):
python train_ppo.py --num_envs 1024 --num_learning_epochs 2 --num_steps_per_env 64 --max_iterations 200

# Training with custom settings
python train_ppo.py --exp_name my_experiment --num_envs 2048 --max_iterations 5000

# Training on CPU
python train_ppo.py --device cpu --num_envs 512

# With custom XML path
python train_ppo.py --xml-path /path/to/custom/go2.xml
"""
