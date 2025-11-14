#!/usr/bin/env python3
"""
PPO training script for DynaFlow using Brax/MJX.

This follows the official MuJoCo MJX tutorial approach for massive performance gains.

"""

import os
import functools
from datetime import datetime
import argparse

import jax
from jax import numpy as jp
import matplotlib.pyplot as plt

from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model

# Import our custom environment
from go2_env_brax import Go2Env

# Import all necessary packages

# Supporting
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"  # 0.9 causes too much lag.
import time
import itertools
import functools
from datetime import datetime
from etils import epath
from typing import Any, Dict, Sequence, Tuple, Callable, NamedTuple, Optional, Union, List
from ml_collections import config_dict
import matplotlib.pyplot as plt

# Math
import jax
import jax.numpy as jp
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=100)  # More legible printing from numpy.
from jax import config  # Analytical gradients work much better with double precision.
# config.update("jax_debug_nans", True)
# config.update("jax_enable_x64", True)
# config.update('jax_default_matmul_precision', 'high')
from flax.training import orbax_utils
from flax import struct
from orbax import checkpoint as ocp

# Sim
import mujoco
import mujoco.mjx as mjx

# Brax
from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.mjx.pipeline import _reformat_contact
from brax.training.acme import running_statistics
from brax.io import html, mjcf, model

# Algorithm
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks


# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

def domain_randomize(sys, rng):
  """Randomizes the mjx.Model."""
  @jax.vmap
  def rand(rng):
    _, key = jax.random.split(rng, 2)
    # friction
    friction = jax.random.uniform(key, (1,), minval=0.6, maxval=1.4)
    friction = sys.geom_friction.at[:, 0].set(friction)
    # actuator
    _, key = jax.random.split(key, 2)
    gain_range = (-5, 5)
    param = jax.random.uniform(
        key, (1,), minval=gain_range[0], maxval=gain_range[1]
    ) + sys.actuator_gainprm[:, 0]
    gain = sys.actuator_gainprm.at[:, 0].set(param)
    bias = sys.actuator_biasprm.at[:, 1].set(-param)
    return friction, gain, bias

  friction, gain, bias = rand(rng)

  in_axes = jax.tree_util.tree_map(lambda x: None, sys)
  in_axes = in_axes.tree_replace({
      'geom_friction': 0,
      'actuator_gainprm': 0,
      'actuator_biasprm': 0,
  })

  sys = sys.tree_replace({
      'geom_friction': friction,
      'actuator_gainprm': gain,
      'actuator_biasprm': bias,
  })

  return sys, in_axes




def main():
    parser = argparse.ArgumentParser(description="Train Go2 with Brax PPO")
    parser.add_argument("-e", "--exp_name", type=str, default="ppo_policy")
    parser.add_argument("--num_envs", type=int, default=8192, 
                        help="Number of parallel environments (default: 4096)")
    parser.add_argument("--num_timesteps", type=int, default=100_000_000,
                        help="Total training timesteps (default: 100M)")
    parser.add_argument("--num_evals", type=int, default=10,
                        help="Number of evaluation checkpoints (default: 10)")
    parser.add_argument("--episode_length", type=int, default=1000,
                        help="Episode length in steps (default: 1000)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for training (default: 256)")
    parser.add_argument("--unroll_length", type=int, default=24, #20 vs 24?
                        help="Number of steps to unroll for training (default: 20)")
    parser.add_argument("--num_minibatches", type=int, default=32,
                        help="Number of minibatches (default: 32)")
    parser.add_argument("--num_updates_per_batch", type=int, default=4,
                        help="PPO updates per batch (default: 4)")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--entropy_cost", type=float, default=1e-2,
                        help="Entropy regularization coefficient (default: 1e-2)")
    parser.add_argument("--discounting", type=float, default=0.97,
                        help="Discount factor (default: 0.97)")
    parser.add_argument("--reward_scaling", type=float, default=1.0,
                        help="Reward scaling factor (default: 1.0)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--xml-path", type=str, default=None,
                        help="Path to Go2 XML file (optional)")
    args = parser.parse_args()
    
    # Create log directory (use absolute path)
    log_dir = os.path.abspath(f"logs/{args.exp_name}")
    os.makedirs(log_dir, exist_ok=True)
    
    print("=" * 80)
    print("DynaFlow Training with Brax/MJX")
    print("=" * 80)
    print(f"Experiment: {args.exp_name}")
    print(f"Log directory: {log_dir}")
    print(f"Number of environments: {args.num_envs}")
    print(f"Total timesteps: {args.num_timesteps:,}")
    print(f"Episode length: {args.episode_length}")
    print("=" * 80)
    
    # Register custom environment
    envs.register_environment('go2', Go2Env)
    
    # Create environment
    print("\nCreating Go2 environment...")

    env = envs.get_environment('go2')
    eval_env = envs.get_environment('go2')
    print(f"✓ Environment created")
    print(f"  - Observation space: {env.observation_size}")
    print(f"  - Action space: {env.action_size}")
    print(f"  - Backend: {env.backend}")
    
    # Setup checkpoint saving
    ckpt_path = epath.Path(f'{log_dir}/ckpts')
    ckpt_path.mkdir(parents=True, exist_ok=True)
    
    def policy_params_fn(current_step, make_policy, params):
        # save checkpoints
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        path = ckpt_path / f'{current_step}'
        orbax_checkpointer.save(path, params, force=True, save_args=save_args)
    
    make_networks_factory = functools.partial(
    ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(512, 256, 128))
    
    # Configure training
    print("\nConfiguring PPO training...")
    train_fn = functools.partial(
      ppo.train, 
      num_timesteps=args.num_timesteps, 
      num_evals=args.num_evals,
      reward_scaling=args.reward_scaling, 
      episode_length=args.episode_length, 
      normalize_observations=True,
      action_repeat=1, 
      unroll_length=args.unroll_length, 
      num_minibatches=args.num_minibatches,
      num_updates_per_batch=args.num_updates_per_batch, 
      discounting=args.discounting, 
      learning_rate=args.learning_rate,
      entropy_cost=args.entropy_cost, 
      num_envs=args.num_envs, 
      batch_size=args.batch_size,
      network_factory=make_networks_factory,
      randomization_fn=None, #maybe we turn off domain randomization for now for faster training?
      policy_params_fn=policy_params_fn,
      seed=args.seed)
    
    print(f"✓ Training configured")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Unroll length: {args.unroll_length}")
    print(f"  - Discount factor: {args.discounting}")
    
    # Setup progress tracking
    x_data = []
    y_data = []
    y_std = []
    times = [datetime.now()]
    
    def progress(num_steps, metrics):
        """Progress callback for tracking training."""
        times.append(datetime.now())
        x_data.append(num_steps)
        y_data.append(metrics['eval/episode_reward'])
        y_std.append(metrics['eval/episode_reward_std'])
        
        # Print progress
        elapsed = (times[-1] - times[1]).total_seconds()
        steps_per_sec = num_steps / elapsed if elapsed > 0 else 0
        
        print(f"\nStep {num_steps:,} / {args.num_timesteps:,}")
        print(f"  Reward: {y_data[-1]:.2f} ± {y_std[-1]:.2f}")
        print(f"  Steps/sec: {steps_per_sec:.1f}")
        print(f"  Elapsed: {elapsed/60:.1f} min")
        
        # Save plot
        plt.figure(figsize=(10, 6))
        plt.errorbar(x_data, y_data, yerr=y_std, alpha=0.8)
        plt.xlabel('Training Steps')
        plt.ylabel('Episode Reward')
        plt.title(f'{args.exp_name}: Training Progress')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{log_dir}/training_progress.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Train!
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")
    

    make_inference_fn, params, metrics = train_fn(
        environment=env,
        progress_fn=progress,
        eval_env=eval_env
    )
    
    # Print final statistics
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Time to JIT compile: {(times[1] - times[0]).total_seconds():.1f}s")
    print(f"Time to train: {(times[-1] - times[1]).total_seconds()/60:.1f} min")
    print(f"Final reward: {y_data[-1]:.2f} ± {y_std[-1]:.2f}")
    
    # Save model
    model_path = f'{log_dir}/policy'
    print(f"\nSaving policy to {model_path}...")
    model.save_params(model_path, params)
    print("✓ Policy saved")
    
    # Save training metrics
    import pickle
    with open(f'{log_dir}/metrics.pkl', 'wb') as f:
        pickle.dump({
            'x_data': x_data,
            'y_data': y_data,
            'y_std': y_std,
            'times': times,
            'final_metrics': metrics,
        }, f)
    print(f"✓ Metrics saved to {log_dir}/metrics.pkl")
    
    print("\n" + "=" * 80)
    print("All done! 🎉")
    print("=" * 80)
    print(f"\nTo visualize the trained policy, run:")
    print(f"  python visualize_brax.py --exp_name {args.exp_name}")


if __name__ == "__main__":
    # Enable XLA optimizations
    os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_gpu_triton_gemm_any=True'
    
    main()


"""
Usage Examples:

# Basic training (fast - recommended for RTX 4080):
python train_brax.py --num_envs 4096 --num_timesteps 50_000_000

# Very fast testing/debugging:
python train_brax.py --num_envs 2048 --num_timesteps 10_000_000 --exp_name test_run
"""
