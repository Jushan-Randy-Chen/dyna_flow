#!/usr/bin/env python3
"""
Sample from a trained DynaFlow model to generate robot trajectories.

This performs iterative refinement using the learned flow matching model.
"""

import argparse
import os
from typing import Optional

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import mediapy as media

from model import create_action_predictor, bytes_to_params
from rollout import MuJoCoGo2Rollout
from losses import interpolate_xt, velocity_field


def normalize_quat(x: jnp.ndarray) -> jnp.ndarray:
    """Normalize quaternion component of state vector (indices 3:7)."""
    quat = x[..., 3:7]
    norm = jnp.linalg.norm(quat, axis=-1, keepdims=True).clip(min=1e-6)
    return x.at[..., 3:7].set(quat / norm)


@jax.jit
def euler_step(
    x_t: jnp.ndarray,
    velocity: jnp.ndarray,
    dt: float
) -> jnp.ndarray:
    """Single Euler integration step."""
    return x_t + dt * velocity


def sample_trajectory(
    model_apply,
    params,
    rollout_op: MuJoCoGo2Rollout,
    x0: jnp.ndarray,
    cond: Optional[jnp.ndarray],
    n_steps: int = 50,
    rng: Optional[jax.random.PRNGKey] = None,
) -> jnp.ndarray:
    """
    Sample a trajectory using iterative refinement.
    
    Starting from x0, iteratively refine using the learned velocity field:
    X_{t+dt} = X_t + dt * u_θ(X_t, t)
    
    Args:
        model_apply: Model.apply function
        params: Model parameters
        rollout_op: Rollout operator
        x0: Initial state (batch, state_dim)
        cond: Optional conditioning (batch, cond_dim)
        n_steps: Number of refinement steps
        rng: Random key
    Returns:
        Final trajectory (batch, H+1, state_dim)
    """
    if rng is None:
        rng = random.PRNGKey(0)
    
    batch_size = x0.shape[0]
    H = 10  # Horizon (should match training)
    state_dim = x0.shape[-1]
    
    # Initialize with noise around initial state
    std = 0.1
    noise = random.normal(rng, (batch_size, H+1, state_dim)) * std
    X_t = jnp.repeat(x0[:, None, :], H+1, axis=1) + noise
    X_t = X_t.at[:, 0, :].set(x0)  # Fix initial state
    X_t = normalize_quat(X_t)
    
    dt = 1.0 / n_steps
    
    for i in range(n_steps):
        t = jnp.ones((batch_size, 1)) * (i * dt)
        
        # Predict actions
        U_hat = model_apply(
            params,
            X_t,
            t,
            cond=cond,
            deterministic=True,
        )
        
        # Rollout to get predicted X_1
        X1_hat = rollout_op(x0, U_hat)
        
        # Compute velocity field
        v = velocity_field(X_t, X1_hat, t)
        
        # Euler step
        X_t = euler_step(X_t, v, dt)
        X_t = X_t.at[:, 0, :].set(x0)  # Keep initial state fixed
        X_t = normalize_quat(X_t)
    
    return X_t


def main():
    parser = argparse.ArgumentParser(description="Sample from DynaFlow model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.npz)",
    )
    parser.add_argument(
        "--xml-path",
        type=str,
        default="Unitree_go2/go2_mjx_gym.xml",
        help="Path to MuJoCo XML",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Number of trajectories to generate",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=50,
        help="Number of refinement steps",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/samples.npz",
        help="Output path",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--commands",
        nargs=3,
        type=float,
        default=None,
        help="Command velocities [vx vy vyaw]",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("DynaFlow Trajectory Sampling")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Samples: {args.n_samples}")
    print(f"Steps: {args.n_steps}")
    print("=" * 80)
    
    # Load model
    print("\nLoading model...")
    data = np.load(args.model, allow_pickle=True)
    
    state_dim = int(data['state_dim'])
    action_dim = int(data['action_dim'])
    horizon = int(data['horizon'])
    cond_dim = int(data['cond_dim']) if data['cond_dim'] != -1 else None
    
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Horizon: {horizon}")
    print(f"  Cond dim: {cond_dim}")
    
    # Recreate model architecture
    rng = random.PRNGKey(args.seed)
    model, init_params = create_action_predictor(
        state_dim=state_dim,
        action_dim=action_dim,
        cond_dim=cond_dim,
        rng=rng,
    )
    
    # Load trained parameters
    params = bytes_to_params(data['model'], init_params)
    print("✓ Model loaded")
    
    # Setup rollout
    print("\nInitializing rollout...")
    rollout = MuJoCoGo2Rollout(xml_path=args.xml_path, dt=0.02)
    print("✓ Rollout ready")
    
    # Generate initial states (standing pose)
    print(f"\nGenerating {args.n_samples} trajectories...")
    
    # Default initial state: standing pose
    x0 = jnp.zeros((args.n_samples, state_dim))
    x0 = x0.at[:, 2].set(0.3)  # Base height
    x0 = x0.at[:, 3].set(1.0)  # Quaternion w=1
    # Default joint positions (standing)
    default_joints = jnp.array([0.0, 0.9, -1.8] * 4)  # 12 joints
    x0 = x0.at[:, 7:19].set(default_joints[None, :])
    
    # Optional conditioning
    if args.commands:
        cond = jnp.tile(jnp.array(args.commands)[None, :], (args.n_samples, 1))
        print(f"Using commands: {args.commands}")
    else:
        cond = None if cond_dim is None else jnp.zeros((args.n_samples, cond_dim))
    
    # Sample trajectories
    rng, sample_rng = random.split(rng)
    trajectories = sample_trajectory(
        model_apply=model.apply,
        params=params,
        rollout_op=rollout,
        x0=x0,
        cond=cond,
        n_steps=args.n_steps,
        rng=sample_rng,
    )
    
    print(f"✓ Generated {trajectories.shape[0]} trajectories")
    print(f"  Shape: {trajectories.shape}")
    
    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(
        args.output,
        trajectories=np.array(trajectories),
        cond=np.array(cond) if cond is not None else None,
    )
    print(f"\n✓ Saved to {args.output}")
    
    print("\nDone! 🎉")


if __name__ == "__main__":
    main()


"""
Usage Examples:

# Sample trajectories from trained model:
python sample_flow_matching.py --model logs/dynaflow_jax.npz

# Generate more samples with specific commands:
python sample_flow_matching.py \\
    --model logs/dynaflow_jax.npz \\
    --n-samples 100 \\
    --commands 1.0 0.0 0.0  # Forward velocity

# Higher quality with more refinement steps:
python sample_flow_matching.py \\
    --model logs/dynaflow_jax.npz \\
    --n-steps 100
"""
